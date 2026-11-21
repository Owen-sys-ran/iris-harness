"""传统（无权重）增强后端集合。

这些算子不依赖训练权重，参数本身就是 Harness 可进化的对象：
白平衡强度、CLAHE clip、Retinex 尺度、滤波孔径/滤波强度等。
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import EnhancerBackend


class GrayWorldWBBackend(EnhancerBackend):
    """灰度世界白平衡，校正色偏（水下 / 偏色场景可用作后端或预处理）。"""

    name = "grayworld"
    task = "wb"

    def enhance(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        mean_rgb = out.reshape(-1, 3).mean(axis=0)
        gray = mean_rgb.mean()
        gain = gray / np.clip(mean_rgb, 1e-3, None)
        strength = float(self.config.get("strength", 1.0))
        gain = 1.0 + (gain - 1.0) * strength
        out = out * gain[None, None, :]
        return self.to_uint8(out)


class CLAHEBackend(EnhancerBackend):
    """Lab 空间 L 通道自适应直方图均衡，提升局部对比度。"""

    name = "clahe"
    task = "contrast"

    def enhance(self, img: np.ndarray) -> np.ndarray:
        clip = float(self.config.get("clip", 2.0))
        tile = int(self.config.get("tile", 8))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


class MSRBackend(EnhancerBackend):
    """多尺度 Retinex，动态范围压缩 / 低光与雾感增强的传统基线。"""

    name = "msr"
    task = "contrast"

    def enhance(self, img: np.ndarray) -> np.ndarray:
        sigmas = self.config.get("sigmas", [15, 80, 250])
        imgf = img.astype(np.float32) + 1.0
        retinex = np.zeros_like(imgf)
        for s in sigmas:
            blur = cv2.GaussianBlur(imgf, (0, 0), s)
            retinex += np.log(imgf) - np.log(blur + 1e-6)
        retinex /= len(sigmas)
        # 百分号拉伸到 0-255
        out = np.zeros_like(retinex)
        for c in range(3):
            lo, hi = np.percentile(retinex[..., c], (1, 99))
            out[..., c] = np.clip((retinex[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
        return self.to_uint8(out * 255.0)


class BilateralDenoiseBackend(EnhancerBackend):
    """双边滤波去噪：保边平滑。"""

    name = "bilateral"
    task = "denoise"

    def enhance(self, img: np.ndarray) -> np.ndarray:
        d = int(self.config.get("d", 5))
        sigma_color = float(self.config.get("sigma_color", 40))
        sigma_space = float(self.config.get("sigma_space", 40))
        out = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return out


class NLMDenoiseBackend(EnhancerBackend):
    """非局部均值去噪，强度更高但更慢。"""

    name = "nlm"
    task = "denoise"

    def enhance(self, img: np.ndarray) -> np.ndarray:
        h = float(self.config.get("h", 10))
        hcolor = float(self.config.get("h_color", 10))
        out = cv2.fastNlMeansDenoisingColored(img, None, h, hcolor, 7, 21)
        return out
