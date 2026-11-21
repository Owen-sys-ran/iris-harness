"""配置化增强流水线（Harness 本体）。

流水线分段: 预处理 -> 冻结后端 -> 后处理。
Harness 进化的对象就是这里的配置项；后端权重始终冻结。

可进化部件:
  preprocess : white_balance / gamma / stretch(百分号对比度拉伸)
  backend    : 后端选择 + backend_config 内部参数
  postprocess: gain / gamma / channel_gain(独立通道校色) / sharpen(Unsharp Mask)
"""
from __future__ import annotations

import copy
import cv2
import numpy as np

from .backends.registry import build_backend
from .backends.classical import GrayWorldWBBackend


def _gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-6:
        return img
    f = img.astype(np.float32) / 255.0
    return np.clip(np.power(f, gamma) * 255.0, 0, 255).astype(np.uint8)


def _percentile_stretch(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    out = img.astype(np.float32)
    for c in range(3):
        p_lo, p_hi = np.percentile(out[..., c], (lo, hi))
        if p_hi - p_lo < 1e-3:
            continue
        out[..., c] = np.clip((out[..., c] - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def _channel_gain(img: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    out = img.astype(np.float32) * np.array(gains)[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    if amount is None or amount <= 1e-6:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    out = img.astype(np.float32) + amount * (img.astype(np.float32) - blur.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


class HarnessPipeline:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.backend = build_backend(
            self.config["backend"], self.config.get("backend_config", {})
        )

    # ---- 配置访问 ----
    @property
    def name(self) -> str:
        return self.config.get("name", "unnamed")

    @property
    def backend_name(self) -> str:
        return self.config["backend"]

    def get(self, stage: str, key: str, default=None):
        return self.config.get(stage, {}).get(key, default)

    def edited(self, edit: dict) -> "HarnessPipeline":
        """返回应用一条 harness 编辑后的新流水线（原流水线不变）。"""
        if edit["stage"] == "combo":
            pipe = self
            for sub in edit["new"]:
                pipe = pipe.edited(sub)
            return pipe
        cfg = copy.deepcopy(self.config)
        stage, key, new = edit["stage"], edit["key"], edit["new"]
        if stage == "backend":
            cfg["backend"] = new
        elif stage == "backend_config":
            cfg.setdefault("backend_config", {})[key] = new
        else:
            cfg.setdefault(stage, {})[key] = new
        return HarnessPipeline(cfg)

    # ---- 推理 ----
    def run(self, img: np.ndarray) -> np.ndarray:
        x = img
        # 预处理
        if self.get("preprocess", "white_balance") == "grayworld":
            x = GrayWorldWBBackend({}).enhance(x)
        pre_gamma = self.get("preprocess", "gamma")
        if pre_gamma is not None:
            x = _gamma_correct(x, float(pre_gamma))
        stretch = self.get("preprocess", "stretch")
        if stretch is not None:
            x = _percentile_stretch(x, float(stretch[0]), float(stretch[1]))
        # 冻结后端（可选水平翻转 TTA 融合）
        use_tta = bool(self.get("postprocess", "tta_flip", False))
        out1 = self.backend.enhance(x)
        if use_tta:
            out2 = np.fliplr(self.backend.enhance(np.fliplr(x)))
            x = np.clip((out1.astype(np.float32) + out2.astype(np.float32)) / 2.0, 0, 255).astype(np.uint8)
        else:
            x = out1
        # 后处理
        gain = float(self.get("postprocess", "gain", 1.0))
        if abs(gain - 1.0) > 1e-6:
            x = np.clip(x.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        post_gamma = self.get("postprocess", "gamma")
        if post_gamma is not None:
            x = _gamma_correct(x, float(post_gamma))
        cg = self.get("postprocess", "channel_gain")
        if cg is not None:
            x = _channel_gain(x, tuple(float(v) for v in cg))
        sharpen = self.get("postprocess", "sharpen")
        if sharpen is not None:
            x = _sharpen(x, float(sharpen))
        return x
