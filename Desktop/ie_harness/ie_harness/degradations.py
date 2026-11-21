"""成对退化合成：由清晰 GT 生成雾 / 低光 / 水下色偏 / 噪声样例。

固定随机种子，保证 baseline 与候选 harness 在同一批退化图上对比。
"""
from __future__ import annotations

import numpy as np
import cv2


def _gauss_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255)


def add_haze(img: np.ndarray, beta: float = 0.55, A: float = 220.0,
             seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # 平滑深度图：渐变 + 低频扰动
    depth = np.linspace(0.3, 1.0, w)[None, :] * np.linspace(0.4, 1.0, h)[:, None]
    depth = depth + rng.normal(0, 0.08, (h, w))
    depth = np.clip(depth, 0.1, 1.5)
    t = np.exp(-beta * depth)[..., None]
    imgf = img.astype(np.float32)
    out = imgf * t + A * (1 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_lowlight(img: np.ndarray, scale: float = 0.25, noise_sigma: float = 6.0,
                 seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * scale
    out = _gauss_noise(out, noise_sigma, rng)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_underwater(img: np.ndarray, r_decay: float = 0.55, b_decay: float = 0.92,
                   veil: float = 0.18, blur: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    out[..., 0] *= r_decay
    out[..., 1] *= 0.95
    out[..., 2] *= b_decay
    # 蓝绿色雾罩
    out[..., 0] += veil * 60
    out[..., 1] += veil * 90
    out[..., 2] += veil * 110
    out = cv2.GaussianBlur(out, (blur, blur), 0)
    out = _gauss_noise(out, 4.0, rng)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(img: np.ndarray, sigma: float = 25.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = _gauss_noise(img.astype(np.float32), sigma, rng)
    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADERS = {
    "dehaze": add_haze,
    "lowlight": add_lowlight,
    "underwater": add_underwater,
    "denoise": add_noise,
}
