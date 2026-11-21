"""评测指标集合。

成对指标（有 GT）: PSNR, SSIM, 通道均值色偏 (mean channel shift)。
无参考代理指标: 灰度对比度 (std), 简化 UIQM（水下场景）。

说明: CPU 环境未引入 LPIPS/VGG 权重；主观一致性后续可由 CLIP-IQA 补充。
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def psnr_gt(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(psnr(gt, pred, data_range=255))


def ssim_gt(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(ssim(gt, pred, channel_axis=2, data_range=255))


def channel_shift(pred: np.ndarray, gt: np.ndarray) -> float:
    """预测与 GT 的逐通道均值绝对差（色偏代理），越小越好。"""
    pm = pred.reshape(-1, 3).mean(axis=0)
    gm = gt.reshape(-1, 3).mean(axis=0)
    return float(np.abs(pm - gm).mean())


def contrast_std(img: np.ndarray) -> float:
    gray = img.mean(axis=2)
    return float(gray.std())


def _uicm(img: np.ndarray) -> float:
    r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(-np.sqrt(rg ** 2 + yb ** 2).mean() + 0.3 * np.sqrt(rg.std() ** 2 + yb.std() ** 2))


def _uism(img: np.ndarray) -> float:
    gray = img.mean(axis=2).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float(gx.mean() + gy.mean())


def _uiconm(img: np.ndarray) -> float:
    gray = img.mean(axis=2).astype(np.float32)
    return float(gray.std() / (gray.mean() + 1e-6))


def uiqm(img: np.ndarray) -> float:
    """简化 UIQM = 0.2821*UICM + 0.2953*UISM + 0.3665*UICONM（归一化前的代理值）。"""
    return float(0.2821 * _uicm(img) + 0.2953 * _uism(img) + 0.3665 * _uiconm(img))


def evaluate_paired(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "psnr": psnr_gt(pred, gt),
        "ssim": ssim_gt(pred, gt),
        "channel_shift": channel_shift(pred, gt),
        "contrast_std": contrast_std(pred),
    }


def evaluate_unpaired(img: np.ndarray, task: str) -> dict:
    out = {"contrast_std": contrast_std(img)}
    if task == "underwater":
        out["uiqm"] = uiqm(img)
    return out
