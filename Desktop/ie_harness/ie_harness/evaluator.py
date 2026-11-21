"""Evaluator: 批量评测 + 聚合(mean±std) + 多指标联合判定 + 根因统计。

判定采用归一化综合评分，避免单一 PSNR 主导:
  score = 0.5*ΔPSNR/10 + 0.3*ΔSSIM/0.2 - 0.2*Δshift/20
硬退化保护(任一触发即 regressed): PSNR ≤ -0.3dB / SSIM ≤ -0.01 / 色偏 ≥ +3.0
  improved: score ≥ 0.05 且无硬退化
  regressed: 触发任一硬退化
  neutral  : 其余
"""
from __future__ import annotations

import numpy as np

from .metrics import evaluate_paired, evaluate_unpaired


def _agg(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def composite(cand: dict, base: dict) -> dict:
    """返回综合分与各指标增量。"""
    dpsnr = cand["psnr"]["mean"] - base["psnr"]["mean"]
    dssim = cand["ssim"]["mean"] - base["ssim"]["mean"]
    dshift = cand["channel_shift"]["mean"] - base["channel_shift"]["mean"]
    score = 0.5 * dpsnr / 10.0 + 0.3 * dssim / 0.2 - 0.2 * dshift / 20.0
    return {"dpsnr": dpsnr, "dssim": dssim, "dshift": dshift, "score": float(score)}


class Evaluator:
    def __init__(self, task: str):
        self.task = task

    def evaluate_pipeline(self, pipeline, samples: list[dict]) -> dict:
        per_image = []
        out_vals, rb_ratios, mean_vals, contrasts = [], [], [], []
        for s in samples:
            out = pipeline.run(s["degraded"])
            m = evaluate_paired(out, s["gt"])
            m.update(evaluate_unpaired(out, self.task))
            m["name"] = s["name"]
            per_image.append(m)

            gray = out.mean(axis=2)
            rgb = out.reshape(-1, 3).mean(axis=0)
            mean_vals.append(float(gray.mean()))
            contrasts.append(float(gray.std()))
            rb_ratios.append(float(rgb[0] / np.clip(rgb[2], 1e-3, None)))
            out_vals.append(out)

        agg = {
            "psnr": _agg([m["psnr"] for m in per_image]),
            "ssim": _agg([m["ssim"] for m in per_image]),
            "channel_shift": _agg([m["channel_shift"] for m in per_image]),
            "contrast_std": _agg([m["contrast_std"] for m in per_image]),
            # 根因统计（取均值），供 Proposer 排序
            "mean_val": float(np.mean(mean_vals)),
            "contrast": float(np.mean(contrasts)),
            "rb_ratio": float(np.mean(rb_ratios)),
        }
        if "uiqm" in per_image[0]:
            agg["uiqm"] = _agg([m["uiqm"] for m in per_image])
        # 兼容旧接口: 直接暴露标量均值
        agg["psnr_mean"] = agg["psnr"]["mean"]
        agg["ssim_mean"] = agg["ssim"]["mean"]
        agg["channel_shift_mean"] = agg["channel_shift"]["mean"]
        return {"aggregate": agg, "per_image": per_image}

    @staticmethod
    def judge(cand: dict, base: dict) -> str:
        delta = composite(cand, base)
        dpsnr, dssim, dshift = delta["dpsnr"], delta["dssim"], delta["dshift"]
        # 硬退化保护
        if dpsnr <= -0.3 or dssim <= -0.01 or dshift >= 3.0:
            return "regressed"
        if delta["score"] >= 0.05:
            return "improved"
        return "neutral"
