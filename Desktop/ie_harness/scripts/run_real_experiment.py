"""真实 benchmark 上的 CRSI-Harness 闭环实验。

用法:
    python3 scripts/run_real_experiment.py [--rounds 2] [--fresh]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ie_harness.pipeline import HarnessPipeline
from ie_harness.sim.memory import SIMMemory
from ie_harness.runner import CRSIRunner

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG = os.path.join(ROOT, "configs", "harness_default.yaml")
SIM_PATH = os.path.join(ROOT, "sim_store", "sim_real.json")
REAL = os.path.join(ROOT, "data", "real")
RESULTS = os.path.join(ROOT, "results", "real")


def load_samples(manifest: dict) -> dict[str, list[dict]]:
    samples: dict[str, list[dict]] = {}
    for task, items in manifest.items():
        lst = []
        for it in items:
            gt = np.asarray(Image.open(os.path.join(REAL, task, "gt", it["gt"])).convert("RGB"))
            inp = np.asarray(Image.open(os.path.join(REAL, task, "input", it["input"])).convert("RGB"))
            lst.append({"name": it["name"], "gt": gt, "degraded": inp, "task": task})
        samples[task] = lst
    return samples


def _pick_representative(per_image: list[dict], k: int = 3) -> list[int]:
    order = sorted(range(len(per_image)), key=lambda i: per_image[i]["psnr"])
    n = len(order)
    picks = [order[0], order[n // 2], order[-1]]
    return picks[:k]


def make_comparison(task: str, samples: list[dict], base_pipe: HarnessPipeline,
                    best_pipe: HarnessPipeline, per_image: list[dict], out_path: str) -> None:
    picks = _pick_representative(per_image)
    cols = ["GT", "Input", "Baseline", "Evolved"]
    cell = 224
    label_h = 22
    canvas = Image.new("RGB", (cell * len(cols), (cell + label_h) * len(picks)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for r, idx in enumerate(picks):
        s = samples[idx]
        outs = [s["gt"], s["degraded"], base_pipe.run(s["degraded"]), best_pipe.run(s["degraded"])]
        for c, (label, arr) in enumerate(zip(cols, outs)):
            img = Image.fromarray(arr).resize((cell, cell))
            y = r * (cell + label_h)
            canvas.paste(img, (c * cell, y + label_h))
            draw.text((c * cell + 6, y + 4), f"{label} | {s['name']}", fill=(0, 0, 0))
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.fresh and os.path.exists(SIM_PATH):
        os.remove(SIM_PATH)

    with open(CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(REAL, "manifest_real.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = load_samples(manifest)
    sim = SIMMemory(SIM_PATH)

    runner = CRSIRunner(config, sim, samples, rounds=args.rounds, hard_frac=0.5, candidate_limit=8)
    report = runner.run()

    rows = []
    for task, rep in report["tasks"].items():
        base_pipe = HarnessPipeline(config["tasks"][task]["baseline"])
        best_pipe = HarnessPipeline(rep["best_config"])
        make_comparison(task, samples[task], base_pipe, best_pipe, rep["per_image"],
                        os.path.join(RESULTS, f"comparison_{task}.png"))
        for variant, pipe, agg in [
            ("baseline", base_pipe, rep["baseline_metrics"]),
            ("evolved", best_pipe, rep["best_metrics"]),
        ]:
            rows.append({
                "task": task, "variant": variant, "backend": pipe.backend_name,
                "psnr_mean": round(agg["psnr"]["mean"], 3), "psnr_std": round(agg["psnr"]["std"], 3),
                "ssim_mean": round(agg["ssim"]["mean"], 3), "ssim_std": round(agg["ssim"]["std"], 3),
                "shift_mean": round(agg["channel_shift"]["mean"], 3),
                "shift_std": round(agg["channel_shift"]["std"], 3),
            })

    csv_path = os.path.join(RESULTS, "metrics_real.csv")
    fields = ["task", "variant", "backend", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
              "shift_mean", "shift_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(RESULTS, "report_real.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("=== SIM 统计:", report["sim_stats"])
    for task, rep in report["tasks"].items():
        b, e = rep["baseline_metrics"], rep["best_metrics"]
        print(f"[{task}] {rep['best_tag']} {rep['best_config']['backend']} | "
              f"PSNR {b['psnr']['mean']:.2f}±{b['psnr']['std']:.2f} -> {e['psnr']['mean']:.2f}±{e['psnr']['std']:.2f} | "
              f"SSIM {b['ssim']['mean']:.3f} -> {e['ssim']['mean']:.3f} | "
              f"shift {b['channel_shift']['mean']:.2f} -> {e['channel_shift']['mean']:.2f}")
    print(f"结果已保存: {RESULTS}")


if __name__ == "__main__":
    main()
