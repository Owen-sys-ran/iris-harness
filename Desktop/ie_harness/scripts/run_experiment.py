"""CRSI-Harness 闭环实验主脚本。

用法:
    python3 scripts/run_experiment.py [--rounds 2] [--fresh]
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
SIM_PATH = os.path.join(ROOT, "sim_store", "sim.json")
RESULTS = os.path.join(ROOT, "results")


def load_samples(manifest: dict) -> dict[str, list[dict]]:
    samples: dict[str, list[dict]] = {}
    for task, items in manifest.items():
        lst = []
        for it in items:
            gt = np.asarray(Image.open(os.path.join(ROOT, "data", "gt", f"{it['gt']}.png")).convert("RGB"))
            deg = np.asarray(Image.open(os.path.join(ROOT, "data", "degraded", task, f"{it['name']}.png")).convert("RGB"))
            lst.append({"name": it["name"], "gt": gt, "degraded": deg, "task": task})
        samples[task] = lst
    return samples


def make_comparison(task: str, samples: list[dict], base_pipe: HarnessPipeline,
                    best_pipe: HarnessPipeline, out_path: str) -> None:
    cols = ["GT", "Degraded", "Baseline", "Evolved"]
    cell = 256
    label_h = 24
    rows = len(samples)
    canvas = Image.new("RGB", (cell * len(cols), (cell + label_h) * rows), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for r, s in enumerate(samples):
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
    ap.add_argument("--fresh", action="store_true", help="清空历史 SIM 重新开始")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.fresh and os.path.exists(SIM_PATH):
        os.remove(SIM_PATH)

    with open(CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(ROOT, "data", "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = load_samples(manifest)
    sim = SIMMemory(SIM_PATH)

    runner = CRSIRunner(config, sim, samples, rounds=args.rounds, hard_frac=0.5, candidate_limit=6)
    report = runner.run()

    # 重新构造 baseline / evolved 流水线，出对比图与逐图指标
    rows = []
    for task, rep in report["tasks"].items():
        base_pipe = HarnessPipeline(config["tasks"][task]["baseline"])
        best_pipe = HarnessPipeline(rep["best_config"])
        make_comparison(task, samples[task], base_pipe, best_pipe,
                        os.path.join(RESULTS, f"comparison_{task}.png"))
        for variant, pipe, metrics in [
            ("baseline", base_pipe, rep["baseline_metrics"]),
            ("evolved", best_pipe, rep["best_metrics"]),
        ]:
            row = {"task": task, "variant": variant, "backend": pipe.backend_name, **metrics}
            rows.append(row)

    # CSV 指标表
    csv_path = os.path.join(RESULTS, "metrics_table.csv")
    fields = ["task", "variant", "backend", "psnr", "ssim", "channel_shift", "contrast_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in fields})

    with open(os.path.join(RESULTS, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("=== SIM 统计:", report["sim_stats"])
    for task, rep in report["tasks"].items():
        b, e = rep["baseline_metrics"], rep["best_metrics"]
        print(f"[{task}] {rep['best_tag']} backend={rep['best_config']['backend']} | "
              f"PSNR {b['psnr']:.2f}->{e['psnr']:.2f}  SSIM {b['ssim']:.3f}->{e['ssim']:.3f}  "
              f"shift {b['channel_shift']:.2f}->{e['channel_shift']:.2f}")
    print(f"结果已保存: {RESULTS}")


if __name__ == "__main__":
    main()
