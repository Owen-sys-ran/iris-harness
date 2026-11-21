"""准备成对实验数据：skimage 内置清晰图 -> 统一 256x256 GT -> 合成四类退化。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image
from skimage import data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ie_harness.degradations import DEGRADERS

ROOT = os.path.join(os.path.dirname(__file__), "..")
GT_DIR = os.path.join(ROOT, "data", "gt")
DEG_DIR = os.path.join(ROOT, "data", "degraded")
SIZE = 256

GT_IMAGES = {
    "astronaut": data.astronaut(),
    "coffee": data.coffee(),
    "chelsea": data.chelsea(),
    "hubble": data.hubble_deep_field(),
}

# 各任务退化参数（seed 固定，保证可复现）
DEGRADE_PARAMS = {
    "dehaze": {"beta": 0.55, "A": 220.0},
    "lowlight": {"scale": 0.25, "noise_sigma": 6.0},
    "underwater": {"r_decay": 0.55, "b_decay": 0.92, "veil": 0.18},
    "denoise": {"sigma": 25.0},
}


def main() -> None:
    os.makedirs(GT_DIR, exist_ok=True)
    manifest: dict[str, list[dict]] = {t: [] for t in DEGRADERS}

    gts: dict[str, np.ndarray] = {}
    for name, arr in GT_IMAGES.items():
        gt = np.asarray(Image.fromarray(arr).resize((SIZE, SIZE), Image.BICUBIC))
        gts[name] = gt
        Image.fromarray(gt).save(os.path.join(GT_DIR, f"{name}.png"))

    for task, fn in DEGRADERS.items():
        out_dir = os.path.join(DEG_DIR, task)
        os.makedirs(out_dir, exist_ok=True)
        for i, (name, gt) in enumerate(gts.items()):
            deg = fn(gt, seed=i, **DEGRADE_PARAMS[task])
            Image.fromarray(deg).save(os.path.join(out_dir, f"{name}.png"))
            manifest[task].append({"name": name, "gt": name, "seed": i})

    with open(os.path.join(ROOT, "data", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"准备完成: {len(gts)} 张 GT, {len(manifest)} 类退化 -> {ROOT}/data")


if __name__ == "__main__":
    main()
