"""下载/整理真实 benchmark 成对数据。

- lowlight : LOL test 成对低光 (15 对)            HF: ishicode/low-light-dataset
- dehaze   : RESIDE 风格去雾测试集 (100 对)       HF: linxin020826/dehazing_test
- underwater: EUVP 成对水下 (抽样 40 对)           HF: Ken1053/EUVP
- denoise  : Kodak24 清晰图 + 现场高斯噪声 sigma=25 HF: msdkhairi/kodak

输出: data/real/<task>/{input,gt}/*.png + manifest_real.json
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
REAL = os.path.join(ROOT, "data", "real")
ARCH = os.path.join(REAL, "_archive")
os.makedirs(ARCH, exist_ok=True)

HF = "https://huggingface.co/datasets/{}/resolve/main/{}"


def _list_files(repo: str, sub: str) -> list[str]:
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{sub}?recursive=true"
    with urllib.request.urlopen(url, timeout=20) as r:
        entries = json.load(r)
    return [e["path"] for e in entries if e["type"] == "file" and e["path"].lower().endswith((".png", ".jpg", ".jpeg"))]


def _download(repo: str, path: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    url = HF.format(repo, path)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def prep_lowlight(manifest: dict) -> None:
    repo = "ishicode/low-light-dataset"
    in_dir = os.path.join(REAL, "lowlight", "input")
    gt_dir = os.path.join(REAL, "lowlight", "gt")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    lows = [p for p in _list_files(repo, "test/low")]
    items = []
    for p in sorted(lows):
        name = os.path.basename(p)
        _download(repo, f"test/low/{name}", os.path.join(in_dir, name))
        _download(repo, f"test/high/{name}", os.path.join(gt_dir, name))
        items.append({"name": name[:-4], "input": name, "gt": name})
    manifest["lowlight"] = items
    print(f"lowlight: {len(items)} 对")


def prep_dehaze(manifest: dict, limit: int = 100) -> None:
    repo = "linxin020826/dehazing_test"
    in_dir = os.path.join(REAL, "dehaze", "input")
    gt_dir = os.path.join(REAL, "dehaze", "gt")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    hazy = [os.path.basename(p) for p in _list_files(repo, "test_dehazing/hazy")]
    items = []
    for name in sorted(hazy)[:limit]:
        _download(repo, f"test_dehazing/hazy/{name}", os.path.join(in_dir, name))
        _download(repo, f"test_dehazing/clear/{name}", os.path.join(gt_dir, name))
        items.append({"name": name[:-4], "input": name, "gt": name})
    manifest["dehaze"] = items
    print(f"dehaze: {len(items)} 对")


def prep_underwater(manifest: dict, n: int = 40, seed: int = 0) -> None:
    parquet = os.path.join(ARCH, "euvp.parquet")
    if not os.path.exists(parquet):
        raise FileNotFoundError("euvp.parquet 缺失，请先下载")
    in_dir = os.path.join(REAL, "underwater", "input")
    gt_dir = os.path.join(REAL, "underwater", "gt")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    df = pd.read_parquet(parquet)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    items = []
    for k, i in enumerate(idx):
        name = f"uw_{k:03d}.png"
        Image.open(io.BytesIO(df.iloc[i]["input_image"]["bytes"])).convert("RGB").save(os.path.join(in_dir, name))
        Image.open(io.BytesIO(df.iloc[i]["edited_image"]["bytes"])).convert("RGB").save(os.path.join(gt_dir, name))
        items.append({"name": name[:-4], "input": name, "gt": name})
    manifest["underwater"] = items
    print(f"underwater: {len(items)} 对")


def prep_denoise(manifest: dict, sigma: float = 25.0, seed: int = 0) -> None:
    zip_path = os.path.join(ARCH, "kodak.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError("kodak.zip 缺失")
    in_dir = os.path.join(REAL, "denoise", "input")
    gt_dir = os.path.join(REAL, "denoise", "gt")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".png")):
            arr = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
            base = os.path.basename(name)
            Image.fromarray(arr).save(os.path.join(gt_dir, base))
            noise = rng.normal(0, sigma, arr.shape)
            noisy = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(noisy).save(os.path.join(in_dir, base))
            items.append({"name": base[:-4], "input": base, "gt": base, "sigma": sigma})
    manifest["denoise"] = items
    print(f"denoise: {len(items)} 张 (sigma={sigma})")


def main() -> None:
    manifest: dict = {}
    prep_lowlight(manifest)
    prep_dehaze(manifest)
    prep_underwater(manifest)
    prep_denoise(manifest)
    with open(os.path.join(REAL, "manifest_real.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("真实数据清单已保存:", os.path.join(REAL, "manifest_real.json"))


if __name__ == "__main__":
    main()
