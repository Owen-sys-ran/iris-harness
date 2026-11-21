"""从 results/real/report_real.json 生成真实集 Markdown 报告。"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
REAL = os.path.join(ROOT, "results", "real")
REPORT_JSON = os.path.join(REAL, "report_real.json")
SIM_JSON = os.path.join(ROOT, "sim_store", "sim_real.json")
OUT = os.path.join(REAL, "REAL_REPORT.md")

TASK_CN = {"dehaze": "去雾", "lowlight": "低光", "underwater": "水下", "denoise": "去噪"}

DATASET_TABLE = [
    ("低光", "LOL test（成对）", "15 对", "ishicode/low-light-dataset", "真实低光/正常曝光成对"),
    ("去雾", "RESIDE 风格测试集（成对）", "100 对", "linxin020826/dehazing_test", "真实清晰图 + 大气散射雾"),
    ("水下", "EUVP paired（抽样）", "40 对", "Ken1053/EUVP", "真实水下偏色/正常成像成对"),
    ("去噪", "Kodak24 + 高斯噪声 σ=25", "24 张", "msdkhairi/kodak", "清晰图现场加噪，标准去噪协议"),
]

BACKEND_TABLE = [
    ("Zero-DCE", "低光", "深度 CNN 79.4k", "Epoch99.pth"),
    ("FUnIE-GAN", "水下", "深度 UNet-GAN 7.02M", "funie_generator.pth"),
    ("AOD-Net", "去雾", "深度 CNN 1.8k", "final_model.pth"),
    ("GrayWorld/CLAHE/MSR/Bilateral/NLM", "校色/对比/去噪", "传统算子 0 权重", "OpenCV / skimage"),
]


def m(agg, key):
    return agg[key]["mean"], agg[key]["std"]


def main() -> None:
    rep = json.load(open(REPORT_JSON, encoding="utf-8"))
    sim = json.load(open(SIM_JSON, encoding="utf-8"))
    stats = rep["sim_stats"]
    L: list[str] = []

    L.append("# 真实 Benchmark 上的 IE-CRSI-Harness 实验报告\n")
    L.append("> 冻结增强骨干权重，只进化外围 Harness（预处理 / 后端选择 / 后端参数 / 后处理 / TTA），"
             "SIM 按退化根因跨轮次积累成功与失败经验。本报告全部基于真实成对评测集。\n")

    L.append("## 1. 真实评测数据集\n")
    L.append("| 任务 | 数据集 | 规模 | 来源 | 说明 |")
    L.append("|---|---|---|---|---|")
    for row in DATASET_TABLE:
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 2. 封装后端（权重冻结）\n")
    L.append("| 模型 | 任务 | 类型/参数量 | 权重 |")
    L.append("|---|---|---|---|")
    for row in BACKEND_TABLE:
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## 3. 主结果（全量集，mean±std）\n")
    L.append("| 任务 | 后端(B→E) | PSNR B | PSNR E | ΔPSNR | SSIM B | SSIM E | ΔSSIM | 色偏 B | 色偏 E | 结论 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for task, r in rep["tasks"].items():
        b, e = r["baseline_metrics"], r["best_metrics"]
        bp, bps = m(b, "psnr"); ep, eps = m(e, "psnr")
        bs, bss = m(b, "ssim"); es, ess = m(e, "ssim")
        bc, bcs = m(b, "channel_shift"); ec, ecs = m(e, "channel_shift")
        tag = {"baseline": "保持基线", "evolved": "Harness 进化", "rolled-back": "回退保护拦截"}[r["best_tag"]]
        L.append(
            f"| {TASK_CN[task]} | {r['best_config']['backend']} | "
            f"{bp:.2f}±{bps:.2f} | {ep:.2f}±{eps:.2f} | {ep-bp:+.2f} | "
            f"{bs:.3f}±{bss:.3f} | {es:.3f}±{ess:.3f} | {es-bs:+.3f} | "
            f"{bc:.2f}±{bcs:.2f} | {ec:.2f}±{ecs:.2f} | {tag} |")
    L.append("")

    L.append("## 4. SIM 闭环行为\n")
    L.append(f"- SIM 累计 **{stats['total']}** 条：improved {stats['improved']}，"
             f"regressed {stats['regressed']}，neutral {stats['neutral']}；失败条目留存用于剪枝。")
    pruned = sum(len([h for h in r["history"] if h["stage"] == "pruned"])
                 for r in rep["tasks"].values())
    L.append(f"- Selector 依据 SIM 历史自动剪枝 **{pruned}** 个曾回退的 (编辑键, 取值) 候选。\n")

    L.append("### 判定采用归一化综合评分\n")
    L.append("`score = 0.5·ΔPSNR/10 + 0.3·ΔSSIM/0.2 − 0.2·Δ色偏/20`；"
             "硬退化保护：PSNR≤−0.3dB 或 SSIM≤−0.01 或色偏≥+3.0 直接判 regressed。"
             "避免单一 PSNR 埋没 SSIM/色偏上的改善。\n")

    L.append("## 5. 逐任务观察\n")
    for task, r in rep["tasks"].items():
        b, e = r["baseline_metrics"], r["best_metrics"]
        dp = e["psnr"]["mean"] - b["psnr"]["mean"]
        ds = e["ssim"]["mean"] - b["ssim"]["mean"]
        dsh = e["channel_shift"]["mean"] - b["channel_shift"]["mean"]
        dstd = e["psnr"]["std"] - b["psnr"]["std"]
        edits = list(dict.fromkeys(h["edit"]["key"] for h in r["history"]
                                   if h["stage"] == "evaluated" and h["verdict"] == "improved"))
        note = []
        if abs(dp) < 0.05 and (ds > 0.005 or dsh < -0.5):
            note.append("PSNR 基本持平，但 SSIM/色偏/最差样本方差收敛")
        if dstd < -0.3:
            note.append(f"PSNR 标准差 {b['psnr']['std']:.2f}→{e['psnr']['std']:.2f}（差样本被拉回）")
        note_s = "；" + "，".join(note) if note else ""
        L.append(f"- **{TASK_CN[task]}**：{r['best_tag']}，后端 {r['best_config']['backend']}；"
                 f"PSNR {b['psnr']['mean']:.2f}→{e['psnr']['mean']:.2f}（{dp:+.2f}），"
                 f"SSIM {b['ssim']['mean']:.3f}→{e['ssim']['mean']:.3f}（{ds:+.3f}），"
                 f"色偏 {b['channel_shift']['mean']:.2f}→{e['channel_shift']['mean']:.2f}（{dsh:+.2f}）；"
                 f"改进轴：{', '.join(edits) if edits else '无单因素改进'}{note_s}。")
    L.append("")

    L.append("## 6. 局限与下一步\n")
    L.append("- 深度后端为轻量老模型，部分任务上限受骨干约束；Harness 只优化外围，不更新权重。")
    L.append("- 去噪/去雾备选传统后端偏弱，后续可接入 RIDNet/轻量 Transformer 等更强冻结骨干做插拔对比。")
    L.append("- 指标未含 LPIPS / CLIP-IQA 与主观评分；候选空间仍以单因素 + 少量二阶组合为主。")
    L.append("- EUVP 为抽样 40 对；后续可扩到全量并补充 UIEB、SID、SOTS 等真实基准。\n")

    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"报告已生成: {OUT}")


if __name__ == "__main__":
    main()
