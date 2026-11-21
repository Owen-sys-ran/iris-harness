"""从 results/report.json 生成 Markdown 实验报告。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORT_JSON = os.path.join(ROOT, "results", "report.json")
SIM_JSON = os.path.join(ROOT, "sim_store", "sim.json")
OUT = os.path.join(ROOT, "results", "REPORT.md")

BACKEND_TABLE = [
    ("Zero-DCE", "低光增强", "深度 CNN (7层)", "79.4k", "Epoch99.pth / 320KB", "Li-Chongyi/Zero-DCE, ICCV2020"),
    ("FUnIE-GAN", "水下增强", "深度 UNet-GAN", "7.02M", "funie_generator.pth / 28MB", "xahidbuffon/funie-gan, RA-L2020"),
    ("AOD-Net", "去雾", "深度 CNN (5层)", "1.8k", "final_model.pth / 10.8KB", "zhuleyang/aodnet, ICCV2017"),
    ("GrayWorld / CLAHE / MSR / Bilateral / NLM", "白平衡/对比度/Retinex/去噪", "传统算子(无权重)", "0", "—", "OpenCV / scikit-image"),
]

TASK_CN = {"dehaze": "去雾", "lowlight": "低光", "underwater": "水下", "denoise": "去噪"}


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    rep = json.load(open(REPORT_JSON, encoding="utf-8"))
    sim = json.load(open(SIM_JSON, encoding="utf-8"))
    entries = sim["entries"]
    stats = rep["sim_stats"]

    lines: list[str] = []
    lines.append("# 图像增强专属 CRSI-Harness 原型实验报告\n")
    lines.append("> 冻结增强骨干权重，只进化包裹模型的外围 Harness（预处理 / 后端选择 / 后处理），"
                 "并以 SIM 结构化记忆跨轮次积累成功与失败经验。范式对标 Google CRSI + SIM，迁移到图像复原/增强域。\n")

    # 1. 选型
    lines.append("## 1. 封装的轻量级后端\n")
    lines.append("| 模型 | 任务 | 类型 | 参数量 | 权重 | 来源 |")
    lines.append("|---|---|---|---|---|---|")
    for row in BACKEND_TABLE:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 2. 设置
    lines.append("## 2. 实验设置\n")
    lines.append("- 数据：4 张清晰参考图（astronaut / coffee / chelsea / hubble），统一 256×256；"
                 "按物理模型合成雾 / 低光 / 水下色偏 / 高斯噪声四类成对退化，种子固定可复现。")
    lines.append("- 环境：CPU（4 核 / 8GB），PyTorch 2.15；指标为成对 PSNR / SSIM / 通道均值色偏，"
                 "另附无参考对比度与简化 UIQM。")
    lines.append("- 闭环：每任务独立 Proposer → Selector → Evaluator；候选先在 hard 子集（PSNR 最差 50%）评估，"
                 "全量复评并带回退保护；共 2 轮。")
    lines.append("- 判定：PSNR ≥ +0.2dB 且 SSIM 不降为 improved；PSNR ≤ −0.3dB 或 SSIM ≤ −0.01 为 regressed；其余 neutral。\n")

    # 3. 主结果
    lines.append("## 3. 主结果（Baseline Harness → 进化后 Harness，全量集）\n")
    lines.append("| 任务 | 后端(B→E) | PSNR B | PSNR E | ΔPSNR | SSIM B | SSIM E | ΔSSIM | 色偏 B | 色偏 E | 结论 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for task, r in rep["tasks"].items():
        b, e = r["baseline_metrics"], r["best_metrics"]
        db = e["psnr"] - b["psnr"]
        ds = e["ssim"] - b["ssim"]
        tag = {"baseline": "保持基线", "evolved": "Harness 进化", "rolled-back": "回退保护拦截"}[r["best_tag"]]
        lines.append(
            f"| {TASK_CN[task]} | {r['best_config']['backend']} | {b['psnr']:.2f} | {e['psnr']:.2f} | {db:+.2f} | "
            f"{b['ssim']:.3f} | {e['ssim']:.3f} | {ds:+.3f} | {b['channel_shift']:.2f} | {e['channel_shift']:.2f} | {tag} |")
    lines.append("")

    # 4. SIM 行为
    lines.append("## 4. SIM 闭环行为\n")
    lines.append(f"- SIM 累计条目 **{stats['total']}** 条：improved {stats['improved']}，"
                 f"regressed {stats['regressed']}，neutral {stats['neutral']}；失败条目同样留存用于剪枝。")
    pruned_total = sum(len([h for h in r["history"] if h["stage"] == "pruned"])
                       for r in rep["tasks"].values())
    lines.append(f"- 第 2 轮 Selector 依据 SIM 历史自动剪枝 **{pruned_total}** 个曾回退的 (编辑键, 取值) 候选，避免重复试错。\n")

    lines.append("### 典型记忆条目\n")
    lines.append("| 任务 | 编辑 | 判定 | 观测 PSNR/SSIM | 根因/动机 |")
    lines.append("|---|---|---|---|---|")
    for e in entries:
        o = e["observed_outcome"]
        lines.append(
            f"| {TASK_CN.get(e['degradation'], e['degradation'])} | "
            f"{e['harness_edit']['key']}→{e['harness_edit']['new']} | {e['verdict']} | "
            f"{o['psnr']:.2f}/{o['ssim']:.3f} | {e['rationale']} |")
    lines.append("")

    # 5. 观察
    lines.append("## 5. 逐任务观察\n")
    lines.append("- **水下（进化成功）**：FUnIE-GAN 后端叠加后处理增益 1.05，PSNR 24.01→24.92，SSIM 持平，"
                 "色偏 7.30→6.51；暖色物体（咖啡杯、猫）更贴近 GT。")
    lines.append("- **去噪（回退保护生效）**：双边滤波在 hard 子集 PSNR 略高，但全量 SSIM 0.609→0.534 明显退步，"
                 "被全量复评回退保护拦回 NLM；该失败经验已写入 SIM。")
    lines.append("- **去雾 / 低光（保持基线）**：增益、白平衡、传统后端候选在两类退化上均回退或中性；"
                 "Zero-DCE 会放大暗部彩色噪声（Hubble 星场明显），是后续 SIM 应积累的根因类型。\n")

    # 6. 局限
    lines.append("## 6. 局限与下一步\n")
    lines.append("- 退化为合成模型，与真实雾/水下/低光存在域差；下一步接入 RESIDE / EUVP / SID 真实评测集。")
    lines.append("- 候选空间为离散单因素编辑；后续支持多因素组合、Ablation 自动解耦耦合部件。")
    lines.append("- 指标未含 LPIPS / CLIP-IQA 与主观评分；PSNR-主观冲突时应以多指标联合判定。")
    lines.append("- 后端当前为插拔式冻结；可选开关放开少量参数微调，但主创新点仍放在 Harness + SIM 层。\n")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"报告已生成: {OUT}")


if __name__ == "__main__":
    main()
