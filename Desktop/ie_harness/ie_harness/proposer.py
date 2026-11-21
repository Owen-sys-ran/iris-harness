"""Proposer: 读取当前 Harness、SIM 历史、失败根因统计，生成候选编辑。

- 单因素编辑为主，便于把性能变化归因到具体 harness 部件；
- 根因驱动排序: 暗部 -> 提亮轴; 色偏 -> 白平衡/通道校色; 低对比 -> 拉伸/锐化;
- 支持二阶组合: 在第一轮改进轴之上做两两交叉。
"""
from __future__ import annotations

from .sim.memory import tokenize
# 候选空间: (stage, key, values, 关联根因)
CANDIDATE_SPACE: dict[str, list[dict]] = {
    "lowlight": [
        {"stage": "postprocess", "key": "gain", "values": [1.1, 1.2, 1.35], "causes": ["dark"]},
        {"stage": "postprocess", "key": "gamma", "values": [0.7, 0.85, 1.15], "causes": ["dark"]},
        {"stage": "postprocess", "key": "sharpen", "values": [0.3, 0.6], "causes": ["low_contrast"]},
        {"stage": "preprocess", "key": "stretch", "values": [[1, 99]], "causes": ["low_contrast"]},
        {"stage": "preprocess", "key": "white_balance", "values": ["grayworld"], "causes": ["color_cast"]},
        {"stage": "backend", "key": "backend", "values": ["msr", "clahe"], "causes": ["dark", "low_contrast"]},
    ],
    "dehaze": [
        {"stage": "postprocess", "key": "gamma", "values": [1.15, 1.25], "causes": ["haze_residual", "bright"]},
        {"stage": "postprocess", "key": "gain", "values": [0.95, 1.05, 1.12], "causes": ["haze_residual", "bright", "dark"]},
        {"stage": "postprocess", "key": "channel_gain", "values": [[0.92, 1.0, 1.0], [0.95, 1.0, 1.02]], "causes": ["color_cast"]},
        {"stage": "preprocess", "key": "stretch", "values": [[2, 98], [1, 99]], "causes": ["haze_residual", "low_contrast"]},
        {"stage": "postprocess", "key": "sharpen", "values": [0.3, 0.5], "causes": ["haze_residual"]},
        {"stage": "postprocess", "key": "tta_flip", "values": [True], "causes": ["haze_residual"]},
        {"stage": "preprocess", "key": "white_balance", "values": ["grayworld"], "causes": ["color_cast"]},
        {"stage": "backend", "key": "backend", "values": ["msr", "clahe"], "causes": ["haze_residual"]},
    ],
    "underwater": [
        {"stage": "postprocess", "key": "tta_flip", "values": [True], "causes": ["color_cast", "low_contrast"]},
        {"stage": "postprocess", "key": "channel_gain",
         "values": [[1.05, 1.0, 0.95], [1.03, 1.0, 0.97]], "causes": ["color_cast"]},
        {"stage": "postprocess", "key": "gain", "values": [0.95, 1.05], "causes": ["dark", "color_cast"]},
        {"stage": "preprocess", "key": "white_balance", "values": ["grayworld"], "causes": ["color_cast"]},
        {"stage": "postprocess", "key": "sharpen", "values": [0.3], "causes": ["low_contrast"]},
        {"stage": "backend", "key": "backend", "values": ["grayworld", "msr"], "causes": ["color_cast"]},
    ],
    "denoise": [
        {"stage": "backend", "key": "backend", "values": ["bilateral", "nlm"], "causes": ["noise_residual"]},
        {"stage": "backend_config", "key": "h", "values": [7, 10, 15], "causes": ["noise_residual"], "backend": "nlm"},
        {"stage": "backend_config", "key": "sigma_color", "values": [30, 50], "causes": ["noise_residual"], "backend": "bilateral"},
        {"stage": "postprocess", "key": "sharpen", "values": [0.2], "causes": ["detail_loss"]},
    ],
}

RATIONALE = {
    "gain": "线性缩放整体亮度/对比度，补偿暗部或雾感残留",
    "gamma": "曲线映射，非线性抬升暗部或压制高光",
    "sharpen": "Unsharp Mask 补偿去模糊/去噪带来的细节损失",
    "stretch": "百分号对比度拉伸，扩展动态范围",
    "white_balance": "灰度世界白平衡校正色偏",
    "channel_gain": "R/B 独立通道增益，校正水下/偏色",
    "backend": "在同任务备选冻结后端间切换路线",
    "h": "NLM 滤波强度，权衡噪声残留与细节平滑",
    "sigma_color": "双边滤波色彩滤波孔径，保边平滑权衡",
    "tta_flip": "水平翻转推理后与原图输出平均，抑制生成伪影、稳定结构",
}


def _mean(v):
    return v["mean"] if isinstance(v, dict) else v


def analyze_failure(stats: dict) -> set[str]:
    """由当前输出统计归纳失败根因标签。"""
    causes: set[str] = set()
    if stats.get("mean_val", 128) < 90:
        causes.add("dark")
    if stats.get("mean_val", 128) > 150:
        causes.add("bright")
    if stats.get("contrast", 50) < 38:
        causes.add("low_contrast")
    shift = _mean(stats.get("channel_shift", 0))
    psnr = _mean(stats.get("psnr", 20))
    if abs(stats.get("rb_ratio", 1.0) - 1.0) > 0.12 or shift > 12:
        causes.add("color_cast")
    if psnr < 18:
        causes.add("haze_residual")
        causes.add("noise_residual")
    return causes


class Proposer:
    def __init__(self, sim):
        self.sim = sim

    def _priority(self, cand: dict, causes: set[str], improved: dict) -> int:
        key, new = cand["edit"]["key"], cand["edit"]["new"]
        if (key, tokenize(new)) in improved:
            return 0  # 历史改进，最优先
        if any(c in cand.get("causes", []) for c in causes):
            return 1  # 命中当前根因
        return 2

    def propose(self, degradation: str, current: dict, stats: dict | None = None,
                limit: int = 8) -> list[dict]:
        causes = analyze_failure(stats or {})
        improved = self.sim.improved_edits(degradation)
        space = CANDIDATE_SPACE.get(degradation, [])
        candidates: list[dict] = []
        cur_backend = current.get("backend")

        for axis in space:
            # backend_config 候选仅在对应后端已选中时生成
            if axis["stage"] == "backend_config" and axis.get("backend") != cur_backend:
                continue
            old = self._get_old(current, axis["stage"], axis["key"])
            for new in axis["values"]:
                if _same(old, new):
                    continue
                candidates.append({
                    "edit": {"stage": axis["stage"], "key": axis["key"], "old": old, "new": new},
                    "rationale": RATIONALE.get(axis["key"], "harness 候选编辑"),
                    "causes": axis.get("causes", []),
                    "pred_outcome": {"direction": "expect improvement on paired metrics"},
                })

        candidates.sort(key=lambda c: self._priority(c, causes, improved))
        return candidates[:limit]

    def propose_combos(self, degradation: str, current: dict, improved_edits: list[dict],
                       limit: int = 4) -> list[dict]:
        """二阶组合: 在第一轮被判定 improved 的单因素轴之间两两交叉。

        同一编辑键只保留一条最优轴，避免 gain+gain 这类同键冗余组合。
        """
        by_key: dict[str, dict] = {}
        for e in improved_edits:
            by_key.setdefault(e["key"], e)
        axes = list(by_key.values())
        combos: list[dict] = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                e1, e2 = axes[i], axes[j]
                combo = {"stage": "combo", "key": f"{e1['key']}+{e2['key']}",
                         "old": None, "new": [e1, e2]}
                combos.append({
                    "edit": combo,
                    "rationale": f"组合两条改进轴: {e1['key']} + {e2['key']}",
                    "causes": [],
                    "pred_outcome": {"direction": "expect additive improvement"},
                })
        return combos[:limit]

    @staticmethod
    def _get_old(current: dict, stage: str, key: str):
        if stage == "backend":
            return current.get("backend")
        if stage == "backend_config":
            return current.get("backend_config", {}).get(key)
        return current.get(stage, {}).get(key)


def _same(a, b) -> bool:
    return tokenize(a) == tokenize(b)
