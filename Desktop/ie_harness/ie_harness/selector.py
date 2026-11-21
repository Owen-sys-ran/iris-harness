"""Selector: 依据 SIM 历史对候选做剪枝。

- 同退化下历史判定 regressed 的 (编辑键, 取值) 直接剔除，避免重复踩坑；
- 历史 improved 的 (编辑键, 取值) 标记为优先；
- 剪枝理由写回候选，供 SIM provenance 追溯。
经验跨后端复用，不按 backbone 过滤。
"""
from __future__ import annotations

from .sim.memory import tokenize


class Selector:
    def __init__(self, sim):
        self.sim = sim

    def select(self, degradation: str, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        regressed = self.sim.regressed_edits(degradation)
        improved = self.sim.improved_edits(degradation)
        kept, pruned = [], []
        for c in candidates:
            key = c["edit"]["key"]
            tok = tokenize(c["edit"]["new"])
            if (key, tok) in regressed:
                c["pruned_reason"] = f"SIM 记录该编辑在 {degradation} 上历史回退，剔除"
                pruned.append(c)
                continue
            c["priority"] = "improved-before" if (key, tok) in improved else "fresh"
            kept.append(c)
        return kept, pruned
