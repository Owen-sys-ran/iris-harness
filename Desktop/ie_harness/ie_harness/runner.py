"""CRSI 闭环 Runner: 冻结 policy model，递归进化 Harness，SIM 跨轮次积累经验。

流程（每个退化任务独立闭环）:
  baseline 评测 -> 根因统计 -> Proposer 候选(单因素 + 二轮组合) -> Selector 记忆剪枝
  -> Evaluator 在 hard 子集评测 -> 判定并写回 SIM（成功/失败都存）
  -> 更新当前最优 Harness -> 全量复评 + 回退保护。
"""
from __future__ import annotations

import copy

from .pipeline import HarnessPipeline
from .proposer import Proposer
from .selector import Selector
from .evaluator import Evaluator


class CRSIRunner:
    def __init__(self, config: dict, sim, samples_by_task: dict[str, list[dict]],
                 rounds: int = 2, hard_frac: float = 0.5, candidate_limit: int = 8):
        self.config = config
        self.sim = sim
        self.samples = samples_by_task
        self.rounds = rounds
        self.hard_frac = hard_frac
        self.candidate_limit = candidate_limit
        self.proposer = Proposer(sim)
        self.selector = Selector(sim)

    def _hard_subset(self, samples: list[dict], baseline_per_image: list[dict]) -> list[dict]:
        n = max(1, int(len(samples) * self.hard_frac))
        order = sorted(range(len(samples)), key=lambda i: baseline_per_image[i]["psnr"])
        idx = set(order[:n])
        return [s for i, s in enumerate(samples) if i in idx]

    def _run_task(self, task: str) -> dict:
        samples = self.samples[task]
        evaluator = Evaluator(task)
        base_cfg = copy.deepcopy(self.config["tasks"][task]["baseline"])
        base_pipe = HarnessPipeline(base_cfg)

        base_full = evaluator.evaluate_pipeline(base_pipe, samples)
        hard = self._hard_subset(samples, base_full["per_image"])
        base_hard = evaluator.evaluate_pipeline(base_pipe, hard)

        best_pipe = base_pipe
        best_hard = base_hard["aggregate"]
        best_tag = "baseline"
        history: list[dict] = []
        improved_axes: list[dict] = []

        for rnd in range(self.rounds):
            candidates = self.proposer.propose(task, best_pipe.config, best_hard, limit=self.candidate_limit)
            # 第二轮起，在已改进轴上做二阶组合
            if rnd >= 1 and improved_axes:
                candidates += self.proposer.propose_combos(task, best_pipe.config, improved_axes)

            kept, pruned = self.selector.select(task, candidates)
            for c in pruned:
                history.append({"round": rnd, "stage": "pruned", **c})

            for c in kept:
                cand_pipe = best_pipe.edited(c["edit"])
                cand_res = evaluator.evaluate_pipeline(cand_pipe, hard)
                agg = cand_res["aggregate"]
                verdict = evaluator.judge(agg, best_hard)
                entry = {
                    "degradation": task,
                    "backbone": cand_pipe.backend_name,
                    "harness_edit": c["edit"],
                    "rationale": c["rationale"],
                    "pred_outcome": c["pred_outcome"],
                    "observed_outcome": agg,
                    "verdict": verdict,
                    "provenance": {
                        "round": rnd,
                        "subset": "hard",
                        "n_samples": len(hard),
                        "baseline_metrics": best_hard,
                    },
                }
                self.sim.add(entry)
                history.append({"round": rnd, "stage": "evaluated", "verdict": verdict,
                                 "edit": c["edit"], "observed": agg})
                if verdict == "improved":
                    best_pipe, best_hard, best_tag = cand_pipe, agg, "evolved"
                    if c["edit"]["stage"] != "combo":
                        improved_axes.append(c["edit"])

        # 全量复评最优 Harness
        final_res = evaluator.evaluate_pipeline(best_pipe, samples)

        # 全量回退保护：hard 子集改进若在全量显著回退，退回 baseline 并写入 SIM
        bm, fm = base_full["aggregate"], final_res["aggregate"]
        dp = fm["psnr"]["mean"] - bm["psnr"]["mean"]
        ds = fm["ssim"]["mean"] - bm["ssim"]["mean"]
        dshift = fm["channel_shift"]["mean"] - bm["channel_shift"]["mean"]
        if best_tag != "baseline" and (dp <= -0.25 or ds <= -0.008 or dshift >= 3.0):
            self.sim.add({
                "degradation": task,
                "backbone": best_pipe.backend_name,
                "harness_edit": {"stage": "postprocess", "key": "fullset-rollback",
                                 "old": None, "new": best_pipe.backend_name},
                "rationale": "全量复评回退保护：hard 子集改进未迁移到全量样本",
                "pred_outcome": {"direction": "expect improvement to transfer"},
                "observed_outcome": fm,
                "verdict": "regressed",
                "provenance": {"round": "final", "subset": "full", "n_samples": len(samples),
                               "baseline_metrics": bm},
            })
            best_pipe, final_res, best_tag = base_pipe, base_full, "rolled-back"

        return {
            "task": task,
            "baseline_metrics": bm,
            "best_metrics": fm,
            "best_config": best_pipe.config,
            "best_tag": best_tag,
            "history": history,
            "per_image": final_res["per_image"],
        }

    def run(self) -> dict:
        report = {"tasks": {}, "sim_stats": self.sim.stats()}
        for task in self.samples:
            report["tasks"][task] = self._run_task(task)
        report["sim_stats"] = self.sim.stats()
        return report
