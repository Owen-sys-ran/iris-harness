"""Self-Improvement Memory (SIM) for image-enhancement harness evolution.

每条记忆按【退化根因】组织，记录一次 harness 编辑的：
根因、编辑内容、动机、预期结果、观测结果、判定与来源上下文。
成功与失败条目同等保留——失败记录用于在 Selector 阶段规避重复踩坑。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any


def tokenize(v: Any) -> str:
    try:
        return json.dumps(v, sort_keys=True)
    except Exception:
        return str(v)


class SIMMemory:
    def __init__(self, store_path: str):
        self.store_path = store_path
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
        self._data: dict[str, Any] = {"entries": []}
        if os.path.exists(store_path):
            with open(store_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    # ---- 写入 ----
    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry.setdefault("entry_id", uuid.uuid4().hex[:8])
        entry.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        self._data["entries"].append(entry)
        self._flush()
        return entry

    def _flush(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ---- 检索 ----
    def all_entries(self) -> list[dict[str, Any]]:
        return self._data["entries"]

    def search(self, degradation: str | None = None, backend: str | None = None,
               edit_key: str | None = None) -> list[dict[str, Any]]:
        out = []
        for e in self._data["entries"]:
            if degradation and e.get("degradation") != degradation:
                continue
            if backend and e.get("backbone") != backend:
                continue
            if edit_key and e.get("harness_edit", {}).get("key") != edit_key:
                continue
            out.append(e)
        return out

    def regressed_edits(self, degradation: str) -> set[tuple[str, str]]:
        """历史上在该退化上导致回退的 (编辑键, 取值token)，供 Selector 精确剪枝。

        不按 backbone 过滤：根因经验跨后端迁移正是 SIM 的核心价值。
        """
        pairs = set()
        for e in self.search(degradation=degradation):
            if e.get("verdict") == "regressed":
                pairs.add((e["harness_edit"]["key"], tokenize(e["harness_edit"]["new"])))
        return pairs

    def improved_edits(self, degradation: str) -> dict[tuple[str, str], dict]:
        pairs: dict[tuple[str, str], dict] = {}
        for e in self.search(degradation=degradation):
            if e.get("verdict") == "improved":
                pairs[(e["harness_edit"]["key"], tokenize(e["harness_edit"]["new"]))] = e
        return pairs

    def stats(self) -> dict[str, int]:
        counts = {"improved": 0, "regressed": 0, "neutral": 0, "total": 0}
        for e in self._data["entries"]:
            counts[e.get("verdict", "neutral")] += 1
            counts["total"] += 1
        return counts
