"""记忆生命周期管理。

对应简历 ①「通过设定重要性阈值自动将高价值短期工作记忆整合为长期情景/语义记忆」。

规则：
  * 工作记忆中 importance >= IMPORTANCE_PROMOTE_THRESHOLD 的条目，
    在 consolidate() 时被提升为情景记忆（episodic）；
  * 情景记忆中被反复访问（access_count >= 2）的条目，进一步蒸馏为语义记忆（semantic）；
  * 语义记忆中高度相似的条目做合并，避免冗余膨胀。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .. import config
from .store import MemoryItem, MemoryStore


def promote_working(store: MemoryStore,
                    threshold: float = config.IMPORTANCE_PROMOTE_THRESHOLD) -> List[MemoryItem]:
    """工作记忆 -> 情景记忆。"""
    promoted = []
    for it in list(store.by_type("working")):
        if it.importance >= threshold:
            it.type = "episodic"
            it.meta["promoted_from"] = "working"
            promoted.append(it)
    if promoted:
        store.flush()
    return promoted


def promote_episodic(store: MemoryStore, min_access: int = 2) -> List[MemoryItem]:
    """情景记忆 -> 语义记忆（被反复访问过的经历抽象为知识）。"""
    promoted = []
    for it in list(store.by_type("episodic")):
        if it.access_count >= min_access:
            it.type = "semantic"
            it.meta["abstracted_from"] = "episodic"
            promoted.append(it)
    if promoted:
        store.flush()
    return promoted


def merge_semantic(store: MemoryStore, sim_threshold: float = config.SEMANTIC_MERGE_THRESHOLD
                   ) -> List[tuple]:
    """合并语义记忆中高度相似的条目（相似度 > 阈值）。"""
    if store.embedder is None:
        return []
    items = store.by_type("semantic")
    vecs = {it.id: store.embedder.encode(it.content) for it in items}
    removed, merged_pairs = [], []
    ids = [it.id for it in items]
    seen = set()
    for i, a in enumerate(ids):
        if a in seen:
            continue
        for b in ids[i + 1:]:
            if b in seen:
                continue
            sim = float(np.dot(vecs[a], vecs[b]))
            if sim >= sim_threshold:
                # 保留重要度更高者，合并访问计数
                ia, ib = store.get(a), store.get(b)
                keep, drop = (ia, ib) if ia.importance >= ib.importance else (ib, ia)
                keep.access_count += drop.access_count
                keep.importance = max(keep.importance, drop.importance)
                keep.meta.setdefault("merged", []).append(drop.content[:40])
                store.delete(drop.id)
                seen.add(drop.id)
                merged_pairs.append((keep.id, drop.id, round(sim, 4)))
    return merged_pairs


def consolidate(store: MemoryStore, verbose: bool = True) -> Dict[str, object]:
    """一次完整的整理（供 SubAgent / 后台 CronJob 调用）。"""
    promoted_w = promote_working(store)
    promoted_e = promote_episodic(store)
    merged = merge_semantic(store)
    store.flush()
    report = {
        "working_to_episodic": [i.id for i in promoted_w],
        "episodic_to_semantic": [i.id for i in promoted_e],
        "semantic_merged": merged,
        "stats_after": store.stats(),
    }
    if verbose:
        print(f"[生命周期] 工作->情景 {len(promoted_w)} 条, "
              f"情景->语义 {len(promoted_e)} 条, 语义合并 {len(merged)} 对")
    return report
