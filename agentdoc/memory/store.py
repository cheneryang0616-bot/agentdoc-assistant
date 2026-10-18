"""记忆存储：工作 / 情景 / 语义 三级记忆，文件持久化（JSONL）。

* 工作记忆（working）  ：会话内临时信息，驻留内存，随会话结束可清理；
* 情景记忆（episodic） ：具体事件与经历，带时间戳，按时间序列组织；
* 语义记忆（semantic） ：抽象概念 / 规则 / 知识，长期沉淀。

底层存储可在 Markdown / JSONL / DB 之间切换；此处用 JSONL（轻量、可读、便于
Agent 通过 memory 工具（read_file / write_file）交互）。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import config

TYPES = ("working", "episodic", "semantic")


@dataclass
class MemoryItem:
    content: str
    type: str = "working"
    id: str = field(default_factory=lambda: "mem_" + uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    importance: float = 0.3
    access_count: int = 0
    source: str = "user"
    meta: Dict[str, Any] = field(default_factory=dict)

    def age_hours(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        return (now - self.created_at) / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryStore:
    def __init__(self, root: Optional[Path] = None, embedder=None) -> None:
        self.root = Path(root or config.MEMORY_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.items: Dict[str, MemoryItem] = {}
        # 按类型分文件持久化
        self.files = {t: self.root / f"{t}.jsonl" for t in TYPES}
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self) -> None:
        for t, f in self.files.items():
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.items[d["id"]] = MemoryItem(**d)

    def flush(self) -> None:
        for t, f in self.files.items():
            rows = [i.to_dict() for i in self.items.values() if i.type == t]
            f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
                         encoding="utf-8")

    # ---------------- CRUD ----------------
    def add(self, content: str, type: str = "working", importance: float = 0.3,
            source: str = "user", **meta) -> MemoryItem:
        item = MemoryItem(content=content, type=type, importance=importance,
                          source=source, meta=meta)
        self.items[item.id] = item
        self.flush()
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self.items.get(item_id)

    def by_type(self, type: str) -> List[MemoryItem]:
        return [i for i in self.items.values() if i.type == type]

    def delete(self, item_id: str) -> bool:
        if item_id in self.items:
            del self.items[item_id]
            self.flush()
            return True
        return False

    def touch(self, item_id: str) -> None:
        it = self.items.get(item_id)
        if it:
            it.last_access = time.time()
            it.access_count += 1

    # ---------------- 语义检索 + 混合排序 ----------------
    def search(self, query: str, top_k: int = 3, type: Optional[str] = None,
               use_hybrid: bool = True) -> List[MemoryItem]:
        from .scoring import hybrid_score
        cands = [i for i in self.items.values() if (type is None or i.type == type)]
        if not cands:
            return []
        if self.embedder is None:
            return cands[:top_k]
        qv = self.embedder.encode(query)
        now = time.time()
        scored = []
        for it in cands:
            rel = float(np.dot(qv, self.embedder.encode(it.content)))
            s = hybrid_score(it.importance, it.age_hours(now), rel) if use_hybrid else rel
            scored.append((s, it, rel))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, it, rel in scored[:top_k]:
            it.meta["_last_score"] = round(s, 4)
            it.meta["_last_relevance"] = round(rel, 4)
            self.touch(it.id)
            out.append(it)
        return out

    def stats(self) -> Dict[str, int]:
        return {t: len(self.by_type(t)) for t in TYPES}
