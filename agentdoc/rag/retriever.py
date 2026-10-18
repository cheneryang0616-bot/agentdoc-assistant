"""向量存储 + 高级检索（MQE 多查询扩展 + HyDE 假设文档嵌入 + RRF 融合）。

对应简历 ②「检索端引入多查询扩展(MQE)与假设文档嵌入(HyDE)，构建'扩展-检索-合并'框架」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import config
from .embedder import HashingEmbedder, tokenize


# --------------------------------------------------------------------------- #
# 向量存储
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    id: str
    text: str
    score: float
    doc_id: str = ""
    kind: str = "doc"
    meta: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """极简内存向量库，支持 kind / 元数据过滤（模拟 Qdrant 的 payload 过滤）。"""

    def __init__(self) -> None:
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict[str, Any]] = []

    def add(self, item_id: str, text: str, vec: np.ndarray,
            kind: str = "doc", doc_id: str = "", **meta) -> None:
        self.ids.append(item_id)
        self.texts.append(text)
        self.vecs.append(np.asarray(vec, dtype=np.float32))
        self.meta.append({"kind": kind, "doc_id": doc_id, **meta})

    def __len__(self) -> int:
        return len(self.ids)

    def _matrix(self) -> np.ndarray:
        return np.vstack(self.vecs) if self.vecs else np.zeros((0, config.EMBED_DIM), np.float32)

    def search(self, qvec: np.ndarray, top_k: int = 5,
               kind: Optional[str] = None, where: Optional[Dict] = None) -> List[Hit]:
        if not self.vecs:
            return []
        M = self._matrix()
        sims = M @ np.asarray(qvec, dtype=np.float32)
        order = np.argsort(-sims)
        out: List[Hit] = []
        for i in order:
            m = self.meta[i]
            if kind and m.get("kind") != kind:
                continue
            if where and any(m.get(k) != v for k, v in where.items()):
                continue
            out.append(Hit(self.ids[i], self.texts[i], float(sims[i]),
                           doc_id=m.get("doc_id", ""), kind=m.get("kind", "doc"), meta=m))
            if len(out) >= top_k:
                break
        return out


# --------------------------------------------------------------------------- #
# 查询扩展
# --------------------------------------------------------------------------- #
_STOP_QW = ("什么是", "是什么", "有哪些", "怎么", "如何", "为什么", "请问", "介绍一下",
            "能不能", "可以", "的", "吗", "呢", "？", "?")

_SYNONYMS = {
    "jvm": ["java虚拟机", "java virtual machine"],
    "java虚拟机": ["jvm"],
    "垃圾回收": ["gc", "garbage collection"],
    "gc": ["垃圾回收"],
    "线程": ["thread", "并发"],
    "内存": ["memory", "堆", "heap"],
    "应用": ["应用场景", "使用场景", "用途"],
    "概念": ["定义", "含义", "是什么"],
    "是什么": ["概念", "定义", "含义", "原理"],
    "原理": ["机制", "是什么"],
    "机制": ["原理", "如何实现"],
    "python": ["python3"],
}


def expand_queries(query: str) -> List[str]:
    """MQE：规则化多查询扩展（真实系统用 LLM 生成，这里可离线复现）。"""
    q = query.strip()
    variants = [q]

    # 1) 去疑问词后的核心词
    core = q
    for w in _STOP_QW:
        core = core.replace(w, "")
    core = core.strip(" ，,。:：、")
    if core and core != q and len(core) >= 2:
        variants.append(core)

    # 2) 同义词替换
    low = q.lower()
    for k, syns in _SYNONYMS.items():
        if k in low:
            for s in syns:
                variants.append(re.sub(k, s, low, flags=re.IGNORECASE))

    # 3) 关键词拼接变体
    toks = [t for t in tokenize(q) if len(t) >= 2][:6]
    if toks:
        variants.append(" ".join(toks))

    # 去重保序
    seen, out = set(), []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def hyde_pseudo_doc(query: str) -> str:
    """HyDE：生成一段"假设的答案文档"，用它的嵌入去检索。

    离线版用一个模板化的假设段落（真实系统调用 LLM 生成）；
    关键作用是让检索向量落在"文档分布"而非"疑问句分布"上。
    """
    core = query
    for w in _STOP_QW:
        core = core.replace(w, "")
    core = core.strip(" ，,。:：、？?")
    return (f"关于 {core} 的说明：{core} 指的是一个核心概念。"
            f"{core} 的常见应用场景包括若干典型情形，"
            f"在实际工程中需要理解 {core} 的原理、机制与最佳实践。")


# --------------------------------------------------------------------------- #
# 高级检索器
# --------------------------------------------------------------------------- #
def rrf_fuse(rank_lists: Sequence[List[Hit]], k: int = config.RRF_K) -> List[Hit]:
    """Reciprocal Rank Fusion：把多路检索结果按排名倒数加权合并。"""
    scores: Dict[str, float] = {}
    store: Dict[str, Hit] = {}
    for rl in rank_lists:
        for rank, h in enumerate(rl, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            store.setdefault(h.id, h)
    fused = []
    for hid, s in sorted(scores.items(), key=lambda x: -x[1]):
        h = store[hid]
        fused.append(Hit(h.id, h.text, s, h.doc_id, h.kind, h.meta))
    return fused


class Retriever:
    def __init__(self, store: VectorStore, embedder: HashingEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    # ---- 基础检索 ----
    def search(self, query: str, top_k: int = config.TOP_K,
               kind: Optional[str] = None, where: Optional[Dict] = None) -> List[Hit]:
        return self.store.search(self.embedder.encode(query), top_k, kind, where)

    # ---- MQE 检索 ----
    def mqe_search(self, query: str, top_k: int = config.TOP_K) -> List[Hit]:
        lists = [self.store.search(self.embedder.encode(v), top_k)
                 for v in expand_queries(query)]
        return rrf_fuse(lists)[:top_k]

    # ---- HyDE 检索 ----
    def hyde_search(self, query: str, top_k: int = config.TOP_K) -> List[Hit]:
        pseudo = hyde_pseudo_doc(query)
        return self.store.search(self.embedder.encode(pseudo), top_k)

    # ---- 扩展-检索-合并（MQE + HyDE + RRF） ----
    def advanced_search(self, query: str, top_k: int = config.TOP_K) -> List[Hit]:
        lists = [self.store.search(self.embedder.encode(v), top_k)
                 for v in expand_queries(query)]
        lists.append(self.hyde_search(query, top_k))
        return rrf_fuse(lists)[:top_k]
