"""工具处理器：把 RAG 检索、记忆读写、SubAgent、文件读取封装成可调用工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .. import config


def build_rag_search(retriever, use_advanced: bool = True):
    def rag_search(query: str, top_k: int = config.TOP_K) -> str:
        hits = (retriever.advanced_search(query, top_k) if use_advanced
                else retriever.search(query, top_k))
        payload = {
            "query": query,
            "mode": "advanced(MQE+HyDE+RRF)" if use_advanced else "dense",
            "hits": [
                {"doc_id": h.doc_id, "chunk_id": h.id, "score": round(h.score, 4),
                 "text": h.text}
                for h in hits
            ],
        }
        return json.dumps(payload, ensure_ascii=False)
    return rag_search


def build_memory_tools(memory):
    def memory_write(content: str, importance: float = 0.6, type: str = "working") -> str:
        it = memory.add(content, type=type, importance=float(importance), source="agent")
        return json.dumps({"ok": True, "id": it.id, "type": it.type,
                           "importance": it.importance}, ensure_ascii=False)

    def memory_read(query: str, top_k: int = 3) -> str:
        items = memory.search(query, top_k=top_k)
        return json.dumps({"query": query, "count": len(items),
                           "memories": [
                               {"id": m.id, "type": m.type, "content": m.content,
                                "importance": m.importance,
                                "score": m.meta.get("_last_score")}
                               for m in items]}, ensure_ascii=False)
    return memory_write, memory_read


def build_subagent_tool(run_subagent_fn):
    def subagent(task: str = "consolidate_memory") -> str:
        return run_subagent_fn(task)
    return subagent


def build_file_read(offload_dir: Optional[Path] = None):
    root = Path(offload_dir or config.OFFLOAD_DIR)

    def file_read(path: str, max_chars: int = 4000) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = root / path
        if not p.exists():
            return f"[file_read] 文件不存在: {p}"
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    return file_read
