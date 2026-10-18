"""agentdoc MCP 服务端：把 rag_search / memory_read / memory_write 暴露为 MCP 工具。

启动（stdio 传输，供 MCP 客户端拉起）：
    python3 -m agentdoc.mcp.server [--embedder hash|auto|st] [--verbose]

注意：stdout 只能输出协议消息；日志一律走 stderr，否则会破坏协议流。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import PROTOCOL_VERSION, StdioServer

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "rag_search",
        "description": "在知识库中做语义检索（MQE + HyDE + RRF），返回命中的文档片段。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_read",
        "description": "从长期记忆中检索相关条目（工作/情景/语义）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_write",
        "description": "把一条重要信息写入长期记忆。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "importance": {"type": "number", "description": "0~1"},
            },
            "required": ["content"],
        },
    },
]


class MCPServer(StdioServer):
    def __init__(self, session, verbose: bool = False) -> None:
        super().__init__({"name": "agentdoc-mcp", "version": "1.0.0"},
                         TOOLS, self._call, verbose)
        self.sess = session

    def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "rag_search":
                hits = self.sess.retriever.advanced_search(
                    args.get("query", ""), int(args.get("top_k", 5)))
                payload = {"query": args.get("query"), "hits": [
                    {"doc_id": h.doc_id, "chunk_id": h.id,
                     "score": round(h.score, 4), "text": h.text} for h in hits]}
            elif name == "memory_read":
                items = self.sess.memory.search(args.get("query", ""),
                                                top_k=int(args.get("top_k", 3)))
                payload = {"memories": [
                    {"id": m.id, "type": m.type, "content": m.content,
                     "importance": m.importance} for m in items]}
            elif name == "memory_write":
                it = self.sess.memory.add(args.get("content", ""), type="working",
                                          importance=float(args.get("importance", 0.6)),
                                          source="mcp")
                payload = {"ok": True, "id": it.id}
            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                        "isError": True}
            return {"content": [{"type": "text",
                                 "text": json.dumps(payload, ensure_ascii=False)}],
                    "isError": False}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                    "isError": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="agentdoc MCP server (stdio)")
    ap.add_argument("--embedder", default=None, choices=["auto", "hash", "st"])
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    from ..rag import build_embedder
    from ..session import AgentSession

    sess = AgentSession(
        embedder=build_embedder(args.embedder),
        memory_dir=Path(args.memory_dir) if args.memory_dir else None,
    )
    server = MCPServer(sess, verbose=args.verbose)
    server.log("ready, tools:", [t["name"] for t in TOOLS])
    server.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
