"""MCP Host 集成测试：agentdoc 作为宿主连接多个外部 MCP server。

验证：多 server 拉起 / 工具聚合与命名空间 / 跨 server 调用 / 接入 AgentLoop。

    python3 tests/test_mcp_host.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentdoc.mcp import MCPHost
from agentdoc.rag import build_embedder
from agentdoc.session import AgentSession

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ✓ {name}")


def main() -> None:
    host = MCPHost(ROOT / "mcp_servers.json", root=ROOT)
    host.start()
    try:
        # 1) 多 server
        check("连接了 2 个 server", set(host.clients) == {"agentdoc", "fs"})

        # 2) 工具聚合 + 命名空间
        names = [t["qualified_name"] for t in host.tools()]
        check("聚合工具数 >= 6", len(names) >= 6)
        check("全部带命名空间前缀", all("__" in n for n in names))
        check("含 fs__list_dir", "fs__list_dir" in names)
        check("含 agentdoc__rag_search", "agentdoc__rag_search" in names)

        # 3) 跨 server 调用：文件系统 server
        listing = json.loads(host.call("fs__list_dir", {"path": "kb"}))
        files = [e["name"] for e in listing["entries"]]
        check("fs__list_dir 列出知识库", "jvm_concept.md" in files)

        txt = json.loads(host.call("fs__read_text", {"path": "kb/jvm_concept.md"}))
        check("fs__read_text 读到内容", "JVM" in txt.get("text", ""))

        # 4) 跨 server 调用：agentdoc 检索 server
        hits = json.loads(host.call("agentdoc__rag_search",
                                    {"query": "JVM 的垃圾回收机制", "top_k": 3}))
        check("agentdoc__rag_search 返回命中", len(hits.get("hits", [])) > 0)

        # 5) 注册进 Agent，并覆盖本地 rag_search
        sess = AgentSession(embedder=build_embedder("hash"),
                            memory_dir=ROOT / "results" / "mcp_host_memory")
        registered = host.register_into(
            sess.registry, overrides={"agentdoc__rag_search": "rag_search"})
        check("注册了多个 MCP 工具", len(registered) >= 5)
        check("rag_search 已被 MCP 版本覆盖", "rag_search" in registered)

        out = sess.ask("JVM 是什么？它有哪些应用场景？", keep_history=False)
        check("AgentLoop 至少一轮", out["turns"] >= 1)
        check("确实走了 MCP", len(host.call_log) >= 1)
        check("MCP 收到子问题",
              any("jvm" in str(c["arguments"].get("query", "")).lower()
                  for c in host.call_log))
        check("最终回答非空", bool(out["answer"].strip()))
    finally:
        host.close()

    print(f"\nMCP Host 集成测试：全部通过（{PASS} 项断言）")


if __name__ == "__main__":
    main()
