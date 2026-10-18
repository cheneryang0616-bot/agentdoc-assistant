"""MCP 集成测试：握手 / 工具发现 / 工具调用 / 接入 AgentLoop。

不需要网络和 API Key：
    python3 tests/test_mcp.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc.mcp import MCPClient, MCPToolAdapter, result_text
from agentdoc.rag import build_embedder
from agentdoc.session import AgentSession

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ✓ {name}")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="agentdoc_mcp_"))
    client = MCPClient.spawn(embedder="hash", memory_dir=str(tmp))
    try:
        # 1) 握手
        info = client.initialize()
        check("initialize 返回 serverInfo",
              info.get("serverInfo", {}).get("name") == "agentdoc-mcp")
        check("协商出 protocolVersion", bool(info.get("protocolVersion")))
        check("声明 tools 能力", "tools" in info.get("capabilities", {}))

        # 2) 工具发现
        adapter = MCPToolAdapter(client)
        names = adapter.names()
        check("tools/list 返回 3 个工具", len(names) == 3)
        check("包含 rag_search / memory_read / memory_write",
              set(names) == {"rag_search", "memory_read", "memory_write"})
        check("工具带 inputSchema",
              all(t.get("inputSchema", {}).get("type") == "object"
                  for t in adapter.tools))

        # 3) 直接调用 rag_search
        res = client.call_tool("rag_search", {"query": "JVM 的垃圾回收机制", "top_k": 3})
        check("tools/call 未报错", res.get("isError") is False)
        payload = json.loads(result_text(res))
        check("rag_search 返回命中", len(payload.get("hits", [])) > 0)
        check("命中带 chunk_id", "chunk_id" in payload["hits"][0])

        # 4) 记忆写入 / 读取
        w = json.loads(result_text(client.call_tool(
            "memory_write", {"content": "用户偏好中文回答", "importance": 0.8})))
        check("memory_write 成功", w.get("ok") is True)
        r = json.loads(result_text(client.call_tool(
            "memory_read", {"query": "用户偏好", "top_k": 3})))
        check("memory_read 能读回", len(r.get("memories", [])) > 0)

        # 5) 接入 AgentLoop：覆盖同名 rag_search，业务代码零改动
        sess = AgentSession(embedder=build_embedder("hash"), memory_dir=tmp)
        assert "rag_search" in sess.registry.names()
        adapter.register_into(sess.registry, overrides={"rag_search": "rag_search"})
        out = sess.ask("JVM 是什么？它有哪些应用场景？", keep_history=False)
        check("AgentLoop 至少一轮", out["turns"] >= 1)
        check("确实走了 MCP（有调用记录）", len(adapter.calls) >= 1)
        check("MCP 收到的是子问题",
              any("jvm" in str(c["arguments"].get("query", "")).lower()
                  for c in adapter.calls))
        check("最终回答非空", bool(out["answer"].strip()))
    finally:
        client.close()

    print(f"\nMCP 集成测试：全部通过（{PASS} 项断言）")


if __name__ == "__main__":
    main()
