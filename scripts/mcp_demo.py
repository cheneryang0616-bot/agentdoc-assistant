"""MCP 端到端演示：把 agentdoc 的 rag_search 换成 MCP 远程工具，AgentLoop 一行不改。

运行：
    python3 scripts/mcp_demo.py                 # 默认 hash 嵌入（快）
    python3 scripts/mcp_demo.py --embedder st   # 用 BGE 真实嵌入

流程：
    1) 拉起 MCP server（stdio 子进程）
    2) initialize 握手
    3) tools/list 发现工具
    4) tools/call 直接调 rag_search
    5) 把 MCP 工具注册进 Agent 的工具表，覆盖本地 rag_search
    6) 跑一次多跳问答，验证调用确实走了 MCP
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.mcp import MCPClient, MCPToolAdapter, result_text
from agentdoc.rag import build_embedder
from agentdoc.session import AgentSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash", choices=["hash", "st", "auto"])
    args = ap.parse_args()

    mem_dir = config.RESULTS_DIR / "mcp_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("1) 启动 MCP server（stdio 子进程）")
    print("=" * 64)
    client = MCPClient.spawn(embedder=args.embedder, memory_dir=str(mem_dir))
    info = client.initialize()
    print(f"  serverInfo      : {info.get('serverInfo')}")
    print(f"  protocolVersion : {info.get('protocolVersion')}")
    print(f"  capabilities    : {info.get('capabilities')}")

    print()
    print("=" * 64)
    print("2) tools/list —— 客户端动态发现工具")
    print("=" * 64)
    adapter = MCPToolAdapter(client)
    for t in adapter.tools:
        schema = t.get("inputSchema", {})
        req = schema.get("required", [])
        print(f"  · {t['name']:<14} 必填={req}")
        print(f"      {t.get('description', '')}")

    print()
    print("=" * 64)
    print("3) tools/call —— 直接调用 rag_search")
    print("=" * 64)
    res = client.call_tool("rag_search", {"query": "JVM 的垃圾回收机制", "top_k": 3})
    text = result_text(res)
    payload = json.loads(text)
    for h in payload.get("hits", []):
        print(f"  {h['score']:.4f}  {h['chunk_id']:<20} {h['text'][:46]}")

    print()
    print("=" * 64)
    print("4) 把 MCP 工具接入 Agent（覆盖本地 rag_search）")
    print("=" * 64)
    sess = AgentSession(embedder=build_embedder(args.embedder),
                        memory_dir=mem_dir)
    before = sess.registry.names()
    registered = adapter.register_into(sess.registry, overrides={"rag_search": "rag_search"})
    print(f"  注册前工具: {before}")
    print(f"  注册后工具: {sess.registry.names()}")
    print(f"  覆盖同名工具: {registered}")

    print()
    print("=" * 64)
    print("5) 跑一次多跳问答（AgentLoop 未做任何修改）")
    print("=" * 64)
    r = sess.ask("JVM 是什么？它在生产环境中有哪些典型应用场景？", keep_history=False)
    print(f"  轮数: {r['turns']}")
    for t in r["trace"]:
        print(f"    turn{t['turn']} -> {t['tool']} ({t['output_chars']} 字符)")
    print(f"  MCP 实际调用次数: {len(adapter.calls)}")
    for c in adapter.calls:
        print(f"    · {c['name']}({c['arguments'].get('query', '')})")
    print(f"  回答: {r['answer'][:120].replace(chr(10), ' ')}...")

    client.close()
    print()
    print("完成：AgentLoop / 工具注册表 / 记忆，全部零改动地跑在 MCP 之上。")


if __name__ == "__main__":
    main()
