"""反向 MCP 演示：agentdoc 作为 host，同时连接多个外部 MCP server。

与 `mcp_demo.py` 的区别：
  * mcp_demo.py   —— agentdoc 把**自己**的能力暴露成 MCP server（当服务端）
  * 本脚本        —— agentdoc 作为**宿主**去连接别的 server（当客户端/宿主）

运行：
    python3 scripts/mcp_host_demo.py

配置：mcp_servers.json（兼容 Claude Desktop 的 mcpServers 写法）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.mcp import MCPHost
from agentdoc.rag import build_embedder
from agentdoc.session import AgentSession

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=" * 68)
    print("1) 加载 mcp_servers.json，逐个拉起外部 MCP server")
    print("=" * 68)
    host = MCPHost(ROOT / "mcp_servers.json", root=ROOT)
    host.start(verbose=True)
    print(f"  已连接 server: {list(host.clients)}")
    for name, tools in host.server_tools.items():
        print(f"  · {name:<10} -> {[t['name'] for t in tools]}")

    print()
    print("=" * 68)
    print("2) 工具聚合（命名空间隔离，避免同名冲突）")
    print("=" * 68)
    for t in host.tools():
        req = t["input_schema"].get("required", [])
        print(f"  {t['qualified_name']:<26} server={t['server']:<9} 必填={req}")

    print()
    print("=" * 68)
    print("3) 跨 server 直接调用")
    print("=" * 68)
    files = json.loads(host.call("fs__list_dir", {"path": "kb"}))
    print(f"  fs__list_dir('kb')    -> {len(files['entries'])} 个文件")
    for e in files["entries"][:4]:
        print(f"      {e['type']:<5} {e['name']}")

    hits = json.loads(host.call("agentdoc__rag_search",
                                {"query": "JVM 的垃圾回收机制", "top_k": 2}))
    print(f"  agentdoc__rag_search  -> {len(hits['hits'])} 条命中")
    for h in hits["hits"]:
        print(f"      {h['score']:.4f} {h['chunk_id']}")

    print()
    print("=" * 68)
    print("4) 把多个 server 的工具一起注册进 Agent（同名覆盖本地 rag_search）")
    print("=" * 68)
    sess = AgentSession(embedder=build_embedder("hash"),
                        memory_dir=config.RESULTS_DIR / "mcp_host_memory")
    before = list(sess.registry.names())
    registered = host.register_into(
        sess.registry, overrides={"agentdoc__rag_search": "rag_search"})
    print(f"  注册前: {before}")
    print(f"  新增  : {registered}")
    print(f"  注册后: {sess.registry.names()}")

    print()
    print("=" * 68)
    print("5) 跑一次多跳问答（工具来自两个不同 MCP server）")
    print("=" * 68)
    host.call_log.clear()          # 只统计 Agent 本轮的调用
    r = sess.ask("JVM 是什么？它有哪些应用场景？", keep_history=False)
    print(f"  轮数: {r['turns']}")
    for t in r["trace"]:
        print(f"    turn{t['turn']} -> {t['tool']} ({t['output_chars']} 字符)")
    print(f"  MCP 实际调用: {len(host.call_log)} 次")
    for c in host.call_log:
        print(f"    · {c['qualified']}  {list(c['arguments'].values())[:1]}")
    print(f"  回答: {r['answer'][:100].replace(chr(10), ' ')}...")

    host.close()
    print()
    print("完成：一个 Agent 同时挂载了两个异构 MCP server，业务代码零改动。")


if __name__ == "__main__":
    main()
