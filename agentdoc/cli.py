"""命令行入口： python -m agentdoc.cli [--demo] [--query ...]"""
from __future__ import annotations

import argparse
import sys

from .session import AgentSession

DEMO_QUERIES = [
    "JVM 是什么？它在生产环境中有哪些典型应用场景？",   # 多跳：需要 2 次检索
    "帮我记住：我偏好用中文回答，并且更关注 Java 生态。",   # 记忆写入
    "我之前说过我关注什么技术栈？",                     # 记忆读取
    "整理记忆",                                        # SubAgent 记忆整理
]


def print_result(res: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {res['query']}")
    print(f"[AgentLoop 轮数] {res['turns']}")
    for t in res["trace"]:
        print(f"   · turn{t['turn']} 调用 {t['tool']}  ({t['output_chars']} 字符)")
    comp = res["compression"]
    print(f"[上下文] {comp['before_tokens']} -> {comp['after_tokens']} tokens "
          f"(L1落盘 {comp['l1_offloaded']}, L2占位 {comp['l2_placeholder_turns']}, "
          f"L3 {'触发' if comp['l3_triggered'] else '未触发'})")
    print(f"A: {res['answer'][:600]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="agentdoc 智能文档问答 Agent")
    ap.add_argument("--demo", action="store_true", help="跑内置演示问题")
    ap.add_argument("--query", type=str, help="单条提问")
    args = ap.parse_args(argv)

    if args.demo:
        # 演示前清空记忆，保证可复现
        from . import config as _cfg
        for f in _cfg.MEMORY_DIR.glob("*.jsonl"):
            f.unlink()

    sess = AgentSession()
    print(f"[启动] 知识库分块 {len(sess.chunks)} 个；工具 {sess.registry.names()}")

    if args.query:
        print_result(sess.ask(args.query))
        return 0

    if args.demo:
        for q in DEMO_QUERIES:
            print_result(sess.ask(q))
        return 0

    # 交互模式
    while True:
        try:
            q = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("q", "exit", ""):
            break
        print_result(sess.ask(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
