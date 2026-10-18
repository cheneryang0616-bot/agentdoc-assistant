"""自检：报告当前可用的嵌入后端与 LLM 后端，并做一次冒烟测试。

运行：
    python3 scripts/verify_backends.py
    python3 scripts/verify_backends.py --smoke     # 额外调用一次 LLM

环境变量速查：
    嵌入： AGENTDOC_EMBEDDER = auto | hash | st
           AGENTDOC_ST_MODEL = BAAI/bge-small-zh-v1.5
    LLM：  AGENTDOC_LLM_PROVIDER = offline | anthropic | deepseek | dashscope |
                                   moonshot | openai | ollama | vllm | auto
           AGENTDOC_BASE_URL / AGENTDOC_API_KEY / AGENTDOC_MODEL
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.llm import LLMClient
from agentdoc.rag.embedder import _st_importable, build_embedder


def human(name: str, val: str) -> None:
    print(f"  {name:<28} {val}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="额外调用一次 LLM")
    args = ap.parse_args()

    print("=" * 62)
    print("1) 嵌入后端")
    print("=" * 62)
    human("sentence-transformers 已安装", "是" if _st_importable() else "否")
    human("AGENTDOC_EMBEDDER", os.environ.get("AGENTDOC_EMBEDDER", "auto"))
    human("目标模型", config.ST_MODEL)
    try:
        emb = build_embedder()
        human("实际选用", f"{emb.name} (dim={emb.dim})")
        v = emb.encode("JVM 是什么")
        human("编码自检", f"shape={v.shape}, norm={float((v ** 2).sum()) ** 0.5:.4f}")
    except Exception as e:
        human("实际选用", f"失败：{e}")

    print()
    print("=" * 62)
    print("2) LLM 后端")
    print("=" * 62)
    human("AGENTDOC_LLM_PROVIDER", os.environ.get("AGENTDOC_LLM_PROVIDER", "auto"))
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
              "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY", "AGENTDOC_BASE_URL"):
        human(k, "已设置" if os.environ.get(k) else "(空)")
    client = LLMClient()
    human("实际选用", f"{client.backend.name} (model={client.model})")
    human("是否真实模型", "是" if client.backend.name != "offline" else "否（离线规则规划器）")

    if args.smoke:
        print()
        print("=" * 62)
        print("3) 冒烟测试（一次 messages.create，无工具）")
        print("=" * 62)
        try:
            resp = client.messages.create(
                system="你是简洁的助手。", messages=[{"role": "user", "content": "用一句话解释什么是 RAG。"}],
                tools=[], max_tokens=200)
            from agentdoc.llm import extract_text
            print("  stop_reason:", resp.stop_reason)
            print("  输出:", extract_text(resp.content)[:300])
        except Exception as e:
            print(f"  调用失败：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
