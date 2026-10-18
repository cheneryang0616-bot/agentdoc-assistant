"""SubAgent：拥有独立（fresh）上下文的子 Agent，用于记忆整理。

对应方案 2.3 与参考代码 run_subagent()。
"""
from __future__ import annotations

from typing import List, Optional

from . import config
from .llm import Block, LLMClient, extract_text

SUBAGENT_SYSTEM = """你是记忆整理 SubAgent（MEMORY_CONSOLIDATOR）。
职责：读取现有记忆，删除过于久远的、合并过于细致的记忆，
并把高价值记忆提升为长期记忆。完成后给出简短整理报告。"""

CONSOLIDATE_PROMPT = ("整理记忆：请对现有记忆进行巩固（consolidate）。"
                      "删除过久远的记忆、合并重复或过于细致的记忆、"
                      "并将高价值短期记忆提升为长期记忆。")


def run_subagent(prompt: str, client: LLMClient, registry,
                 system: str = SUBAGENT_SYSTEM,
                 max_turns: int = 12, max_tokens: int = 8000,
                 model: Optional[str] = None) -> str:
    """独立上下文的子 Agent 循环（与主 Agent 完全隔离）。"""
    sub_messages: List[dict] = [{"role": "user", "content": prompt}]  # fresh context
    response = None
    for _ in range(max_turns):  # safety limit
        response = client.messages.create(
            model=model or client.model, system=system,
            messages=sub_messages, tools=registry.definitions(), max_tokens=max_tokens)
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if isinstance(block, Block) and block.type == "tool_use":
                output = registry.call(block.name, block.input)   # subagent 也可以调用工具
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(output)[:50000]})
        if not results:
            break
        sub_messages.append({"role": "user", "content": results})

    # 若最后一轮没有文本，退回上一条 assistant
    text = extract_text(response.content) if response else ""
    if not text and response is not None and response.stop_reason == "tool_use":
        text = "记忆整理已完成。"
    return text
