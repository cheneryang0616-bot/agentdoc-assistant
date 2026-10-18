"""AgentLoop —— 解决多跳问题。

对应方案 2.1 的参考代码，逐行保留其语义（run_one_turn / agent_loop），
并补充：工具执行轨迹 trace、最大轮数保护。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .llm import Block, LLMClient, to_serializable


@dataclass
class LoopState:
    messages: List[dict]
    turn_count: int = 0
    transition_reason: Optional[str] = None
    trace: List[dict] = field(default_factory=list)


def run_one_turn(state: LoopState, client: LLMClient, registry,
                 system: str, max_tokens: int = 8000,
                 model: Optional[str] = None) -> bool:
    """跑一轮：LLM 决策 -> 若需工具则执行并回灌 -> 返回是否继续。"""
    response = client.messages.create(
        model=model or client.model,
        system=system,
        messages=state.messages,
        tools=registry.definitions(),
        max_tokens=max_tokens,
    )
    state.messages.append({"role": "assistant", "content": to_serializable(response.content)})

    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    results = []
    for block in response.content:
        if isinstance(block, Block) and block.type == "tool_use":
            output = registry.call(block.name, block.input)
            state.trace.append({
                "turn": state.turn_count + 1,
                "tool": block.name,
                "input": block.input,
                "output_chars": len(str(output)),
            })
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(output)[:50000]})

    if not results:
        state.transition_reason = None
        return False

    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True


def agent_loop(state: LoopState, client: LLMClient, registry, system: str,
               max_turns: int = 12, max_tokens: int = 8000,
               model: Optional[str] = None) -> LoopState:
    while state.turn_count < max_turns and run_one_turn(
            state, client, registry, system, max_tokens, model):
        pass
    return state
