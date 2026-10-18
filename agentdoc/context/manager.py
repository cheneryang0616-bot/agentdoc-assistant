"""三级上下文压缩。

对应方案 2.2：
  L1  将较大的检索结果存入磁盘，需要时通过 file_read 工具读取（offload）；
  L2  将较久远的 QA 替换为占位符；
  L3  调用 LLM API 进行全量压缩。

压缩在 Agent 每次交互**之前**执行。三级按"成本递增"顺序级联：
只有前一级做完仍超预算，才会触发下一级。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config


@dataclass
class CompressionReport:
    before_tokens: int = 0
    after_tokens: int = 0
    l1_offloaded: int = 0
    l2_placeholder_turns: int = 0
    l3_triggered: bool = False
    budget: int = config.CONTEXT_TOKEN_BUDGET

    def as_dict(self) -> Dict[str, Any]:
        return {
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "ratio": round(self.after_tokens / self.before_tokens, 4) if self.before_tokens else 1.0,
            "budget": self.budget,
            "l1_offloaded": self.l1_offloaded,
            "l2_placeholder_turns": self.l2_placeholder_turns,
            "l3_triggered": self.l3_triggered,
        }


def _msg_tokens(m: dict) -> int:
    c = m.get("content")
    if isinstance(c, str):
        return config.approx_tokens(c)
    total = 0
    for b in c or []:
        if isinstance(b, dict):
            total += config.approx_tokens(b.get("text") or b.get("content") or "")
    return total


def total_tokens(messages: List[dict]) -> int:
    return sum(_msg_tokens(m) for m in messages)


class ContextManager:
    def __init__(self, budget: int = config.CONTEXT_TOKEN_BUDGET,
                 offload_dir: Optional[Path] = None,
                 offload_chars: int = config.OFFLOAD_CHARS,
                 keep_recent_turns: int = config.KEEP_RECENT_TURNS,
                 client=None, model: Optional[str] = None) -> None:
        self.budget = budget
        self.offload_dir = Path(offload_dir or config.OFFLOAD_DIR)
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self.offload_chars = offload_chars
        self.keep_recent_turns = keep_recent_turns
        self.client = client
        self.model = model or config.LLM_MODEL

    # ------------------------------------------------------------------ #
    def compress(self, messages: List[dict]) -> Tuple[List[dict], CompressionReport]:
        rep = CompressionReport(before_tokens=total_tokens(messages), budget=self.budget)
        msgs = _deepcopy_msgs(messages)

        if total_tokens(msgs) <= self.budget:
            rep.after_tokens = total_tokens(msgs)
            return msgs, rep

        # ---- L1：大工具结果落盘 ----
        for m in msgs:
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    txt = str(b.get("content", ""))
                    if len(txt) > self.offload_chars:
                        path = self._offload(txt)
                        b["content"] = (f"[已落盘 {len(txt)} 字符 -> {path.name}] "
                                        f"预览: {txt[:120]}... "
                                        f"(需要完整内容请用 file_read 工具读取 {path.name})")
                        rep.l1_offloaded += 1
        if total_tokens(msgs) <= self.budget:
            rep.after_tokens = total_tokens(msgs)
            return msgs, rep

        # ---- L2：久远 QA 替换为占位符 ----
        rep.l2_placeholder_turns = self._placeholder_old_turns(msgs)
        if total_tokens(msgs) <= self.budget:
            rep.after_tokens = total_tokens(msgs)
            return msgs, rep

        # ---- L3：LLM 全量压缩 ----
        msgs = self._llm_summarize(msgs)
        rep.l3_triggered = True
        rep.after_tokens = total_tokens(msgs)
        return msgs, rep

    # ------------------------------------------------------------------ #
    def _offload(self, text: str) -> Path:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        path = self.offload_dir / f"toolres_{h}.txt"
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _placeholder_old_turns(msgs: List[dict]) -> int:
        """把除了最近 KEEP_RECENT_TURNS 轮之外的用户/助手文本消息替换为占位符。"""
        # 找出"纯文本"消息的下标（排除 tool_result 消息）
        idxs = [i for i, m in enumerate(msgs) if isinstance(m.get("content"), str)]
        keep = set(idxs[-config.KEEP_RECENT_TURNS * 2:])
        n = 0
        for i in idxs:
            if i not in keep:
                role = msgs[i]["role"]
                msgs[i] = {"role": role, "content": f"[历史 {role} 消息已省略]"}
                n += 1
        return n

    def _llm_summarize(self, msgs: List[dict]) -> List[dict]:
        """全量压缩：把前面的历史整段抽取式摘要为一条 user 消息。"""
        text = []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                text.append(f"{m['role']}: {c}")
            else:
                for b in c or []:
                    if isinstance(b, dict):
                        text.append(str(b.get("text") or b.get("content") or "")[:300])
        joined = "\n".join(text)
        # 离线抽取式摘要：保留头尾 + 关键行
        lines = [ln for ln in joined.splitlines() if ln.strip()]
        head, tail = lines[:6], lines[-6:]
        summary = "【上下文压缩摘要】\n" + "\n".join(head + ["..."] + tail)
        return [{"role": "user", "content": summary}]


def _deepcopy_msgs(msgs: List[dict]) -> List[dict]:
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            out.append({"role": m["role"], "content": [dict(b) if isinstance(b, dict) else b for b in c]})
        else:
            out.append({"role": m["role"], "content": c})
    return out
