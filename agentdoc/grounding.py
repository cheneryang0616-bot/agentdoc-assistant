"""答案级 groundedness 校验（LLM-as-judge）。

失败分析表明：可靠性问题不是"检索不到"，而是"知识库根本没有答案时仍然作答"。
检索分数、词面覆盖率、语义相似度都无法区分"同主题但无答案"——
必须让模型同时看【问题】【检索片段】【候选答案】，判断：
    answerable —— 片段是否足以回答问题
    grounded   —— 答案是否完全由片段支撑

只输出 JSON，便于程序化拦截。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .llm import extract_text

GROUND_SYSTEM = """你是 RAG 答案质检员。请严格判断两件事：
1) answerable：给定的【检索片段】是否**足以**回答【问题】（片段里得有直接依据，不能靠常识补）；
2) grounded：候选答案中的每个事实性陈述是否都能在【检索片段】中找到依据。
要求：
- 问题涉及片段中从未出现的概念/数值/专有名词时，answerable=false；
- 答案包含片段之外的知识、或编造数值时，grounded=false；
- 只输出 JSON，不要解释，格式：{"answerable": true/false, "grounded": true/false, "reason": "一句话"}"""


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class GroundingChecker:
    def __init__(self, client, enabled: bool = True, max_context_chars: int = 2000,
                 max_answer_chars: int = 1200) -> None:
        self.client = client
        self.enabled = enabled
        self.max_context_chars = max_context_chars
        self.max_answer_chars = max_answer_chars

    # ------------------------------------------------------------------ #
    def check(self, question: str, contexts: List[str], answer: str) -> Dict[str, Any]:
        ctx = "\n---\n".join(contexts)[: self.max_context_chars]
        prompt = (f"【问题】{question}\n\n【检索片段】\n{ctx}\n\n"
                  f"【候选答案】\n{answer[: self.max_answer_chars]}")
        try:
            resp = self.client.messages.create(
                system=GROUND_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=[], max_tokens=300)
            obj = _parse_json(extract_text(resp.content))
        except Exception as e:
            return {"ok": True, "skipped": True, "reason": f"质检调用失败: {e}"}
        if not obj:
            return {"ok": True, "skipped": True, "reason": "质检返回无法解析"}
        answerable = bool(obj.get("answerable", True))
        grounded = bool(obj.get("grounded", True))
        return {"ok": answerable and grounded, "answerable": answerable,
                "grounded": grounded, "reason": str(obj.get("reason", ""))[:200]}

    # ------------------------------------------------------------------ #
    def enforce(self, question: str, contexts: List[str], answer: str):
        """返回 (最终答案, 质检结果)。不合格时替换为拒答。"""
        if not self.enabled or not contexts or not answer.strip():
            return answer, None
        v = self.check(question, contexts, answer)
        if v.get("skipped") or v.get("ok"):
            return answer, v
        why = v.get("reason") or ("检索片段不足以支撑该回答"
                                  if not v.get("answerable") else "答案包含片段之外的内容")
        return f"知识库中没有足够信息回答「{question}」（质检未通过：{why}）。", v
