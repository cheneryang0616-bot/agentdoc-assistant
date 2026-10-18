"""LLM 抽象层。

对外暴露与 Anthropic Messages API 一致的调用形态：

    client = LLMClient()
    resp = client.messages.create(model=..., system=..., messages=..., tools=..., max_tokens=...)
    resp.stop_reason       # "tool_use" | "end_turn"
    resp.content           # list[Block]

设计要点：
  * OfflineBackend —— 纯规则、确定性的规划器，让整个工程在**无网络、无 API Key** 时
    也能真实地产生 tool_use 决策，从而完整演示 AgentLoop / SubAgent 流程。
  * AnthropicBackend —— 若安装了 anthropic 且设置了 ANTHROPIC_API_KEY，自动切换为真模型。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    type: str                                  # "text" | "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    content: List[Block]
    stop_reason: str = "end_turn"
    usage: Dict[str, int] = field(default_factory=dict)


def to_serializable(content: Any) -> Any:
    """把 Block 列表转成可 JSON 序列化的 dict 列表（便于持久化/落盘）。"""
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, Block):
            if b.type == "text":
                out.append({"type": "text", "text": b.text})
            else:
                out.append({"type": b.type, "id": b.id, "name": b.name, "input": b.input})
        else:
            out.append(b)
    return out


def extract_text(content: Any) -> str:
    """从消息 content（str 或 block 列表）里抽取纯文本。"""
    if isinstance(content, str):
        return content
    out: List[str] = []
    for b in content or []:
        if isinstance(b, Block):
            if b.type == "text":
                out.append(b.text)
        elif isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif "text" in b:
                out.append(b["text"])
    return "\n".join(x for x in out if x).strip()


# --------------------------------------------------------------------------- #
# 消息解析辅助
# --------------------------------------------------------------------------- #
def _iter_blocks(messages: List[dict]):
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                yield m.get("role"), b
        elif isinstance(c, str):
            yield m.get("role"), Block("text", text=c)


def _is_text_user(m: dict) -> bool:
    if m.get("role") != "user":
        return False
    c = m.get("content")
    if isinstance(c, str):
        return bool(c.strip())
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text" for b in c)
    return False


def last_question_index(messages: List[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if _is_text_user(messages[i]):
            return i
    return -1


def first_user_question(messages: List[dict]) -> str:
    """取**最新**一条纯文本用户消息（跳过 tool_result 回灌消息）。"""
    i = last_question_index(messages)
    if i < 0:
        return ""
    c = messages[i].get("content")
    return c.strip() if isinstance(c, str) else extract_text(c)


def _blocks_after(messages: List[dict], idx: int) -> List[Any]:
    out = []
    for m in messages[idx + 1:]:
        c = m.get("content")
        if isinstance(c, list):
            out.extend(c)
    return out


def collect_tool_uses(blocks: List[Any]) -> List[Block]:
    res = []
    for b in blocks:
        if isinstance(b, Block) and b.type == "tool_use":
            res.append(b)
        elif isinstance(b, dict) and b.get("type") == "tool_use":
            res.append(Block("tool_use", id=b.get("id", ""), name=b.get("name", ""),
                             input=b.get("input", {})))
    return res


def collect_tool_results(blocks: List[Any]) -> List[dict]:
    return [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]


# --------------------------------------------------------------------------- #
# 离线规则后端
# --------------------------------------------------------------------------- #
_SPLIT_RE = re.compile(r"[，,；;。！!？?\n]|以及|并且|同时|而且")
_QUESTION_WORDS = ("什么", "哪些", "如何", "怎样", "怎么", "为什么", "是否", "介绍",
                   "区别", "多少", "能不能", "可不可以说")

_MEM_WRITE_TRIGGERS = ("记住", "记一下", "帮我记", "保存这个", "记录一下")
_MEM_READ_TRIGGERS = ("之前", "上次", "还记得", "我的偏好", "我喜欢什么", "我以前")
_SUBAGENT_TRIGGERS = ("整理记忆", "记忆整理", "记忆巩固", "dream", "auto-dream", "dreaming")
_CONSOLIDATE_TRIGGERS = ("整理", "巩固", "consolidate", "清理", "合并", "提升")
_TERMINAL_TOOLS = {"memory_write", "memory_read", "subagent", "memory_consolidate"}


def split_subqueries(question: str) -> List[str]:
    """把一个多跳问题拆成若干子问题（模仿 LLM 的问题分解）。"""
    q = re.sub(r"^[?？!！。\s]+", "", question.strip())
    q = re.sub(r"[?？!！。\s]+$", "", q)
    parts = [p.strip() for p in _SPLIT_RE.split(q) if p.strip()]
    parts = [p for p in parts if len(p) >= 2]
    return parts or [q]


def _norm(s: str) -> str:
    return re.sub(r"[\s，,。？?！!；;：:]+", "", s).lower()


class OfflineBackend:
    """确定性规则规划器，行为等价于一个"会用工具的 LLM"。

    决策优先级（仅看**本轮问题之后**的工具调用记录）：
      1) 记忆写入 / 读取指令 -> memory_*
      2) 记忆整理指令        -> subagent / memory_consolidate
      3) 若已用过上述"终止型"工具 -> 直接产出最终回答
      4) 多跳问题仍有未检索的子问题 -> rag_search
      5) 信息足够 -> 产出最终回答
    """

    name = "offline"
    default_model = "offline-planner-v1"

    def __init__(self) -> None:
        self._c = 0

    def _tid(self) -> str:
        self._c += 1
        return f"toolu_{self._c:04d}"

    def create(self, model: str, system: str, messages: List[dict],
               tools: List[dict], max_tokens: int, **kw) -> Response:
        tool_names = {t["name"] for t in tools}
        qidx = last_question_index(messages)
        question = first_user_question(messages)
        blocks = _blocks_after(messages, qidx)
        used = collect_tool_uses(blocks)
        used_names = {u.name for u in used}
        results = collect_tool_results(blocks)

        # 1) 记忆写入
        if ("memory_write" in tool_names and "memory_write" not in used_names
                and any(k in question for k in _MEM_WRITE_TRIGGERS)):
            return Response(
                content=[Block("text", text="我会把这条信息写入长期记忆。"),
                         Block("tool_use", id=self._tid(), name="memory_write",
                               input={"content": self._strip_triggers(question),
                                      "importance": 0.75})],
                stop_reason="tool_use")

        # 2) 记忆读取
        if ("memory_read" in tool_names and "memory_read" not in used_names
                and any(k in question for k in _MEM_READ_TRIGGERS)):
            return Response(
                content=[Block("text", text="我先查一下长期记忆。"),
                         Block("tool_use", id=self._tid(), name="memory_read",
                               input={"query": question, "top_k": 3})],
                stop_reason="tool_use")

        # 3) 记忆整理（SubAgent）
        if ("subagent" in tool_names and "subagent" not in used_names
                and any(k.lower() in question.lower() for k in _SUBAGENT_TRIGGERS)):
            return Response(
                content=[Block("text", text="启动后台 SubAgent 执行记忆整理。"),
                         Block("tool_use", id=self._tid(), name="subagent",
                               input={"task": "consolidate_memory"})],
                stop_reason="tool_use")

        # 3.5) 记忆整理动作（SubAgent 内部使用）
        if ("memory_consolidate" in tool_names and "memory_consolidate" not in used_names
                and any(k.lower() in question.lower() for k in _CONSOLIDATE_TRIGGERS)):
            return Response(
                content=[Block("text", text="执行记忆生命周期整理。"),
                         Block("tool_use", id=self._tid(), name="memory_consolidate",
                               input={"verbose": True})],
                stop_reason="tool_use")

        # 3.9) 已用"终止型"工具 -> 直接收尾，不再检索
        if used_names & _TERMINAL_TOOLS:
            return Response(content=[Block("text", text=self._compose(question, results))],
                            stop_reason="end_turn")

        # 4) RAG 检索（多跳：逐个子问题）
        if "rag_search" in tool_names:
            issued = {_norm(u.input.get("query", "")) for u in used if u.name == "rag_search"}
            for sq in split_subqueries(question):
                if _norm(sq) not in issued:
                    return Response(
                        content=[Block("text", text=f"需要检索：{sq}"),
                                 Block("tool_use", id=self._tid(), name="rag_search",
                                       input={"query": sq, "top_k": 5})],
                        stop_reason="tool_use")

        # 5) 收尾回答
        return Response(content=[Block("text", text=self._compose(question, results))],
                        stop_reason="end_turn")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_triggers(q: str) -> str:
        for k in _MEM_WRITE_TRIGGERS:
            q = q.replace(k, "")
        for pre in ("帮我", "请", "麻烦"):
            q = q.lstrip(" ：:，,。").removeprefix(pre)
        return q.strip(" ：:，,。")

    @staticmethod
    def _compose(question: str, results: List[dict]) -> str:
        if not results:
            return f"我没有获得可用信息来回答「{question}」。"
        lines: List[str] = []
        has_retrieval = False
        for r in results:
            raw = str(r.get("content", ""))
            if raw.startswith("[已落盘"):
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                if obj.get("hits") is not None:
                    has_retrieval = True
                    for h in obj["hits"][:3]:
                        lines.append(f"【{h.get('doc_id')}】{h.get('text', '')[:100]}")
                    continue
                if obj.get("memories") is not None:
                    mems = obj["memories"]
                    if mems:
                        for mm in mems:
                            lines.append(f"[{mm.get('type')}记忆] {mm.get('content', '')[:120]}")
                    else:
                        lines.append("（长期记忆中暂无相关内容）")
                    continue
                if obj.get("ok"):
                    lines.append(f"已写入长期记忆（id={obj.get('id')}, "
                                 f"重要性={obj.get('importance')}）")
                    continue
                if "stats_after" in obj:
                    lines.append(f"记忆整理完成，整理后统计：{obj.get('stats_after')}；"
                                 f"工作->情景 {len(obj.get('working_to_episodic', []))} 条，"
                                 f"情景->语义 {len(obj.get('episodic_to_semantic', []))} 条")
                    continue
            lines.append(raw[:180])
        body = "\n".join(lines[:8]) or "（无有效内容）"
        # 纯检索类问题才加「回答「问题」」包裹；记忆类直接返回结果
        return f"回答「{question}」：\n{body}" if has_retrieval else body


# --------------------------------------------------------------------------- #
# 真实后端（可选）
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # noqa: F401  (延迟导入)
        self._client = anthropic.Anthropic()
        self.default_model = os.environ.get("AGENTDOC_MODEL", "claude-sonnet-4-5")

    def create(self, model, system, messages, tools, max_tokens, **kw) -> Response:
        msgs = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                conv = []
                for b in c:
                    if isinstance(b, Block):
                        if b.type == "text":
                            conv.append({"type": "text", "text": b.text})
                        elif b.type == "tool_use":
                            conv.append({"type": "tool_use", "id": b.id,
                                         "name": b.name, "input": b.input})
                    else:
                        conv.append(b)
                msgs.append({"role": m["role"], "content": conv})
            else:
                msgs.append(m)
        raw = self._client.messages.create(model=model, system=system, messages=msgs,
                                           tools=tools, max_tokens=max_tokens)
        blocks = []
        for b in raw.content:
            if b.type == "text":
                blocks.append(Block("text", text=b.text))
            elif b.type == "tool_use":
                blocks.append(Block("tool_use", id=b.id, name=b.name, input=dict(b.input)))
        return Response(content=blocks, stop_reason=raw.stop_reason)


# --------------------------------------------------------------------------- #
# 真实后端：OpenAI 兼容协议（DeepSeek / Qwen / Moonshot / OpenAI / Ollama / vLLM）
# --------------------------------------------------------------------------- #
class OpenAICompatBackend:
    """只需一个 base_url + api_key 即可对接绝大多数国产/开源模型服务。"""

    name = "openai-compatible"

    PROVIDER_DEFAULTS = {
        # provider: (base_url, api_key_env, default_model)
        "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
        "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                      "DASHSCOPE_API_KEY", "qwen-plus"),
        "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                 "DASHSCOPE_API_KEY", "qwen-plus"),
        "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "moonshot-v1-8k"),
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
        "ollama": ("http://localhost:11434/v1", None, "llama3.1"),
        "vllm": ("http://localhost:8000/v1", None, "Qwen/Qwen2.5-7B-Instruct"),
    }

    def __init__(self, base_url: str, api_key: str = "sk-noauth",
                 model: str | None = None, timeout: int = 90,
                 max_retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "sk-noauth"
        self.default_model = model or "gpt-4o-mini"
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, provider: str | None = None) -> "OpenAICompatBackend":
        provider = (provider or os.environ.get("AGENTDOC_LLM_PROVIDER") or "auto").lower()
        base = os.environ.get("AGENTDOC_BASE_URL")
        key = os.environ.get("AGENTDOC_API_KEY")
        model = os.environ.get("AGENTDOC_MODEL")

        if provider in cls.PROVIDER_DEFAULTS:
            dbase, key_env, dmodel = cls.PROVIDER_DEFAULTS[provider]
            base = base or dbase
            if not key and key_env:
                key = os.environ.get(key_env)
            model = model or dmodel
        else:
            base = base or os.environ.get("OPENAI_BASE_URL")
            key = key or os.environ.get("OPENAI_API_KEY")

        if not base:
            raise RuntimeError(
                "未配置 base_url。请设置 AGENTDOC_LLM_PROVIDER=deepseek|dashscope|moonshot|"
                "openai|ollama，或直接设 AGENTDOC_BASE_URL + AGENTDOC_API_KEY + AGENTDOC_MODEL")
        return cls(base, key or "sk-noauth", model)

    # ------------------------------------------------------------------ #
    def create(self, model, system, messages, tools, max_tokens, **kw) -> Response:
        payload = {
            "model": model or self.default_model,
            "messages": self._convert_messages(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        data = self._post(payload)
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        blocks: List[Block] = []
        if msg.get("content"):
            blocks.append(Block("text", text=msg["content"]))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"_raw": raw_args}
            blocks.append(Block("tool_use", id=tc.get("id", ""),
                                name=fn.get("name", ""), input=args))
        stop = "tool_use" if (msg.get("tool_calls")) else "end_turn"
        return Response(content=blocks, stop_reason=stop)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_tools(tools: List[dict]) -> List[dict]:
        out = []
        for t in tools:
            out.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            }})
        return out

    @staticmethod
    def _convert_messages(system: str, messages: List[dict]) -> List[dict]:
        out: List[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role", "user")
            c = m.get("content")
            if isinstance(c, str):
                out.append({"role": role, "content": c})
                continue

            texts, tool_uses, tool_results = [], [], []
            for b in c or []:
                if isinstance(b, Block):
                    if b.type == "text":
                        texts.append(b.text)
                    elif b.type == "tool_use":
                        tool_uses.append(b)
                elif isinstance(b, dict):
                    t = b.get("type")
                    if t == "text":
                        texts.append(b.get("text", ""))
                    elif t == "tool_use":
                        tool_uses.append(Block("tool_use", id=b.get("id", ""),
                                               name=b.get("name", ""),
                                               input=b.get("input", {})))
                    elif t == "tool_result":
                        tool_results.append(b)

            if role == "assistant":
                msg: Dict[str, Any] = {"role": "assistant",
                                       "content": "\n".join(texts) or None}
                if tool_uses:
                    msg["tool_calls"] = [{
                        "id": u.id, "type": "function",
                        "function": {"name": u.name,
                                     "arguments": json.dumps(u.input, ensure_ascii=False)},
                    } for u in tool_uses]
                out.append(msg)
            else:
                if texts:
                    out.append({"role": "user", "content": "\n".join(texts)})
                for r in tool_results:
                    out.append({"role": "tool",
                                "tool_call_id": r.get("tool_use_id", ""),
                                "content": str(r.get("content", ""))})
        return out

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import time
        import urllib.error
        import urllib.request

        url = self.base_url + "/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {self.api_key}")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:500]
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code in (408, 429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last
            except Exception as e:  # 网络异常
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise last if last else RuntimeError("请求失败")


# --------------------------------------------------------------------------- #
# 统一客户端
# --------------------------------------------------------------------------- #
class _Messages:
    def __init__(self, outer: "LLMClient") -> None:
        self._o = outer

    def create(self, model: Optional[str] = None, system: str = "",
               messages: Optional[List[dict]] = None, tools: Optional[List[dict]] = None,
               max_tokens: int = 8000, **kw) -> Response:
        return self._o.backend.create(model or self._o.model, system, messages or [],
                                      tools or [], max_tokens, **kw)


class LLMClient:
    def __init__(self, backend=None, model: Optional[str] = None) -> None:
        self.backend = backend or build_default_backend()
        self.model = model or getattr(self.backend, "default_model", None) or config.LLM_MODEL
        self.messages = _Messages(self)


def build_default_backend():
    """按环境变量选择后端：
        AGENTDOC_LLM_PROVIDER = offline | anthropic | deepseek | dashscope | qwen |
                                moonshot | openai | ollama | vllm | auto
    auto：有任意可用 Key/URL 则用真实模型，否则回退离线规划器。
    """
    provider = (os.environ.get("AGENTDOC_LLM_PROVIDER") or "auto").lower()

    if provider in ("offline", "rule", "mock"):
        return OfflineBackend()
    if provider == "anthropic":
        return AnthropicBackend()
    if provider in OpenAICompatBackend.PROVIDER_DEFAULTS or provider in (
            "openai-compatible", "compat"):
        return OpenAICompatBackend.from_env(provider)

    # auto
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            pass
    if any(os.environ.get(k) for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                                        "DASHSCOPE_API_KEY", "MOONSHOT_API_KEY")) \
            or os.environ.get("AGENTDOC_BASE_URL"):
        try:
            return OpenAICompatBackend.from_env("auto")
        except Exception:
            pass
    return OfflineBackend()
