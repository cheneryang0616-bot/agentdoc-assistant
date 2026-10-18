"""Agent 会话装配层：把 RAG / 记忆 / 工具 / 上下文管理 / AgentLoop 组装成业务闭环。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from . import config
from .context import ContextManager
from .grounding import GroundingChecker
from .llm import LLMClient, extract_text
from .loop import LoopState, agent_loop
from .memory import MemoryStore
from .memory.lifecycle import consolidate
from .rag import HashingEmbedder, Retriever, VectorStore, build_embedder, load_knowledge_base
from .subagent import CONSOLIDATE_PROMPT, SUBAGENT_SYSTEM, run_subagent
from .tools import ToolRegistry
from .tools.handlers import (build_file_read, build_memory_tools, build_rag_search,
                             build_subagent_tool)

MAIN_SYSTEM = """你是一个智能文档问答助手，具备检索（rag_search）与长期记忆（memory）能力。

工具使用规则：
1. 需要事实性知识时，调用 rag_search；遇到多跳问题，拆成多个子问题分步检索后再综合回答。
2. 用户要求“记住/记一下/保存”某信息时，调用 memory_write 写入长期记忆。
3. 用户询问“之前/上次说过什么、我的偏好”时，先调用 memory_read。
4. 用户要求“整理记忆/巩固记忆”时，调用 subagent 启动后台整理。
5. 信息足够时直接回答，不要无意义地反复调用工具。
6. **只依据检索结果作答**；若检索结果与问题无关或不足以回答，明确说明知识库中没有相关内容，
   不要用自己的通用知识补充，也不要编造数值。"""


class AgentSession:
    def __init__(self, memory_dir: Optional[Path] = None, embedder=None,
                 use_advanced_retrieval: bool = True,
                 context_budget: int = config.CONTEXT_TOKEN_BUDGET) -> None:
        config.ensure_dirs()
        self.embedder = embedder or build_embedder()
        self.use_advanced = use_advanced_retrieval
        self.client = LLMClient()

        # --- 知识库索引 ---
        self.chunks = load_knowledge_base()
        # 在语料上拟合 IDF（若嵌入器支持）
        if hasattr(self.embedder, "fit"):
            self.embedder.fit([self._index_text(c) for c in self.chunks])
        self.store = VectorStore()
        for c in self.chunks:
            self.store.add(c.chunk_id, c.text, self.embedder.encode(self._index_text(c)),
                           kind="doc", doc_id=c.doc_id, title=c.title)
        self.retriever = Retriever(self.store, self.embedder)

        # --- 记忆 ---
        self.memory = MemoryStore(memory_dir, embedder=self.embedder)

        # --- 上下文管理（需在注册工具前建好，供 file_read 使用） ---
        self.cm = ContextManager(budget=context_budget, client=self.client)
        self.grounding = GroundingChecker(self.client, enabled=config.GROUNDING)
        self.history: List[dict] = []

        # --- 主工具表 ---
        self.registry = ToolRegistry()
        self._register_main_tools()

        # --- 子 Agent 工具表（SubAgent 专用） ---
        self.child_registry = ToolRegistry()
        self._register_child_tools()

    # ------------------------------------------------------------------ #
    def _index_text(self, c) -> str:
        # 哈希嵌入需要标题加权（重复 3 次）；真实嵌入模型吃原文即可
        if str(getattr(self.embedder, "name", "")).startswith("hashing"):
            return f"{c.doc_title} {c.doc_title} {c.doc_title} {c.title} {c.text}"
        return f"{c.doc_title} {c.title} {c.text}"

    # ------------------------------------------------------------------ #
    def _register_main_tools(self) -> None:
        r = self.registry
        r.register("rag_search",
                   "在知识库中做语义检索。支持多查询扩展(MQE)与假设文档嵌入(HyDE)融合。",
                   {"type": "object", "properties": {
                       "query": {"type": "string", "description": "检索问题"},
                       "top_k": {"type": "integer", "description": "返回条数，默认 5"}},
                    "required": ["query"]},
                   build_rag_search(self.retriever, self.use_advanced))

        mw, mr = build_memory_tools(self.memory)
        r.register("memory_write",
                   "把一条重要信息写入长期记忆。",
                   {"type": "object", "properties": {
                       "content": {"type": "string"},
                       "importance": {"type": "number", "description": "0~1"}},
                    "required": ["content"]}, mw)
        r.register("memory_read",
                   "从长期记忆中检索相关记忆。",
                   {"type": "object", "properties": {
                       "query": {"type": "string"},
                       "top_k": {"type": "integer"}},
                    "required": ["query"]}, mr)
        r.register("subagent",
                   "启动子 Agent 执行后台任务（如记忆整理 / 巩固）。",
                   {"type": "object", "properties": {
                       "task": {"type": "string"}},
                    "required": ["task"]},
                   build_subagent_tool(self._run_subagent))
        r.register("file_read",
                   "读取已落盘的工具结果文件（上下文压缩 L1 使用）。",
                   {"type": "object", "properties": {
                       "path": {"type": "string"},
                       "max_chars": {"type": "integer"}},
                    "required": ["path"]},
                   build_file_read(self.cm.offload_dir))

    def _register_child_tools(self) -> None:
        r = self.child_registry
        mw, mr = build_memory_tools(self.memory)
        r.register("memory_read", "读取长期记忆。",
                   {"type": "object", "properties": {
                       "query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"]}, mr)
        r.register("memory_write", "写入长期记忆。",
                   {"type": "object", "properties": {
                       "content": {"type": "string"},
                       "importance": {"type": "number"}},
                    "required": ["content"]}, mw)
        r.register("memory_consolidate",
                   "对现有记忆执行一次生命周期整理（提升/合并/清理）。",
                   {"type": "object",
                    "properties": {"verbose": {"type": "boolean"}},
                    "required": []},
                   lambda verbose=True: json.dumps(consolidate(self.memory, verbose=bool(verbose)),
                                                  ensure_ascii=False, default=str))
        r.register("file_read", "读取落盘文件。",
                   {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
                   build_file_read(self.cm.offload_dir))

    # ------------------------------------------------------------------ #
    def _run_subagent(self, task: str = "consolidate_memory") -> str:
        prompt = CONSOLIDATE_PROMPT if task in ("consolidate_memory", "记忆整理") else task
        return run_subagent(prompt, self.client, self.child_registry,
                            system=SUBAGENT_SYSTEM)

    # ------------------------------------------------------------------ #
    def history_view(self) -> List[dict]:
        return self.history

    def reset(self) -> None:
        self.history: List[dict] = []

    def ask(self, query: str, keep_history: bool = True, max_turns: int = 12) -> dict:
        if not hasattr(self, "history"):
            self.history = []
        if not keep_history:
            self.history = []

        # 每次交互之前做上下文压缩
        self.history, rep = self.cm.compress(self.history)
        start = len(self.history)          # 本轮从哪开始
        self.history.append({"role": "user", "content": query})

        # 问题级相关性闸门：整句明显库外时直接拒答，不进入 AgentLoop
        if config.MIN_SCORE > 0:
            top = self.retriever.advanced_search(query, 5)
            best = top[0].score if top else 0.0
            if best < config.MIN_SCORE:
                ans = (f"知识库中没有找到与「{query}」相关的内容"
                       f"（相关性 {best:.3f} < 阈值 {config.MIN_SCORE}）。")
                self.history.append({"role": "assistant", "content": ans})
                return {"query": query, "answer": ans, "turns": 0, "trace": [],
                        "retrieved": [], "retrieved_texts": [],
                        "compression": rep.as_dict(), "history_len": len(self.history)}

        state = LoopState(messages=self.history)
        agent_loop(state, self.client, self.registry, MAIN_SYSTEM, max_turns=max_turns)
        self.history = state.messages

        answer = extract_text(self.history[-1]["content"])

        # 只收集**本轮**检索到的文档与原文（从 tool_result 解析）
        retrieved: List[str] = []
        retrieved_texts: List[str] = []
        for m in state.messages[start:]:
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    try:
                        obj = json.loads(str(b.get("content", "")))
                        for h in obj.get("hits", []):
                            retrieved.append(h["doc_id"])
                            if h.get("text"):
                                retrieved_texts.append(h["text"])
                    except Exception:
                        pass

        # 答案级 groundedness 质检：不合格则替换为拒答
        grounding = None
        answer, grounding = self.grounding.enforce(query, retrieved_texts, answer)
        if grounding and (grounding.get("skipped") is False) and not grounding.get("ok", True):
            self.history[-1] = {"role": "assistant", "content": answer}

        return {
            "query": query,
            "answer": answer,
            "turns": state.turn_count,
            "trace": state.trace,
            "retrieved": retrieved,
            "retrieved_texts": retrieved_texts,
            "grounding": grounding,
            "compression": rep.as_dict(),
            "history_len": len(self.history),
        }
