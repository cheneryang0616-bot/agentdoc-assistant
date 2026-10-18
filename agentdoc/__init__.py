"""agentdoc —— 智能文档问答 Agent（AgentLoop + 多级记忆 + RAG + 微调复现）。

模块划分：
    llm        : LLM 抽象层（离线规则后端 / Anthropic 后端可插拔）
    rag        : 多模态规范化 -> 语义分块 -> 384 维嵌入 -> MQE/HyDE 高级检索
    memory     : 工作 / 情景 / 语义 三级记忆 + 生命周期管理 + 混合评分
    tools      : 工具注册表（RAG Search / Memory / SubAgent）
    context    : 三级上下文压缩管理器
    subagent   : 独立上下文的子 Agent（用于记忆整理）
    loop       : AgentLoop（解决多跳问题）
"""
from __future__ import annotations

__version__ = "1.0.0"
