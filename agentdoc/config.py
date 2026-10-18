"""全局配置 —— 所有可调参数集中于此。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "kb"
RESULTS_DIR = ROOT / "results"
MEMORY_DIR = RESULTS_DIR / "memory"
OFFLOAD_DIR = RESULTS_DIR / "offload"

# ---- 检索 / 向量 ----
# 说明：原项目使用 384 维嵌入（MiniLM 系）。本复现为离线哈希模拟（零依赖、可复现），
# 384 维桶过少会导致哈希碰撞、稀释判别性词，故模拟器采用 1024 维保证检索质量；
# fix_dim() 仍完整支持任意维度的动态补齐/截断（含 384），并有单元验证。
EMBED_DIM = 1024
CHUNK_MAX_CHARS = 420
CHUNK_OVERLAP = 60
TOP_K = 5
RRF_K = 60                     # Reciprocal Rank Fusion 常数

# ---- 记忆 ----
MEMORY_HALF_LIFE_HOURS = 72.0  # 时间衰减半衰期
IMPORTANCE_PROMOTE_THRESHOLD = 0.60   # 工作记忆 -> 长期记忆 的重要性阈值
SEMANTIC_MERGE_THRESHOLD = 0.45       # 语义记忆合并的余弦相似度阈值

# ---- 上下文管理（三级压缩） ----
CONTEXT_TOKEN_BUDGET = 2600    # 单次交互允许的上下文 token 预算
OFFLOAD_CHARS = 400            # L1：工具结果超过此长度即落盘
KEEP_RECENT_TURNS = 2          # L2：保留最近 N 轮完整 QA

# ---- LLM ----
LLM_MODEL = os.environ.get("AGENTDOC_MODEL", "offline-planner-v1")
SUBAGENT_MODEL = os.environ.get("AGENTDOC_SUBAGENT_MODEL", LLM_MODEL)

# ---- 嵌入后端 ----
# AGENTDOC_EMBEDDER = auto | hash | st
#   auto: 已安装 sentence-transformers 且模型已缓存 -> 用真实模型，否则哈希
EMBEDDER = os.environ.get("AGENTDOC_EMBEDDER", "auto")
ST_MODEL = os.environ.get("AGENTDOC_ST_MODEL", "BAAI/bge-small-zh-v1.5")


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, MEMORY_DIR, OFFLOAD_DIR, KB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def approx_tokens(text: str) -> int:
    """粗略 token 估计：中文约 1 字 1 token，英文约 4 字符 1 token。
    这里用混合近似 len/1.6，足够做上下文预算判断。"""
    if not text:
        return 0
    return max(1, int(len(text) / 1.6))
