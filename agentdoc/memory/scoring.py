"""记忆混合评分：时间衰减 + 重要性权重 + 语义相关度。

对应简历 ①「引入基于时间衰减与重要性权重的混合评分机制」。

时间衰减采用指数衰减：
    decay(Δt) = 0.5 ** (Δt / half_life)
其中 Δt 为距上次访问 / 创建的小时数。半衰期越短，遗忘越快。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import config


def time_decay(age_hours: float, half_life: float = config.MEMORY_HALF_LIFE_HOURS) -> float:
    if half_life <= 0:
        return 1.0
    return float(0.5 ** (max(0.0, age_hours) / half_life))


def hybrid_score(importance: float, age_hours: float, relevance: float = 0.0,
                 w_relevance: float = 0.55, w_importance: float = 0.30,
                 w_recency: float = 0.15,
                 half_life: float = config.MEMORY_HALF_LIFE_HOURS) -> float:
    """混合评分 = w_r * 语义相关度 + w_i * 重要性 + w_t * 时间新鲜度。

    可用于记忆检索排序，也可用于"是否值得提升为长期记忆"的判定。
    """
    rec = time_decay(age_hours, half_life)
    imp = float(np.clip(importance, 0.0, 1.0))
    rel = float(np.clip(relevance, -1.0, 1.0))
    # 相关度映射到 [0,1]
    rel01 = (rel + 1.0) / 2.0
    return w_relevance * rel01 + w_importance * imp + w_recency * rec
