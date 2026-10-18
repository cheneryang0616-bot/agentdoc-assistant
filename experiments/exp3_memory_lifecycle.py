"""实验 3：记忆生命周期管理 + 混合评分（时间衰减 / 重要性 / 相关度）。

运行：  python3 experiments/exp3_memory_lifecycle.py
产出：  results/exp3_memory_lifecycle.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.memory import MemoryStore
from agentdoc.memory.lifecycle import consolidate
from agentdoc.memory.scoring import time_decay
from agentdoc.rag import HashingEmbedder


def main() -> None:
    config.ensure_dirs()
    emb = HashingEmbedder()
    emb.fit(["JVM 垃圾回收", "Python 引用计数", "RAG 检索增强", "Agent 智能体",
             "Java 线程模型", "Python GIL"])
    store = MemoryStore(config.RESULTS_DIR / "memory_demo", embedder=emb)

    # 清空演示库
    for t in ("working", "episodic", "semantic"):
        for it in list(store.by_type(t)):
            store.delete(it.id)

    now = time.time()
    specs = [
        # (内容, 类型, 重要性, 年龄小时)
        ("用户偏好用中文回答", "working", 0.90, 2),
        ("用户关注 Java 生态", "working", 0.80, 60),
        ("临时寒暄：今天天气不错", "working", 0.10, 200),
        ("用户在做 RAG 项目", "working", 0.65, 10),
        ("用户问过 JVM 垃圾回收", "episodic", 0.55, 5),
        ("用户问过 Python GIL", "episodic", 0.50, 5),
        ("JVM 垃圾回收依赖分代假说", "semantic", 0.70, 30),
        ("JVM 的垃圾回收基于分代假说", "semantic", 0.60, 30),  # 与上条近似重复 -> 将被合并
    ]
    for content, typ, imp, age_h in specs:
        it = store.add(content, type=typ, importance=imp)
        it.created_at = now - age_h * 3600
        it.last_access = it.created_at
    # 让"用户问过 Python GIL"被访问两次 -> 触发情景->语义
    for it in store.by_type("episodic"):
        if "Python GIL" in it.content:
            it.access_count = 2
    store.flush()

    print("== 整理前 ==")
    print("统计:", store.stats())

    # 混合评分演示
    print("\n== 混合评分（query='用户关注的垃圾回收偏好'）==")
    for m in store.search("用户关注的垃圾回收偏好", top_k=4):
        print(f"   score={m.meta['_last_score']:.3f} rel={m.meta['_last_relevance']:+.3f} "
              f"imp={m.importance:.2f} type={m.type}  {m.content[:24]}")

    # 时间衰减曲线
    print("\n== 时间衰减（半衰期 %.0f h）==" % config.MEMORY_HALF_LIFE_HOURS)
    curve = []
    for h in (0, 12, 24, 48, 72, 144, 336):
        v = time_decay(h)
        curve.append({"age_hours": h, "decay": round(v, 4)})
        print(f"   {h:>4}h -> {v:.4f}")

    # 生命周期整理
    print("\n== 执行 consolidate() ==")
    report = consolidate(store, verbose=True)
    print("整理后统计:", report["stats_after"])
    print("工作->情景:", report["working_to_episodic"])
    print("情景->语义:", report["episodic_to_semantic"])
    print("语义合并对:", report["semantic_merged"])

    out = config.RESULTS_DIR / "exp3_memory_lifecycle.json"
    out.write_text(json.dumps({"report": report, "decay_curve": curve,
                               "promote_threshold": config.IMPORTANCE_PROMOTE_THRESHOLD,
                               "merge_threshold": config.SEMANTIC_MERGE_THRESHOLD},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[写出] {out}")


if __name__ == "__main__":
    main()
