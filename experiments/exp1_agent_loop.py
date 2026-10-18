"""实验 1：AgentLoop vs 传统单次 RAG —— 多跳问题覆盖率。

分别在**弱检索器（离线哈希）**与**强检索器（BGE）**下对比：
  单次 RAG：整句一次检索 top-5
  AgentLoop：按子问题逐次检索

结论会随检索器强弱变化——这本身就是一个值得讲的发现。

运行：  python3 experiments/exp1_agent_loop.py
产出：  results/exp1_agent_loop.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.rag import build_embedder
from agentdoc.session import AgentSession

MULTIHOP = [
    {"q": "JVM 是什么？它在生产环境中有哪些典型应用场景？",
     "expect": {"jvm_concept", "jvm_application"}},
    {"q": "Python 的 GIL 是什么？Java 的线程模型又是怎样的？",
     "expect": {"python_gil", "java_thread"}},
    {"q": "RAG 的基本流程是什么？Agent 是如何定义的？",
     "expect": {"rag_intro", "agent_intro"}},
    {"q": "JVM 的垃圾回收机制如何？Python 的垃圾回收又怎样？",
     "expect": {"jvm_gc", "python_gc"}},
]


def coverage(expected, got) -> float:
    return len(expected & set(got)) / len(expected) if expected else 0.0


def eval_with(emb_name: str):
    sess = AgentSession(embedder=build_embedder(emb_name))
    rows = []
    for item in MULTIHOP:
        q, exp = item["q"], item["expect"]

        one_shot_docs = [h.doc_id for h in sess.retriever.advanced_search(q, top_k=5)]
        c1 = coverage(exp, one_shot_docs)

        res = sess.ask(q, keep_history=False)
        c2 = coverage(exp, res["retrieved"])

        rows.append({
            "query": q,
            "expected": sorted(exp),
            "one_shot_docs": sorted(set(one_shot_docs)),
            "one_shot_coverage": round(c1, 4),
            "agent_loop_turns": res["turns"],
            "agent_loop_docs": sorted(set(res["retrieved"])),
            "agent_loop_coverage": round(c2, 4),
        })

    n = len(rows)
    return {
        "embedder": getattr(sess.embedder, "name", emb_name),
        "dim": int(getattr(sess.embedder, "dim", 0)),
        "n_questions": n,
        "avg_one_shot_coverage": round(sum(r["one_shot_coverage"] for r in rows) / n, 4),
        "avg_agent_loop_coverage": round(sum(r["agent_loop_coverage"] for r in rows) / n, 4),
        "improvement": round(sum(r["agent_loop_coverage"] - r["one_shot_coverage"]
                                 for r in rows) / n, 4),
        "rows": rows,
    }


def main() -> None:
    out = {"by_embedder": {}}
    for tag in ("hash", "st"):
        try:
            r = eval_with(tag)
        except Exception as e:
            print(f"[跳过] {tag}: {e}\n")
            continue
        out["by_embedder"][tag] = r
        print(f"== 检索器：{r['embedder']} (dim={r['dim']}) ==")
        for row in r["rows"]:
            print(f"  {row['one_shot_coverage']:.2f} -> {row['agent_loop_coverage']:.2f} "
                  f"(轮数 {row['agent_loop_turns']}) | {row['query'][:34]}")
        print(f"  平均覆盖率：单次RAG {r['avg_one_shot_coverage']:.3f} -> "
              f"AgentLoop {r['avg_agent_loop_coverage']:.3f} "
              f"({r['improvement']:+.3f})\n")

    p = config.RESULTS_DIR / "exp1_agent_loop.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {p}")


if __name__ == "__main__":
    main()
