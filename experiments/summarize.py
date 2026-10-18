"""汇总各实验结果，生成 results/RESULTS.md（所有数字来自真实运行）。

运行：  python3 experiments/summarize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config


def load(name):
    p = config.RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    e1 = load("exp1_agent_loop.json")
    e2 = load("exp2_context_compression.json")
    e3 = load("exp3_memory_lifecycle.json")
    e4 = load("exp4_finetune_embedding.json")
    e5 = load("exp5_embedder_compare.json")

    L = ["# 复现实验结果（全部为真实运行产出）", "",
         "> 由 `experiments/summarize.py` 自动汇总；环境：Python 3.9 + numpy（真实嵌入/LLM 为可选增强）。", ""]

    if e1:
        L += ["## 实验 1 · AgentLoop vs 单次 RAG（多跳覆盖率）", ""]
        for tag, r in e1.get("by_embedder", {}).items():
            L += [f"### 检索器：{r['embedder']}（dim={r['dim']}）", "",
                  f"- 问题数：{r['n_questions']}；平均覆盖率：单次 RAG "
                  f"**{r['avg_one_shot_coverage']:.3f}** → AgentLoop "
                  f"**{r['avg_agent_loop_coverage']:.3f}**（{r['improvement']:+.3f}）", "",
                  "| 问题 | 单次RAG 覆盖率 | AgentLoop 覆盖率 | Loop 轮数 |",
                  "|---|---|---|---|"]
            for row in r["rows"]:
                L.append(f"| {row['query'][:28]} | {row['one_shot_coverage']:.2f} | "
                         f"{row['agent_loop_coverage']:.2f} | {row['agent_loop_turns']} |")
            L.append("")

    if e2:
        L += ["## 实验 2 · 三级上下文压缩", "",
              "| 场景 | 轮数 | 压缩前 tokens | 压缩后 tokens | 比率 | L1落盘 | L2占位 | L3 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in e2["rows"]:
            L.append(f"| {r['scenario']} | {r['n_turns']} | {r['before_tokens']} | "
                     f"{r['after_tokens']} | {r['ratio']:.3f} | {r['l1_offloaded']} | "
                     f"{r['l2_placeholder_turns']} | {r['l3_triggered']} |")
        L.append("")

    if e3:
        rep = e3["report"]
        L += ["## 实验 3 · 记忆生命周期管理", "",
              f"- 重要性提升阈值：{e3['promote_threshold']}；语义合并阈值：{e3['merge_threshold']}",
              f"- 工作→情景：{len(rep['working_to_episodic'])} 条",
              f"- 情景→语义：{len(rep['episodic_to_semantic'])} 条",
              f"- 语义合并：{len(rep['semantic_merged'])} 对 → {rep['semantic_merged']}",
              f"- 整理后统计：{rep['stats_after']}", "",
              "时间衰减（半衰期 72h）：", "",
              "| 距今(h) | 0 | 12 | 24 | 48 | 72 | 144 | 336 |",
              "|---|---|---|---|---|---|---|---|",
              "| 衰减系数 | " + " | ".join(f"{c['decay']:.3f}" for c in e3["decay_curve"]) + " |", ""]

    if e4:
        L += ["## 实验 4 · 数据蒸馏 + Embedding 三元损失微调", "",
              f"- 标注 query 数：{e4['n_queries']}",
              f"- LLM Judge 与人类标注一致率：**{e4['judge_agreement']:.4f}**",
              f"- 人类对齐后标签准确率：**{e4['aligned_accuracy']:.4f}**"
              f"（修正 {e4['n_fixed']}/{e4['n_bad_cases']} 个 bad case）",
              f"- 训练/测试：{e4['train_size']}/{e4['test_size']}；三元组 {e4['n_triplets']} 条"
              f"（hard negative）",
              f"- 训练：{e4['epochs']} epoch，loss {e4['loss_first']:.4f} → {e4['loss_last']:.4f}", "",
              "| 评测集 | Recall@1 | Recall@3 | Recall@5 |",
              "|---|---|---|---|"]
        b, a = e4["test_recall_before"], e4["test_recall_after"]
        L.append(f"| 10% 留出（微调前） | {b['1']:.4f} | {b['3']:.4f} | {b['5']:.4f} |")
        L.append(f"| 10% 留出（微调后） | {a['1']:.4f} | {a['3']:.4f} | {a['5']:.4f} |")
        if "cv5_recall_before" in e4:
            cb, ca = e4["cv5_recall_before"], e4["cv5_recall_after"]
            L.append(f"| 5 折CV（微调前） | {cb['1']:.4f} | {cb['3']:.4f} | {cb['5']:.4f} |")
            L.append(f"| 5 折CV（微调后） | {ca['1']:.4f} | {ca['3']:.4f} | {ca['5']:.4f} |")
        ab, aa = e4["all_recall_before"], e4["all_recall_after"]
        L.append(f"| 全量集（微调前） | {ab['1']:.4f} | {ab['3']:.4f} | {ab['5']:.4f} |")
        L.append(f"| 全量集（微调后） | {aa['1']:.4f} | {aa['3']:.4f} | {aa['5']:.4f} |")
        L.append("")

    if e5:
        L += ["## 实验 5 · 真实嵌入 vs 离线哈希", "",
              f"- 标注集：{e5['n_queries']} 条 query / {e5['n_chunks']} 个 chunk", "",
              "| 后端 | 维度 | 编码耗时 | Recall@1 | Recall@3 | Recall@5 (dense) |",
              "|---|---|---|---|---|---|"]
        for name, r in e5["results"].items():
            d = r["dense"]
            L.append(f"| {name} | {r['dim']} | {r['encode_sec']:.2f}s | {d['1']:.4f} | "
                     f"{d['3']:.4f} | {d['5']:.4f} |")
        L.append("")
        L += ["| 后端 | Recall@1 | Recall@3 | Recall@5 (MQE) |", "|---|---|---|---|"]
        for name, r in e5["results"].items():
            m = r["mqe"]
            L.append(f"| {name} | {m['1']:.4f} | {m['3']:.4f} | {m['5']:.4f} |")
        L.append("")
        if "dim_tolerance" in e5:
            dt = e5["dim_tolerance"]
            L += [f"维度容错（{dt['original_dim']} 维截断到 {dt['truncated_dim']} 维再归一化）：", "",
                  "| Recall@1 | Recall@3 | Recall@5 |", "|---|---|---|",
                  f"| {dt['recall_512']['1']:.4f} → {dt['recall_384']['1']:.4f} | "
                  f"{dt['recall_512']['3']:.4f} → {dt['recall_384']['3']:.4f} | "
                  f"{dt['recall_512']['5']:.4f} → {dt['recall_384']['5']:.4f} |", ""]

    out = config.RESULTS_DIR / "RESULTS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[写出] {out}")


if __name__ == "__main__":
    main()
