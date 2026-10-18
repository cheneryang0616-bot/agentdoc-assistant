"""实验 2：三级上下文压缩（L1 落盘 / L2 占位符 / L3 全量压缩）。

三个场景分别"逼出"一级压缩，展示级联（成本递增）行为。

运行：  python3 experiments/exp2_context_compression.py
产出：  results/exp2_context_compression.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.context import ContextManager


def synth_history(n_turns: int, result_chars: int, qa_len: int) -> list:
    msgs = []
    for i in range(n_turns):
        q = f"第 {i+1} 个问题：" + "请检索并总结相关文档。" * (qa_len // 12 + 1)
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"我需要检索。第 {i+1} 轮。"},
            {"type": "tool_use", "id": f"t{i}", "name": "rag_search",
             "input": {"query": f"问题 {i+1}"}},
        ]})
        big = json.dumps({"query": f"问题 {i+1}", "hits": [
            {"doc_id": f"doc{j}", "chunk_id": f"doc{j}#1", "score": 0.5,
             "text": "检索到的正文内容。" * (max(1, result_chars) // 9)} for j in range(3)]},
            ensure_ascii=False)
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": big}]})
        msgs.append({"role": "assistant", "content": "第 "
                     f"{i+1} 轮的回答：" + "根据检索结果给出总结。" * (qa_len // 12 + 1)})
    return msgs


SCENARIOS = [
    {"name": "L1 落盘", "offload_chars": 400, "n_turns": 6, "result_chars": 1600, "qa_len": 40},
    {"name": "L2 占位", "offload_chars": 10 ** 9, "n_turns": 20, "result_chars": 260, "qa_len": 120},
    {"name": "L3 全量", "offload_chars": 10 ** 9, "n_turns": 4, "result_chars": 12000, "qa_len": 40},
]


def main() -> None:
    config.ensure_dirs()
    rows = []
    print(f"预算 = {config.CONTEXT_TOKEN_BUDGET} tokens；保留最近 "
          f"{config.KEEP_RECENT_TURNS} 轮\n")
    print(f"{'场景':<8}{'轮数':>4}{'压缩前':>9}{'压缩后':>9}{'比率':>8}"
          f"{'L1落盘':>8}{'L2占位':>8}{'L3':>7}")
    for sc in SCENARIOS:
        msgs = synth_history(sc["n_turns"], sc["result_chars"], sc["qa_len"])
        cm = ContextManager(budget=config.CONTEXT_TOKEN_BUDGET,
                            offload_chars=sc["offload_chars"])
        _, rep = cm.compress(msgs)
        d = rep.as_dict()
        rows.append({"scenario": sc["name"], **sc, **d})
        print(f"{sc['name']:<8}{sc['n_turns']:>4}{d['before_tokens']:>9}{d['after_tokens']:>9}"
              f"{d['ratio']:>8.3f}{d['l1_offloaded']:>8}{d['l2_placeholder_turns']:>8}"
              f"{str(d['l3_triggered']):>7}")

    out = config.RESULTS_DIR / "exp2_context_compression.json"
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out}")
    files = sorted(config.OFFLOAD_DIR.glob("toolres_*.txt"))
    print(f"[L1 落盘文件] {len(files)} 个，例如 {files[0].name if files else '无'}")


if __name__ == "__main__":
    main()
