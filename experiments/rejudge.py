"""对 exp6 已保存的回答重新判定（不重新调用模型）。

用途：调整判定标准时，用同一批真实回答重算指标，保证可比。
    python3 experiments/rejudge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from experiments.exp6_error_analysis import CASES, judge

BY_ID = {c["id"]: c for c in CASES}


def main() -> None:
    path = config.RESULTS_DIR / "exp6_error_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{'config':<16}{'before':>8}{'after':>8}")
    for name, cfg in data.items():
        before = cfg["pass"]
        for row in cfg["rows"]:
            v = judge(BY_ID[row["id"]], {"answer": row["answer"],
                                        "retrieved": row["retrieved"]})
            row["pass"], row["reason"], row["kind"] = v["pass"], v["reason"], v.get("kind", "")
        cfg["pass"] = sum(r["pass"] for r in cfg["rows"])
        print(f"{name:<16}{before:>8}{cfg['pass']:>8}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {path}")


if __name__ == "__main__":
    main()
