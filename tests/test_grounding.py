"""答案级 groundedness 质检的单元测试（用假客户端，不需要网络）。

    python3 tests/test_grounding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc.grounding import GroundingChecker
from agentdoc.llm import Block, Response

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ✓ {name}")


class FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def create(self, **kw) -> Response:
        self.calls += 1
        return Response(content=[Block("text", text=self.text)], stop_reason="end_turn")


class FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = FakeMessages(text)


Q = "双亲委派模型怎么实现？"
CTX = ["JVM 由类加载子系统、运行时数据区、执行引擎组成。"]
ANSWER = "双亲委派模型通过 ClassLoader.loadClass 实现。"


def main() -> None:
    print("[GroundingChecker] 判定与拦截")

    # 1) 片段不足以回答 -> 替换为拒答
    c = FakeClient('{"answerable": false, "grounded": true, "reason": "片段无实现细节"}')
    out, v = GroundingChecker(c).enforce(Q, CTX, ANSWER)
    check("answerable=false 时拒答", "没有足够信息" in out)
    check("拒答里带质检理由", "片段无实现细节" in out)
    check("返回 verdict", v is not None and v.get("ok") is False)

    # 2) 答案含片段外内容 -> 拒答
    c = FakeClient('{"answerable": true, "grounded": false, "reason": "答案含通用知识"}')
    out, v = GroundingChecker(c).enforce(Q, CTX, ANSWER)
    check("grounded=false 时拒答", "没有足够信息" in out)

    # 3) 通过 -> 原样返回
    c = FakeClient('{"answerable": true, "grounded": true, "reason": "充分"}')
    out, v = GroundingChecker(c).enforce(Q, CTX, ANSWER)
    check("双 true 时保留原答案", out == ANSWER)

    # 4) JSON 在代码块里也能解析
    c = FakeClient('```json\n{"answerable": true, "grounded": true, "reason": "ok"}\n```')
    out, _ = GroundingChecker(c).enforce(Q, CTX, ANSWER)
    check("可解析 ```json 包裹", out == ANSWER)

    # 5) 返回无法解析 -> 跳过（不误杀）
    c = FakeClient("我觉得可以吧")
    out, v = GroundingChecker(c).enforce(Q, CTX, ANSWER)
    check("解析失败时跳过而非拒答", out == ANSWER)
    check("标记 skipped", v is not None and v.get("skipped") is True)

    # 6) 关闭开关 -> 不调用模型
    c = FakeClient("should not be called")
    out, v = GroundingChecker(c, enabled=False).enforce(Q, CTX, ANSWER)
    check("enabled=False 时原样返回", out == ANSWER)
    check("enabled=False 时不发请求", c.messages.calls == 0)

    # 7) 没有检索片段 -> 跳过
    c = FakeClient("should not be called")
    out, v = GroundingChecker(c).enforce(Q, [], ANSWER)
    check("无片段时跳过", out == ANSWER and c.messages.calls == 0)

    print(f"\nGrounding 测试：全部通过（{PASS} 项断言）")


if __name__ == "__main__":
    main()
