"""集成测试：用本地 mock 服务验证 OpenAI 兼容后端的完整工具调用链路。

不需要真实 API Key：起一个本地 HTTP 服务模拟 /v1/chat/completions，
第一次返回 tool_call，第二次返回文本，验证：
  消息格式转换 / tool schema 转换 / tool_call 解析 / AgentLoop 闭环。

运行：  python3 tests/test_openai_backend.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MockHandler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        MockHandler.calls.append(body)
        msgs = body.get("messages", [])
        has_tool_result = any(m.get("role") == "tool" for m in msgs)

        if not has_tool_result:
            # 第一次：调用 rag_search
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc", "type": "function",
                "function": {"name": "rag_search",
                             "arguments": json.dumps({"query": "JVM 是什么"},
                                                     ensure_ascii=False)}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "JVM 是 Java 虚拟机，负责执行字节码。"}
            finish = "stop"

        data = json.dumps({"choices": [{"message": msg, "finish_reason": finish}]},
                          ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # 静音
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    os.environ["AGENTDOC_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["AGENTDOC_API_KEY"] = "test-key"
    os.environ["AGENTDOC_MODEL"] = "mock-model"
    os.environ["AGENTDOC_LLM_PROVIDER"] = "openai"

    # 先确认后端选择正确
    from agentdoc.llm import LLMClient
    from agentdoc.rag import build_embedder
    from agentdoc.session import AgentSession

    client = LLMClient()
    assert client.backend.name == "openai-compatible", client.backend.name
    assert client.model == "mock-model", client.model
    print(f"  ✓ 后端选择：{client.backend.name} / model={client.model}")

    # 走完整的 AgentSession（哈希嵌入，避免加载大模型）
    sess = AgentSession(embedder=build_embedder("hash"))
    res = sess.ask("JVM 是什么？", keep_history=False)

    assert res["turns"] >= 1, "应至少发生一次工具调用"
    print(f"  ✓ AgentLoop 完成，轮数={res['turns']}")
    assert any(t["tool"] == "rag_search" for t in res["trace"]), res["trace"]
    print("  ✓ 工具调用被正确解析为 rag_search")
    assert "JVM" in res["answer"]
    print(f"  ✓ 最终回答：{res['answer'][:60]}")

    # 校验请求格式：system / tools / tool 消息
    first = MockHandler.calls[0]
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][0]["function"]["name"] == "rag_search"
    print("  ✓ 请求含 system 消息与 OpenAI function 工具定义")

    second = MockHandler.calls[1]
    tools_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert tools_msgs and tools_msgs[0]["tool_call_id"] == "call_abc"
    print("  ✓ 工具结果以 role=tool + tool_call_id 回传")

    srv.shutdown()
    print("\nOpenAI 兼容后端集成测试：全部通过")


if __name__ == "__main__":
    main()
