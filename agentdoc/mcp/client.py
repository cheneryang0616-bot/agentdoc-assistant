"""最小 MCP 客户端（stdio）+ 工具适配器。

用法：
    client = MCPClient.spawn(embedder="hash")
    client.initialize()
    tools = client.list_tools()
    result = client.call_tool("rag_search", {"query": "JVM 是什么"})
    client.close()

    # 把 MCP 工具直接注册进 Agent 的工具表（业务代码零改动）
    adapter = MCPToolAdapter(client)
    adapter.register_into(session.registry)
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .server import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[2]


class MCPClient:
    def __init__(self, cmd: List[str], cwd: Optional[Path] = None,
                 init_timeout: float = 60.0) -> None:
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd or ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stderr: List[str] = []
        self._next_id = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.init_timeout = init_timeout

    # ------------------------------------------------------------------ #
    @classmethod
    def spawn(cls, embedder: str = "hash", verbose: bool = False,
              memory_dir: Optional[str] = None) -> "MCPClient":
        cmd = [sys.executable, "-m", "agentdoc.mcp.server", "--embedder", embedder]
        if verbose:
            cmd.append("--verbose")
        if memory_dir:
            cmd += ["--memory-dir", memory_dir]
        return cls(cmd, cwd=ROOT)

    # ------------------------------------------------------------------ #
    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip())
            if len(self._stderr) > 200:
                self._stderr.pop(0)

    def _send(self, obj: Dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        with self._lock:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    # ------------------------------------------------------------------ #
    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = 120.0) -> Dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        deadline = time.time() + timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError(f"MCP 请求超时: {method}\nstderr: "
                                   + "\n".join(self._stderr[-5:]))
            try:
                msg = self._q.get(timeout=remain)
            except queue.Empty:
                raise TimeoutError(f"MCP 请求超时: {method}")
            if msg.get("id") != rid:
                continue          # 其他响应/通知，忽略
            if "error" in msg:
                raise RuntimeError(f"MCP error {msg['error'].get('code')}: "
                                   f"{msg['error'].get('message')}")
            return msg.get("result") or {}

    # ------------------------------------------------------------------ #
    def initialize(self) -> Dict[str, Any]:
        res = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "agentdoc-client", "version": "1.0.0"},
        }, timeout=self.init_timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return res

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "MCPClient":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def result_text(result: Dict[str, Any]) -> str:
    """从 tools/call 结果里拼出纯文本。"""
    parts = []
    for c in result.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts)


class MCPToolAdapter:
    """把 MCP server 的工具映射为 agentdoc 的 ToolRegistry 工具。"""

    def __init__(self, client: MCPClient) -> None:
        self.client = client
        self.tools = client.list_tools()
        self.calls: List[Dict[str, Any]] = []

    def names(self) -> List[str]:
        return [t["name"] for t in self.tools]

    def definitions(self) -> List[Dict[str, Any]]:
        return [{"name": t["name"],
                 "description": t.get("description", ""),
                 "input_schema": t.get("inputSchema", {})} for t in self.tools]

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        self.calls.append({"name": name, "arguments": arguments})
        res = self.client.call_tool(name, arguments)
        text = result_text(res)
        return text if text else json.dumps(res, ensure_ascii=False)

    def _handler(self, tool_name: str):
        def handler(**kwargs):
            return self.call(tool_name, kwargs)
        return handler

    def register_into(self, registry, prefix: str = "",
                      overrides: Optional[Dict[str, str]] = None) -> List[str]:
        """注册到 ToolRegistry；`overrides` 可把 MCP 工具名映射为本地的同名工具。"""
        registered = []
        for t in self.tools:
            name = (overrides or {}).get(t["name"], prefix + t["name"])
            registry.register(name, t.get("description", ""),
                              t.get("inputSchema", {}), self._handler(t["name"]))
            registered.append(name)
        return registered
