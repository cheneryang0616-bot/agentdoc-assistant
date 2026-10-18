"""通用 MCP stdio 服务端基类（JSON-RPC 2.0，按行分隔）。

子类只需提供：
    server_info: dict
    tools: list[dict]            # MCP 工具定义（inputSchema）
    call_handler(name, args) -> dict   # 返回 {"content":[...], "isError":bool}
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List

PROTOCOL_VERSION = "2025-06-18"


class StdioServer:
    def __init__(self, server_info: Dict[str, Any], tools: List[Dict[str, Any]],
                 call_handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
                 verbose: bool = False) -> None:
        self.server_info = server_info
        self.tools = tools
        self.call_handler = call_handler
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def log(self, *args) -> None:
        if self.verbose:
            print("[mcp-server]", *args, file=sys.stderr, flush=True)

    def send(self, obj: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def reply(self, rid: Any, result: Dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def reply_error(self, rid: Any, code: int, message: str) -> None:
        self.send({"jsonrpc": "2.0", "id": rid,
                   "error": {"code": code, "message": message}})

    # ------------------------------------------------------------------ #
    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.reply_error(None, -32700, "Parse error")
                continue
            try:
                self.handle(msg)
            except Exception as e:  # 协议层异常不中断服务
                if "id" in msg:
                    self.reply_error(msg["id"], -32603, f"{type(e).__name__}: {e}")

    def handle(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        rid = msg.get("id")

        if rid is None:  # 通知
            if method == "notifications/initialized":
                self.log("client initialized")
            return

        if method == "initialize":
            params = msg.get("params") or {}
            self.reply(rid, {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": self.server_info,
            })
            return

        if method == "ping":
            self.reply(rid, {})
            return

        if method == "tools/list":
            self.reply(rid, {"tools": self.tools})
            return

        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            self.log("tools/call", name, args)
            self.reply(rid, self.call_handler(name, args))
            return

        self.reply_error(rid, -32601, f"Method not found: {method}")
