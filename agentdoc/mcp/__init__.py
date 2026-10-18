"""MCP（Model Context Protocol）最小实现。

* `server` —— 把 agentdoc 的检索/记忆能力暴露为 MCP 工具（stdio 传输）
* `client` —— 最小 MCP 客户端 + 工具适配器（可直接注册进 ToolRegistry）

协议要点（JSON-RPC 2.0，按行分隔）：
    initialize -> notifications/initialized -> tools/list -> tools/call
"""
from __future__ import annotations

from .client import MCPClient, MCPToolAdapter, result_text
from .server import PROTOCOL_VERSION, MCPServer

__all__ = ["MCPClient", "MCPToolAdapter", "MCPServer", "PROTOCOL_VERSION",
           "result_text"]
