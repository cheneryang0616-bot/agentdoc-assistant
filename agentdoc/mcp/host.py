"""MCP Host：让 agentdoc 作为宿主，同时连接多个外部 MCP server。

与 `client.py` 的区别：
  * `MCPClient`  —— 连接**单个** server
  * `MCPHost`    —— 按配置管理**多个** server，做工具聚合与命名空间隔离
                    （`<server>__<tool>`），避免不同 server 的同名工具冲突。

配置格式（兼容 Claude Desktop 的 mcpServers 写法）：
    {
      "mcpServers": {
        "agentdoc": {"command": "python3", "args": ["-m","agentdoc.mcp.server"],
                     "cwd": ".", "prefix": true},
        "fs":       {"command": "python3", "args": ["examples/mcp_fs_server.py"],
                     "cwd": ".", "prefix": true}
      }
    }
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import MCPClient, result_text

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ServerSpec:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    cwd: str = "."
    prefix: bool = True
    env: Dict[str, str] = field(default_factory=dict)


class MCPHost:
    def __init__(self, config_path: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None,
                 root: Optional[Path] = None) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        if config is None:
            config_path = Path(config_path or self.root / "mcp_servers.json")
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            self.config_path = Path(config_path)
            # 相对 cwd 以配置文件所在目录为基准
            self.base = Path(config_path).parent
        else:
            self.config_path = None
            self.base = self.root
        self.specs = self._parse(config)
        self.clients: Dict[str, MCPClient] = {}
        self.server_tools: Dict[str, List[Dict[str, Any]]] = {}
        self._index: Dict[str, tuple] = {}     # qualified -> (server, tool)
        self.call_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _parse(self, config: Dict[str, Any]) -> List[ServerSpec]:
        specs = []
        for name, cfg in (config.get("mcpServers") or {}).items():
            specs.append(ServerSpec(
                name=name,
                command=cfg.get("command", "python3"),
                args=list(cfg.get("args") or []),
                cwd=cfg.get("cwd", "."),
                prefix=bool(cfg.get("prefix", True)),
                env=dict(cfg.get("env") or {}),
            ))
        return specs

    # ------------------------------------------------------------------ #
    def start(self, verbose: bool = False) -> "MCPHost":
        for spec in self.specs:
            cmd = [sys.executable if spec.command in ("python", "python3") else spec.command]
            cmd += spec.args
            cwd = (self.base / spec.cwd).resolve() if not Path(spec.cwd).is_absolute() \
                else Path(spec.cwd)
            try:
                client = MCPClient(cmd, cwd=cwd)
                info = client.initialize()
                tools = client.list_tools()
            except Exception as e:
                if verbose:
                    print(f"[host] 跳过 {spec.name}: {type(e).__name__}: {e}",
                          file=sys.stderr)
                continue
            self.clients[spec.name] = client
            self.server_tools[spec.name] = tools
            for t in tools:
                q = f"{spec.name}__{t['name']}" if spec.prefix else t["name"]
                if q in self._index:          # 同名冲突时加后缀
                    q = f"{spec.name}__{t['name']}"
                self._index[q] = (spec.name, t["name"])
            if verbose:
                print(f"[host] {spec.name} 已连接: {info.get('serverInfo')} "
                      f"tools={[t['name'] for t in tools]}", file=sys.stderr)
        return self

    # ------------------------------------------------------------------ #
    def tools(self) -> List[Dict[str, Any]]:
        """聚合后的工具列表（带 source 与 qualified_name）。"""
        out = []
        for q, (server, tool) in self._index.items():
            spec = next((s for s in self.specs if s.name == server), None)
            raw = next(t for t in self.server_tools[server] if t["name"] == tool)
            out.append({
                "qualified_name": q,
                "server": server,
                "name": tool,
                "description": raw.get("description", ""),
                "input_schema": raw.get("inputSchema", {}),
            })
        return sorted(out, key=lambda x: (x["server"], x["name"]))

    def call(self, qualified: str, arguments: Dict[str, Any]) -> str:
        if qualified not in self._index:
            raise KeyError(f"未知 MCP 工具: {qualified}")
        server, tool = self._index[qualified]
        self.call_log.append({"qualified": qualified, "arguments": arguments})
        res = self.clients[server].call_tool(tool, arguments)
        text = result_text(res)
        return text if text else json.dumps(res, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def register_into(self, registry, overrides: Optional[Dict[str, str]] = None) -> List[str]:
        """把聚合工具注册进 ToolRegistry；overrides 可把 `srv__tool` 映射成本地名。"""
        registered = []
        for t in self.tools():
            q = t["qualified_name"]
            name = (overrides or {}).get(q, q)
            registry.register(name, f"[MCP:{t['server']}] {t['description']}",
                              t["input_schema"], self._handler(q))
            registered.append(name)
        return registered

    def _handler(self, qualified: str):
        def handler(**kwargs):
            return self.call(qualified, kwargs)
        return handler

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        for c in self.clients.values():
            try:
                c.close()
            except Exception:
                pass
        self.clients.clear()

    def __enter__(self) -> "MCPHost":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
