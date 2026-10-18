"""示例：一个独立的"文件系统 MCP server"，用来演示 agentdoc 作为 host 连接外部服务。

它不是 agentdoc 的一部分——可以单独运行，任何 MCP 客户端（Claude Desktop / Cursor /
agentdoc 的 MCPHost）都能连。

启动：
    python3 examples/mcp_fs_server.py --root ./data --verbose

暴露工具：
    list_dir(path)            列出目录
    read_text(path, max_chars) 读取文本文件
    search_files(pattern)      按 glob 搜索文件名
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentdoc.mcp.base import StdioServer  # noqa: E402

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_dir",
        "description": "列出某个目录下的文件与子目录（相对 root）。",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对路径，默认 ."}},
            "required": [],
        },
    },
    {
        "name": "read_text",
        "description": "读取 root 下的文本文件内容。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "description": "默认 4000"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "按 glob 模式搜索文件名，例如 **/*.md。",
        "inputSchema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


class FsServer(StdioServer):
    def __init__(self, root: Path, verbose: bool = False) -> None:
        super().__init__({"name": "fs-mcp", "version": "1.0.0"}, TOOLS,
                         self._call, verbose)
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise PermissionError(f"越界访问: {rel}")
        return p

    def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "list_dir":
                p = self._safe(args.get("path", "."))
                entries = sorted(
                    [{"name": c.name, "type": "dir" if c.is_dir() else "file"}
                     for c in p.iterdir()], key=lambda x: (x["type"], x["name"]))
                payload = {"path": str(p.relative_to(self.root)), "entries": entries}
            elif name == "read_text":
                p = self._safe(args["path"])
                text = p.read_text(encoding="utf-8", errors="ignore")
                payload = {"path": str(p.relative_to(self.root)),
                           "chars": len(text),
                           "text": text[: int(args.get("max_chars", 4000))]}
            elif name == "search_files":
                hits = [str(x.relative_to(self.root))
                        for x in self.root.rglob(args.get("pattern", "*")) if x.is_file()]
                payload = {"pattern": args.get("pattern"), "matches": sorted(hits)[:200]}
            else:
                return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                        "isError": True}
            return {"content": [{"type": "text",
                                 "text": json.dumps(payload, ensure_ascii=False)}],
                    "isError": False}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                    "isError": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="示例文件系统 MCP server")
    ap.add_argument("--root", default=str(ROOT / "data"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    server = FsServer(Path(args.root), verbose=args.verbose)
    server.log("fs-mcp ready, root =", server.root)
    server.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
