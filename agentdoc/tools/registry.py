"""工具注册表。

对应方案「Tool 的定义，本质上就是让 LLM 知道他可以做什么，然后 LLM 生成相应的
调用脚本，然后在工程侧执行这个脚本」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., str] | None = None

    def schema(self) -> Dict[str, Any]:
        """导出给 LLM 的 schema（不含 handler）。"""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, name: str, description: str, input_schema: Dict[str, Any],
                 handler: Callable[..., str]) -> None:
        self._tools[name] = ToolDef(name, description, input_schema, handler)

    def definitions(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        t = self._tools.get(name)
        if t is None or t.handler is None:
            return f"Unknown tool: {name}"
        try:
            out = t.handler(**(arguments or {}))
        except Exception as e:  # 工具异常不应中断 AgentLoop
            out = f"[工具 {name} 执行失败] {type(e).__name__}: {e}"
        return str(out)

    def handler(self, name: str) -> Callable[..., str] | None:
        t = self._tools.get(name)
        return t.handler if t else None
