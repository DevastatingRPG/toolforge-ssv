from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from server.tools.base import AbstractToolStore
from models import Tool


class SeededInMemoryToolStore(AbstractToolStore):
    """In-memory store initialized from a predefined tool list."""

    def __init__(self, initial_tools: Iterable[Tool]):
        self._tools_by_name: dict[str, Tool] = {}
        self._seeded_names: set[str] = set()

        for tool in initial_tools:
            self.add_tool(tool)
            self._seeded_names.add(tool.name)

    def get_atomic_tools(self) -> list[Tool]:
        return [
            tool.model_copy(deep=True)
            for name, tool in self._tools_by_name.items()
            if name in self._seeded_names and not tool.is_macro
        ]

    def get_all_tools(self) -> list[Tool]:
        return [tool.model_copy(deep=True) for tool in self._tools_by_name.values()]

    def get_tool(self, name: str) -> Optional[Tool]:
        tool = self._tools_by_name.get(name)
        return tool.model_copy(deep=True) if tool is not None else None

    def add_tool(self, tool: Tool) -> None:
        if tool.name in self._tools_by_name:
            raise ValueError(f"Tool '{tool.name}' already exists.")
        self._tools_by_name[tool.name] = tool.model_copy(deep=True)

    def add_tools(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.add_tool(tool)
