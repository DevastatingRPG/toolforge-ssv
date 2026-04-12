from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

try:
    from ...models import Tool
except ImportError:
    from models import Tool


class AbstractToolStore(ABC):
    """Abstract interface for tool catalogs used by the environment."""

    @abstractmethod
    def get_atomic_tools(self) -> list[Tool]:
        """Return all predefined non-macro tools."""

    @abstractmethod
    def get_all_tools(self) -> list[Tool]:
        """Return every currently registered tool."""

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Tool]:
        """Return a single tool by name, if present."""

    @abstractmethod
    def add_tool(self, tool: Tool) -> None:
        """Register a new tool in the store."""

    @abstractmethod
    def add_tools(self, tools: Iterable[Tool]) -> None:
        """Register multiple tools in the store."""
