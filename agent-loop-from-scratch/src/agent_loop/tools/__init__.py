"""
tools/: the agent's tools and the registry that holds them.

THE REGISTRY
    A registry is a lookup table: tool NAME → Tool object. It answers the
    two questions the agent asks:
      * "What do I send the model?"     → registry.definitions()
      * "The model asked for 'X'. What's X?" → registry.get("X")

    Phase 4's loop does exactly this for every tool_use block: look up the
    name, run the tool, and send back the result.

ORDER MATTERS (a bit)
    The tool list is part of every request, near the very start. Keeping
    the order stable (we use insertion order) keeps requests byte-identical
    from one call to the next, which prompt caching needs (Phase 8).
"""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from agent_loop.config import DEFAULT_WORKSPACE_DIR
from agent_loop.tools.base import Tool, ToolError, ToolInputError, validate_input
from agent_loop.tools.calculator import make_calculator_tool
from agent_loop.tools.file_reader import make_read_file_tool
from agent_loop.tools.web_fetch import make_fetch_url_tool

# What `from agent_loop.tools import ...` offers.
__all__ = [
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolRegistry",
    "default_registry",
    "validate_input",
]


class ToolRegistry:
    """A name → Tool lookup table that refuses duplicate names."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                # Two tools with one name: the model couldn't tell them apart.
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        """The `tools` list for the request body: what the model sees."""
        return [tool.definition() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        """Find a tool by name, or None if the model asked for one we don't have."""
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def default_registry(workspace_dir: Path = DEFAULT_WORKSPACE_DIR) -> ToolRegistry:
    """The standard tool set: calculator, read_file (sandboxed) and fetch_url."""
    return ToolRegistry(
        [
            make_calculator_tool(),
            make_read_file_tool(workspace_dir),
            make_fetch_url_tool(),
        ]
    )
