from __future__ import annotations

from .core import ToolPlugin

_preinit_tools: list[ToolPlugin] = []


def register_tool(plugin: ToolPlugin) -> None:
    """Queue a tool for registration when the ToolManager is created.

    Users can call this in their own code before starting the visualizer.
    """
    _preinit_tools.append(plugin)


def consume_preinit_tools() -> list[ToolPlugin]:
    """Return and clear all pre-registered tools."""
    out = list(_preinit_tools)
    _preinit_tools.clear()
    return out