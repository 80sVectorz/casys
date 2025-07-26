from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

@dataclass
class InfoField:
    label: str
    callback: Callable[[], str]

@dataclass
class Tool:
    name: str
    callback: Callable[[tuple[int, int], tuple[int, int]], None]
    single_use: bool = False
    registered: bool = False

class UIModel:
    """
    Carries user-configurable fields, tools, and inspect processors
    for wiring into the Qt-based GUI.
    """
    def __init__(self) -> None:
        self.info_fields: list[InfoField] = []
        self.tools: dict[str, Tool] = {}
        self.inspect_processors: dict[str, Callable[[Any], str]] = {}
        self.tool_active: bool = False

    def add_info_field(self, label: str, callback: Callable[[], str]) -> None:
        """Register a piece of text info to display and update."""
        self.info_fields.append(InfoField(label=label, callback=callback))

    def register_tool(
        self,
        name: str,
        callback: Callable[[tuple[int, int], tuple[int, int]], None],
        single_use: bool = False,
    ) -> None:
        """Register a mouse-based tool with click press/release handling."""
        def wrapped(pos_press: tuple[int, int], pos_release: tuple[int, int]) -> None:
            callback(pos_press, pos_release)
            self.tool_active = False

        self.tools[name] = Tool(name=name, callback=wrapped, single_use=single_use)

    def register_inspect_processor(
        self,
        field_buffer_key: str,
        processor: Callable[[Any], str],
    ) -> None:
        """Register a custom value stringifier for an inspect tool field."""
        self.inspect_processors[field_buffer_key] = processor

    def register_inspect_processors(
        self,
        field_buffer_keys: list[str],
        processor: Callable[[Any], str],
    ) -> None:
        """Register a custom value stringifier for multiple inspect tool fields."""
        for k in field_buffer_keys:
            self.inspect_processors[k] = processor