from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from casys.core import FieldSchema

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

    def register_inspect_processor(
        self,
        field_key: str | object,
        processor: Callable[[Any], str],
    ) -> None:
        """Register a custom value stringifier for an inspect tool field."""
        if isinstance(field_key, str):
            self.inspect_processors[field_key] = processor
        elif isinstance(field_key, FieldSchema):
            self.inspect_processors[field_key.canonical_name()] = processor
        else:
            raise ValueError(f"Failed to interpret field_key '{field_key}'")

    def register_inspect_processors(
        self,
        field_buffer_keys: list[str | object],
        processor: Callable[[Any], str],
    ) -> None:
        """Register a custom value stringifier for multiple inspect tool fields."""
        for k in field_buffer_keys:
            self.register_inspect_processor(k, processor)