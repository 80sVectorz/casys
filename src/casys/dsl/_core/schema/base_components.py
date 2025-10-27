from __future__ import annotations

from functools import cache
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    import numpy as np

from casys.dsl._core.schema.schema_base import DirtySchema, Schema
from casys.dsl._core.schema.soa_layout import SoaField

@dataclass
class FieldSchema(Schema):
    """
    A single field.

    Args:
        name (str): The field name
        data_type (Type[np.generic]): The true numpy datatype
        default_value: A optional default value
    """

    def __post_init__(self):
        pass

    name: str
    data_type: Type[np.generic]
    default_value: Any = None

    def resolve_fields(self) -> dict[str,SoaField]:
        soa_field = self.resolve_field()
        return {soa_field.name: soa_field}
    
    def resolve_field(self) -> SoaField:
        if not hasattr(self,'_soa_field'):
            self._soa_field = self._resolve_field()

        return self._soa_field

    def _resolve_field(self) -> SoaField:
        soa_field = SoaField(
            self.canonical_name(), 
            self.data_type, 
            self.default_value
        )
        return soa_field

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"
    
@dataclass
class GroupSchema(Schema):
    """A grouping of Schema objects."""

    name: str
    fields: dict[str,Schema]

    def __post_init__(self) -> None:
        for field in self.fields.values():
            field.parent = self
            if isinstance(field, DirtySchema) and field.is_dirty:
                self.set_has_dirty_offspring()

    def resolve_fields(self) -> dict[str,SoaField]:
        soa_fields = {}
        for field in self.fields.values():
            field.insert_resolved_fields(soa_fields)

        return soa_fields
    
    def get_children(self) -> list[Schema]:
        return list(self.fields.values())