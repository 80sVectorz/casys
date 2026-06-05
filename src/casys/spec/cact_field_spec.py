from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casys.spec.cac_type import CactField, CactFieldType
    from casys.spec.ca_layer_spec import CaLayerSpec

@dataclass(slots=True)
class CactFieldSpec:
    name: str
    field_type: CactFieldType
    parent_layer: CaLayerSpec | None = None
    virtual_mapping: str | None = None

    @classmethod
    def from_field(cls, field: CactField) -> CactFieldSpec:
        return cls(field.name, field.field_type)

    def resolved_key(self) -> str:
        parent_name = self.parent_layer.name # type: ignore
        if self.field_type.is_virtual and self.virtual_mapping:
            return self.virtual_mapping
        return f'{parent_name}_{self.name}'