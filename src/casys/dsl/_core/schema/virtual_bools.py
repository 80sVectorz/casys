from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Type

import numpy as np
from casys.dsl._core.schema.base_components import FieldSchema
from casys.dsl._core.schema.schema_base import DirtySchema, Schema, SchemaPostProcessor
from casys.dsl._core.schema.soa_layout import SoaField

if TYPE_CHECKING:
    from casys.dsl._core.schema.soa_layout import SoaField
    from casys.dsl._core.schema.schema_base import SchemaPostProcessor

class VirtualBoolField(DirtySchema, FieldSchema):
    """
    A single boolean field that will be packed with other virtual booleans into a bit-plane.

    properties (set by post_processor):
        bit_plane_soa_field (SoaField): The SoA field of the final bit-plane that holds this bool.
        bit_idx (int): The index of this bool in the final bit-plane, starting at 0.
    """

    bit_plane_soa_field: SoaField | None = None
    bit_idx: int = -1

    def __post_init__(self):
        self.set_dirty()

    def get_post_processor(self):
        return VirtualBoolPacker

    def _resolve_field(self) -> SoaField:
        if self.is_dirty:
            raise Exception('Tried to resolve field for VirtualBoolField dirty schema object.')

        assert self.bit_plane_soa_field is not None

        return self.bit_plane_soa_field

class VirtualBoolPacker(SchemaPostProcessor[VirtualBoolField]):
    mappings: list[tuple[VirtualBoolField,int]]
    plane_idx: int = 0
    last_bit_idx: int = 0

    def __init__(self) -> None:
        self.mappings = []

    def process(self, target: VirtualBoolField):
        last_bit_idx = self.last_bit_idx + 1

        if last_bit_idx + 1 >= 64:
            last_bit_idx = 0
            self.plane_idx += 1 

        self.mappings.append((target, self.plane_idx))
        target.bit_idx = last_bit_idx

        self.last_bit_idx = last_bit_idx

    def finalize(self):
        n_planes = self.plane_idx + 1

        dtypes = [np.uint64] * (n_planes-1)
        default_values = [0b0 << 64] * (n_planes-1)

        dtypes.append(
            np.min_scalar_type(0b1 << self.last_bit_idx).type
        )
        default_values.append(0b0 << self.last_bit_idx)

        for s,p in self.mappings:
            if s.default_value:
                default_values[p] |= 1 << s.bit_idx

        soa_fields = [
            SoaField(f'_vbool_plane_{i+1}',data_type = dtypes[i], default_value = default_values[i])
            for i in range(n_planes)
        ]

        for s,p in self.mappings:
            s.bit_plane_soa_field = soa_fields[p]