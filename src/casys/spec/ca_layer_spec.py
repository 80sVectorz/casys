from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from casys.spec.cac_type import CaCellTypeSpec, CactField
    from casys.spec.cact_field_spec import CactFieldSpec

from dataclasses import dataclass
from casys._utils.debug_utils import PrettyReprMixin

@dataclass(frozen=True)
class CaLayerRef:
    name: str
    cact: CaCellTypeSpec

    @property
    def fields(self) -> dict[str, CactField]:
        return self.cact.fields

    @staticmethod
    def get_soa_pairs_multi(
        layers: Iterable[CaLayerSpec | CaLayerRef], 
        filter_predicate: Callable[[str,str],bool] = lambda a,b: True, 
        include_field_objects: bool = False
    ) -> tuple[tuple[str,str],...] | tuple[tuple[str,str,CactField],...]:
        """Get all SoA field pairs"""
        res = tuple(
            (layer.name,fld,layer.cact.fields[fld]) if include_field_objects else (layer.name,fld)
            for layer in layers
            for fld in layer.cact.fields
            if filter_predicate(layer.name, fld)
        )
        return res # type: ignore
    
    def soa_pairs(self) -> tuple[tuple[str,str],...]:
        return self.get_soa_pairs_multi([self]) # type: ignore

@dataclass(frozen=True)
class CaLayerSpec(PrettyReprMixin):
    name: str
    cact: CaCellTypeSpec
    fields: dict[str, CactFieldSpec]

    @property
    def virtual_fields(self) -> dict[str, CactField]:
        return {
            field_name:field 
            for field_name,field in self.cact.fields.items()
            if field.field_type.is_virtual
        }

    @classmethod
    def from_cact(cls,name,cact: CaCellTypeSpec) -> CaLayerSpec:
        fields = {
            fld_name:CactFieldSpec.from_field(fld)
            for fld_name, fld in cact.fields.items()
        }

        new_layer = cls(name,cact,fields)
        for field in fields.values():
            field.parent_layer = new_layer

        return new_layer

    def get_ref(self) -> CaLayerRef:
        return CaLayerRef(self.name, self.cact)