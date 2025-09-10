from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from casys.spec.cac_type import CaCellTypeSpec, CactField

from dataclasses import dataclass
from casys._utils.debug_utils import PrettyReprMixin


@dataclass(frozen=True, slots=True)
class CactBufferDescriptor(PrettyReprMixin):
    name: str
    cact: CaCellTypeSpec

    @staticmethod
    def get_soa_pairs_multi(
        descriptors: Iterable[CactBufferDescriptor], 
        filter_predicate: Callable[[str,str],bool] = lambda a,b: True, 
        include_field_objects: bool = False
    ) -> tuple[tuple[str,str],...] | tuple[tuple[str,str,CactField],...]:
        """Get all SoA field pairs"""
        res = tuple(
            (desc.name,fld,desc.cact.fields[fld]) if include_field_objects else (desc.name,fld)
            for desc in descriptors
            for fld in desc.cact.fields
            if filter_predicate(desc.name, fld)
        )
        return res # type: ignore
    
    def soa_pairs(self) -> tuple[tuple[str,str],...]:
        return self.get_soa_pairs_multi([self]) # type: ignore
    

@dataclass(frozen=True)
class KernelCallDescriptor(PrettyReprMixin):
    kernel_name: str
    kwargs: dict[str,str]

    # def instantiate_access(self, ir: Ir_CaSys) -> SoaFieldUsageInfo:
    #     """
    #     Instantiate layer usage info for bound step-scope buffer

    #     Args:
    #       ir (Ir_CaSys): The CA system IR object.

    #     Returns:
    #       LayerUsageInfo
    #     """
    #     kernel = ir.kernels[self.kernel_name]
    #     m = kernel.metadata
    #     k_layer_usage: SoaFieldUsageInfo = m.get(MDK_SOA_FIELD_USAGE_INFO)

    #     index_lut = {}
    #     for k,v in k_layer_usage.index_lut.items():
    #         if isinstance(k,tuple):
    #             layer,fld = k
    #             index_lut[(self.kwargs[layer],fld)] = v
    #         else:
    #             index_lut[self.kwargs[k]] = v

    #     layer_usage_info = SoaFieldUsageInfo(
    #         index_lut,
    #         k_layer_usage.accesses,
    #         k_layer_usage.reads,
    #         k_layer_usage.writes,
    #         k_layer_usage.guaranteed_writes,
    #         k_layer_usage.local_only_reads,
    #     )

    #     return layer_usage_info
    
    def __hash__(self) -> int:
        return hash((self.kernel_name, tuple(self.kwargs)))