from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Sequence

from numpy import isin

from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_BUFFER_USAGE_INFO, BufferUsageInfo

if TYPE_CHECKING:
    from casys._cac_type import CaCellType
    from casys.dsl._core.ir import Ir_CaSys

from dataclasses import dataclass
from casys._utils.debug_utils import PrettyReprMixin


@dataclass(frozen=True, slots=True)
class CactBufferDescriptor(PrettyReprMixin):
    name: str
    cact: CaCellType

    @staticmethod
    def get_soa_pairs_multi(descriptors: Iterable[CactBufferDescriptor], filter_predicate: Callable[[str,str],bool] = lambda a,b: True) -> tuple[tuple[str,str],...]:
        """Get all SoA field pairs"""
        return tuple(
            (desc.name,fld)
            for desc in descriptors
            for fld in desc.cact._fields
            if filter_predicate(desc.name, fld)
        )
    
    @property
    def soa_pairs(self):
        return self.get_soa_pairs_multi([self])
    
    def mapped(self,name: str) -> CactBufferDescriptor:
        """
        Get a new instance that has a different name.
        Useful for call-site mapping between buffer instance and function argument name.
        """
        return CactBufferDescriptor(name, self.cact)
    


@dataclass(frozen=True)
class KernelCallDescriptor(PrettyReprMixin):
    kernel_name: str
    kwargs: dict[str,str]

    def instantiate_access(self, ir: Ir_CaSys) -> BufferUsageInfo:
        """
        Instantiate buffer usage info for bound step-scope buffer

        Args:
          ir (Ir_CaSys): The CA system IR object.

        Returns:
          BufferUsageInfo
        """
        kernel = ir.kernels[self.kernel_name]
        m = kernel.metadata
        k_buf_usage: BufferUsageInfo = m.get(MDK_BUFFER_USAGE_INFO)

        index_lut = {}
        for k,v in k_buf_usage.index_lut.items():
            if isinstance(k,tuple):
                buf,fld = k
                index_lut[(self.kwargs[buf],fld)] = v
            else:
                index_lut[self.kwargs[k]] = v

        buffer_usage_info = BufferUsageInfo(
            index_lut,
            tuple(k_buf_usage.accesses),
            tuple(k_buf_usage.reads),
            tuple(k_buf_usage.writes),
            tuple(k_buf_usage.guaranteed_writes),
            tuple(k_buf_usage.local_only_reads),
        )

        return buffer_usage_info
    
    def __hash__(self) -> int:
        return hash((self.kernel_name, tuple(self.kwargs)))