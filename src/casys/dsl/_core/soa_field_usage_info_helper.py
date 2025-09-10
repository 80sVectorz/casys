from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_SOA_LAYOUT

from dataclasses import dataclass
from functools import reduce

@dataclass
class SoaFieldUsageInfo:
    """
    Concrete access info for SoA fields

    Attributes:
      Accesses: What fields might be accessed
      reads: What fields might be read from
      writes: What fields might be written to
      guaranteed_writes: What fields will always be overwritten
      local_only_reads: What fields are only read from at the kernel position or not read at all.
    """
    index_lut: dict[str, int]

    accesses: int
    reads: int
    writes: int
    guaranteed_writes: int
    local_only_reads: int

    def check_bitmask(self,bitmask: int, field: str) -> bool:
        fld = 0b1 << self.index_lut[field]
        return bitmask & fld != 0

    def check_accesses(self,field: str) -> bool:
        return self.check_bitmask(self.accesses, field)
    
    def check_reads(self, field: str) -> bool:
        return self.check_bitmask(self.reads, field)

    def check_writes(self, field: str) -> bool:
        return self.check_bitmask(self.writes, field)

    def check_write_guaranteed(self, field: str) -> bool:
        return self.check_bitmask(self.guaranteed_writes, field)

    def check_read_local_only(self, field: str) -> bool:
        return self.check_bitmask(self.local_only_reads, field)

    @property
    def buffers(self) -> list[str]:
        return [k for k in self.index_lut if isinstance(k,str)]
    
    @classmethod
    def merge(cls, merge_targets: Sequence[SoaFieldUsageInfo]) -> SoaFieldUsageInfo | None:
        if len(merge_targets) == 0: return None

        accesses = reduce(int.__or__, (
            target.accesses
            for target in merge_targets
        ))

        reads = reduce(int.__or__, (
            target.reads
            for target in merge_targets
        )) 

        writes = reduce(int.__or__, (
            target.writes
            for target in merge_targets
        )) 

        guaranteed_writes = reduce(int.__or__, (
            target.guaranteed_writes
            for target in merge_targets
        )) 

        local_only_reads = reduce(int.__and__, (
            target.local_only_reads | ~target.reads
            for target in merge_targets
        )) 

        return SoaFieldUsageInfo(
            merge_targets[0].index_lut, # index_lut is always the same
            accesses,
            reads,
            writes,
            guaranteed_writes,
            local_only_reads,
        )


class UnfinishedSoaFieldUsageInfo(SoaFieldUsageInfo):

    def __init__(self, ir: Ir_CaSys) -> None:
        soa_layout = ir.metadata.get(MDK_SOA_LAYOUT)

        field_names = list(soa_layout.fields.keys())
        
        self.index_lut = dict(zip(field_names,range(0,len(field_names))))

        initial_bitmask = 0b0 << (len(field_names)-1)

        self.accesses = initial_bitmask
        self.reads = initial_bitmask
        self.writes = initial_bitmask
        self.guaranteed_writes = initial_bitmask
        self.local_only_reads = initial_bitmask
    
    def _add_accesses(self, field: int):
        self.accesses |= field # type: ignore

    def add_read(self, field, is_local=False):
        lut = self.index_lut
        fld = 0b1 << lut[field]

        self.reads |= fld # type: ignore

        if bool(self.local_only_reads & fld) and not is_local:
            self.local_only_reads ^= fld # type: ignore

        if not bool(self.reads & fld) and is_local:
            self.local_only_reads |= fld # type: ignore

        self._add_accesses(fld)

    def add_write(self, field: str, guaranteed: bool = False):
        lut = self.index_lut
        fld = 0b1 << lut[field]
        self.writes |= fld # type: ignore
        if guaranteed:
            self.guaranteed_writes |= fld # type: ignore
        self._add_accesses(fld)

    def finalized(self) -> SoaFieldUsageInfo:
        return SoaFieldUsageInfo(
            self.index_lut,
            self.accesses,
            self.reads,
            self.writes,
            self.guaranteed_writes,
            self.local_only_reads
        )
    
    