from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaKernel

from dataclasses import dataclass
from functools import reduce

@dataclass
class LayerUsageInfo:
    """
    Concrete access info for layers and layer field pairs

    Attributes:
      Accesses: What layer field pairs might be accessed
      reads: What layer field pairs might be read from
      writes: What layer field pairs might be written to
      guaranteed_writes: What layer field pairs will always be overwritten
      local_only_reads: What layer field pairs are only read from at the kernel position or not read at all.
    """
    index_lut: dict[tuple[str,str] | str, int]

    accesses: list[int] | tuple[int,...]
    reads: list[int] | tuple[int,...]
    writes: list[int] | tuple[int,...]
    guaranteed_writes: list[int] | tuple[int,...]
    local_only_reads: list[int] | tuple[int,...]

    def check_bitmask_layer(self,bitmasks: Sequence[int], layer: str) -> bool:
        if layer not in self.index_lut: return False
        l = self.index_lut[layer]
        return bool(bitmasks[l])

    def check_bitmask_field(self,bitmasks: Sequence[int], layer: str, field: str) -> bool:
        lut = self.index_lut
        if layer in lut and (layer,field) in lut:
            l,fld = lut[layer], 0b1 << lut[(layer,field)]
            return bool(bitmasks[l] & fld)
        return False

    def check_accesses(self,layer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.accesses, layer,field)
        return self.check_bitmask_layer(self.accesses, layer)
    
    def check_reads(self,layer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.reads, layer,field)
        return self.check_bitmask_layer(self.reads, layer)

    def check_writes(self,layer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.writes, layer,field)
        return self.check_bitmask_layer(self.writes, layer)

    def check_write_guaranteed(self,layer: str, field: str) -> bool:
        return self.check_bitmask_field(self.guaranteed_writes, layer,field)

    def check_read_local_only(self,buffer: str, field: str) -> bool:
        return (
            self.check_bitmask_field(self.local_only_reads, buffer,field)
            or not self.check_bitmask_field(self.reads, buffer,field)
        )

    @property
    def buffers(self) -> list[str]:
        return [k for k in self.index_lut if isinstance(k,str)]
    
    @classmethod
    def merge(cls, merge_targets: Sequence[LayerUsageInfo]) -> LayerUsageInfo:
        unique_layers = list(set(
            k
            for target in merge_targets
            for k in target.index_lut.keys()
            if isinstance(k, str)
        ))
        unique_fields = set(
            (k,v)
            for target in merge_targets
            for k,v in target.index_lut.items()
            if isinstance(k, tuple)
        )

        index_lut = {
            **{l:i for i,l in enumerate(unique_layers)},
            **{fld:v for fld,v in unique_fields},
        }

        accesses = [
            reduce(int.__or__, (
                target.accesses[target.index_lut[layer]]
                for target in merge_targets
                if layer in target.index_lut
            )) 
            for layer in unique_layers
        ]

        reads = [
            reduce(int.__or__, (
                target.reads[target.index_lut[layer]]
                for target in merge_targets
                if layer in target.index_lut
            )) 
            for layer in unique_layers
        ]

        writes = [
            reduce(int.__or__, (
                target.writes[target.index_lut[layer]]
                for target in merge_targets
                if layer in target.index_lut
            )) 
            for layer in unique_layers
        ]

        guaranteed_writes = [
            reduce(int.__or__, (
                target.guaranteed_writes[target.index_lut[layer]]
                for target in merge_targets
                if layer in target.index_lut
            )) 
            for layer in unique_layers
        ]

        local_only_reads = [
            reduce(int.__and__, (
                target.local_only_reads[target.index_lut[layer]] | ~target.reads[target.index_lut[layer]]
                for target in merge_targets
                if layer in target.index_lut
            )) 
            for layer in unique_layers
        ]

        return LayerUsageInfo(
            index_lut,
            accesses,
            reads,
            writes,
            guaranteed_writes,
            local_only_reads,
        )


class UnfinishedLayerUsageInfo(LayerUsageInfo):

    def __init__(self, ir: Ir_CaKernel) -> None:
        self.index_lut = {}

        for i, (layer_name, layer) in enumerate(ir.base.layer_args.items()):
            self.index_lut[layer_name] = i

            for j, fld in enumerate(layer.fields):
                self.index_lut[(layer_name,fld)] = j

        layers = ir.base.layer_args 

        initial_bitmasks = [0b0 << len(l.fields) for l in layers.values()]

        self.accesses = initial_bitmasks.copy()
        self.reads = initial_bitmasks.copy()
        self.writes = initial_bitmasks.copy()
        self.guaranteed_writes = initial_bitmasks.copy()
        self.local_only_reads = initial_bitmasks.copy()
    
    def _add_accesses(self, buffer: int, field: int):
        self.accesses[buffer] |= field # type: ignore

    def add_read(self, layer,field, is_local=False):
        lut = self.index_lut
        l,fld = lut[layer], 0b1 << lut[(layer,field)]

        self.reads[l] |= fld # type: ignore

        if bool(self.local_only_reads[l] & fld) and not is_local:
            self.local_only_reads[l] ^= fld # type: ignore

        if not bool(self.reads[l] & fld) and is_local:
            self.local_only_reads[l] |= fld # type: ignore

        self._add_accesses(l,fld)

    def add_write(self, layer,field, guaranteed: bool = False):
        lut = self.index_lut
        l,fld = lut[layer], 0b1 << lut[(layer,field)]
        self.writes[l] |= fld # type: ignore
        if guaranteed:
            self.guaranteed_writes[l] |= fld # type: ignore
        self._add_accesses(l,fld)

    def finalized(self) -> LayerUsageInfo:
        return LayerUsageInfo(
            self.index_lut,
            tuple(self.accesses),
            tuple(self.reads),
            tuple(self.writes),
            tuple(self.guaranteed_writes),
            tuple(self.local_only_reads)
        )
    
    