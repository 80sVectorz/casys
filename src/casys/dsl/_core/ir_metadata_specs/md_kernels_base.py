from __future__ import annotations
import ast
from dataclasses import dataclass, field
import enum
from operator import index
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaKernel
from casys.dsl._core.metadata_store import MetadataKey

from functools import reduce

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaKernel

MDK_SIGNATURE = MetadataKey[list[str]]('', 'signature', factory=list, doc='The final JITed kernel function signature')

MDK_READONLY = MetadataKey[set[str]]('', 'readonly', doc='A set of variables ids that have been marked readonly.')
MDK_POS_VARS = MetadataKey[dict[str,int]]('', 'pos_vars', factory=dict, doc='A dict that maps pos_var marked variables ids to their axes.')
MDK_ALIASES = MetadataKey[dict[str,ast.AST]]('', 'aliases', factory=dict, doc='A dict that maps variable ids to casys_ast nodes.')
MDK_NEEDS_DEDICATED_IDX = MetadataKey[set[str]]('', 'needs_dedicated_idx', factory=set, doc='A set of the buffers that require a dedicated double buffer index.')

@dataclass
class BufferUsageInfo:
    """
    Concrete access info for buffers and buffer field pairs

    Attributes:
      Accesses: What buffer field pairs might be accessed
      reads: What buffer field pairs might be read from
      writes: What buffer field pairs might be written to
      guaranteed_writes: What buffer field pairs will always be overwritten
      local_only_reads: What buffer field pairs are only read from at the kernel position or not read at all.
    """
    index_lut: dict[tuple[str,str] | str, int]

    accesses: list[int] | tuple[int,...]
    reads: list[int] | tuple[int,...]
    writes: list[int] | tuple[int,...]
    guaranteed_writes: list[int] | tuple[int,...]
    local_only_reads: list[int] | tuple[int,...]

    def check_bitmask_buffer(self,bitmasks: Sequence[int], buffer: str) -> bool:
        if buffer not in self.index_lut: return False
        buf = self.index_lut[buffer]
        return bool(bitmasks[buf])

    def check_bitmask_field(self,bitmasks: Sequence[int], buffer: str, field: str) -> bool:
        lut = self.index_lut
        if buffer in lut and (buffer,field) in lut:
            buf,fld = lut[buffer], 0b1 << lut[(buffer,field)]
            return bool(bitmasks[buf] & fld)
        return False

    def check_accesses(self,buffer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.accesses, buffer,field)
        return self.check_bitmask_buffer(self.accesses, buffer)
    
    def check_reads(self,buffer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.reads, buffer,field)
        return self.check_bitmask_buffer(self.reads, buffer)

    def check_writes(self,buffer: str, field: str | None = None) -> bool:
        if field:
            return self.check_bitmask_field(self.writes, buffer,field)
        return self.check_bitmask_buffer(self.writes, buffer)

    def check_write_guaranteed(self,buffer: str, field: str) -> bool:
        return self.check_bitmask_field(self.guaranteed_writes, buffer,field)

    def check_read_local_only(self,buffer: str, field: str) -> bool:
        return (
            self.check_bitmask_field(self.local_only_reads, buffer,field)
            or not self.check_bitmask_field(self.reads, buffer,field)
        )

    @property
    def buffers(self) -> list[str]:
        return [k for k in self.index_lut if isinstance(k,str)]
    
    @classmethod
    def merge(cls, buffer_usage_infos: Sequence[BufferUsageInfo]) -> BufferUsageInfo:
        unique_buffers = list(set(
            k
            for info in buffer_usage_infos
            for k in info.index_lut.keys()
            if isinstance(k, str)
        ))
        unique_fields = set(
            (k,v)
            for info in buffer_usage_infos
            for k,v in info.index_lut.items()
            if isinstance(k, tuple)
        )

        index_lut = {
            **{buf:i for i,buf in enumerate(unique_buffers)},
            **{k:v for k,v in unique_fields},
        }

        accesses = [
            reduce(int.__or__, (
                info.accesses[info.index_lut[buffer]]
                for info in buffer_usage_infos
                if buffer in info.index_lut
            )) 
            for buffer in unique_buffers
        ]

        reads = [
            reduce(int.__or__, (
                info.reads[info.index_lut[buffer]]
                for info in buffer_usage_infos
                if buffer in info.index_lut
            )) 
            for buffer in unique_buffers
        ]

        writes = [
            reduce(int.__or__, (
                info.writes[info.index_lut[buffer]]
                for info in buffer_usage_infos
                if buffer in info.index_lut
            )) 
            for buffer in unique_buffers
        ]

        guaranteed_writes = [
            reduce(int.__or__, (
                info.guaranteed_writes[info.index_lut[buffer]]
                for info in buffer_usage_infos
                if buffer in info.index_lut
            )) 
            for buffer in unique_buffers
        ]

        local_only_reads = [
            reduce(int.__and__, (
                info.local_only_reads[info.index_lut[buffer]] | ~info.reads[info.index_lut[buffer]]
                for info in buffer_usage_infos
                if buffer in info.index_lut
            )) 
            for buffer in unique_buffers
        ]

        return BufferUsageInfo(
            index_lut,
            accesses,
            reads,
            writes,
            guaranteed_writes,
            local_only_reads,
        )


class UnfinishedBufferUsageInfo(BufferUsageInfo):

    def __init__(self, ir: Ir_CaKernel) -> None:
        self.index_lut = {}

        for i, (buf_name, buf) in enumerate(ir.base.buffers.items()):
            self.index_lut[buf_name] = i

            for j, fld in enumerate(buf.cact._fields):
                self.index_lut[(buf_name,fld)] = j

        buffers = ir.base.buffers 

        initial_bitmasks = [0b0 << len(b.cact._fields) for b in buffers.values()]

        self.accesses = initial_bitmasks.copy()
        self.reads = initial_bitmasks.copy()
        self.writes = initial_bitmasks.copy()
        self.guaranteed_writes = initial_bitmasks.copy()
        self.local_only_reads = initial_bitmasks.copy()
    
    def _add_accesses(self, buffer: int, field: int):
        self.accesses[buffer] |= field # type: ignore

    def add_read(self, buffer,field, is_local=False):
        lut = self.index_lut
        buf,fld = lut[buffer], 0b1 << lut[(buffer,field)]

        self.reads[buf] |= fld # type: ignore

        if bool(self.local_only_reads[buf] & fld) and not is_local:
            self.local_only_reads[buf] ^= fld # type: ignore

        if not bool(self.reads[buf] & fld) and is_local:
            self.local_only_reads[buf] |= fld # type: ignore

        self._add_accesses(buf,fld)

    def add_write(self, buffer,field, guaranteed: bool = False):
        lut = self.index_lut
        buf,fld = lut[buffer], 0b1 << lut[(buffer,field)]
        self.writes[buf] |= fld # type: ignore
        if guaranteed:
            self.guaranteed_writes[buf] |= fld # type: ignore
        self._add_accesses(buf,fld)

    def finalized(self) -> BufferUsageInfo:
        return BufferUsageInfo(
            self.index_lut,
            tuple(self.accesses),
            tuple(self.reads),
            tuple(self.writes),
            tuple(self.guaranteed_writes),
            tuple(self.local_only_reads)
        )
    
    
MDK_BUFFER_USAGE_INFO = MetadataKey[BufferUsageInfo]('', 'buffer_usage_info', doc="The kernel's buffer usage info.")