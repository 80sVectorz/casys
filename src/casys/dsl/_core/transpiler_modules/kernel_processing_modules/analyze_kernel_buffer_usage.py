from os import write
from typing import DefaultDict, Sequence, TYPE_CHECKING

if TYPE_CHECKING: 
    from casys.dsl._core.ir_metadata_specs.md_kernels_base import BufferUsageInfo

from collections import defaultdict

from casys.dsl._core.descriptors import CactBufferDescriptor
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore

from casys.dsl._core.ir_metadata_specs.md_kernels_base import (
    MDK_ALIASES,
    MDK_BUFFER_USAGE_INFO,
    UnfinishedBufferUsageInfo,
)

class AnalyzeBufferUsage(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Analyzing kernel buffer usage')

        aliases: dict[str, ast.AST]

        buffer_usage_info: BufferUsageInfo

        def check_local(tuple_node: ast.Tuple) -> bool:
            """Checks if slice is equal to the kernel position"""

            elts = [
                el
                for el in tuple_node.elts
                if not isinstance(el, (casys_ast.Cs_RdIdx, casys_ast.Cs_WrIdx))
            ]

            for i,islice in enumerate(elts):
                alias = getattr(islice,'id',None)
                normalized_node = aliases.get(alias,islice) if alias else islice
                if not (getattr(normalized_node, 'ax', None) == i and isinstance(normalized_node,casys_ast.Cs_KPos)):
                    return False
            return True

        ptrn_buffer_ref = [
            Collect(
                pattern=NodePattern(
                    node_type=ast.Subscript,
                    value=NodePattern(casys_ast.Cs_BufferRef, b=Bind('b'), f=Bind('f')),
                    ctx = Bind('ctx'),
                    slice = Bind('slice')
                ),
                key='subscript'
            ),
        ]

        for name, kernel in ir.kernels.items():
            aliases = kernel.metadata.get(MDK_ALIASES)
            buffer_usage_info = UnfinishedBufferUsageInfo(kernel)

            (finder := FindGuaranteedBufferAccesses()).visit(kernel.ir_ast)
            guaranteed_buffer_writes, guaranteed_buffer_reads = finder.get_guaranteed_reads_and_writes()

            (finder:=PatternFinder(ptrn_buffer_ref)).visit(kernel.ir_ast)

            for m in finder.matches:
                b,f = m['b'], m['f']

                slice_tuple: ast.Tuple = m['slice']

                is_local = check_local(slice_tuple)
                casys_ast.get_meta(m['subscript']).local_access = is_local
                casys_ast.get_meta(slice_tuple).local_access = is_local

                match m['ctx']:
                    case ast.Load():
                        buffer_usage_info.add_read(b,f, is_local=is_local)

                    case ast.Store():
                        is_guaranteed = (b,f) in guaranteed_buffer_writes
                        buffer_usage_info.add_write(b,f, guaranteed=is_guaranteed)

            kernel.metadata.set(MDK_BUFFER_USAGE_INFO, buffer_usage_info.finalized())

            if finder.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                    metadata=kernel.metadata
                )

        trkr.exit_phase()

class FindGuaranteedBufferAccesses(ast.NodeVisitor):
    writes: set[tuple[str,str]]
    writes_before_return: dict[int, set[tuple[str,str]]]
    conditional_writes: list[set[tuple[str,str]]]

    reads: set[tuple[str,str]]
    reads_before_return: dict[int, set[tuple[str,str]]]
    conditional_reads: list[set[tuple[str,str]]]

    conditional_depth: int = 0

    def __init__(self) -> None:
        self.writes = set()
        self.writes_before_return = {}
        self.conditional_writes = []

        self.reads = set()
        self.reads_before_return = {}
        self.conditional_reads = []

        self.visit_IfExp = self.on_conditional_block
        self.visit_If = self.on_conditional_block
        self.visit_While = self.on_conditional_block
        self.visit_For = self.on_conditional_block

    def get_guaranteed_reads_and_writes(self) -> tuple[set[tuple[str,str]], set[tuple[str,str]]]:
        intersected_set_writes: set[tuple[str,str]] | None = None
        intersected_set_reads: set[tuple[str,str]] | None = None

        for k,v in self.writes_before_return.items():
            if intersected_set_writes is None:
                intersected_set_writes = v.copy()
                continue

            intersected_set_writes = intersected_set_writes.intersection(v)

        for k,v in self.reads_before_return.items():
            if intersected_set_reads is None:
                intersected_set_reads = v.copy()
                continue

            intersected_set_reads = intersected_set_reads.intersection(v)

        intersected_set_writes: set[tuple[str,str]] | None = intersected_set_writes if intersected_set_writes else set()
        intersected_set_reads: set[tuple[str,str]] | None = intersected_set_reads if intersected_set_reads else set()

        return intersected_set_writes, intersected_set_reads # type: ignore
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, casys_ast.Cs_BufferRef):
            match node.ctx:
                case ast.Load():
                    if self.conditional_depth == 0:
                        self.reads.add((node.value.b,node.value.f))
                    else:
                        self.conditional_reads[-1].add((node.value.b,node.value.f))
                case ast.Store():
                    if self.conditional_depth == 0:
                        self.writes.add((node.value.b,node.value.f))
                    else:
                        self.conditional_writes[-1].add((node.value.b,node.value.f))

        self.visit(node.slice)

    def on_conditional_block(self, node: ast.If | ast.While | ast.For | ast.IfExp) -> None:
        self.conditional_depth += 1
        self.conditional_writes.append(set())
        self.conditional_reads.append(set())

        if isinstance(node,ast.IfExp):
            self.visit(node.body)
            self.visit(node.orelse)
        else:
            for child in node.body:
                self.visit(child)

        self.conditional_depth -= 1
        self.conditional_writes.pop()
        self.conditional_reads.pop()
    
    def visit_Return(self, node: ast.Return) -> None:
        all_writes = self.writes.copy()
        all_reads = self.reads.copy()

        [all_writes.update(cw) for cw in self.conditional_writes]
        [all_reads.update(cr) for cr in self.conditional_reads]

        self.writes_before_return[id(node)] = all_writes
        self.reads_before_return[id(node)] = all_reads

        