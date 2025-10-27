from typing import TYPE_CHECKING

if TYPE_CHECKING: 
    from casys.dsl._core.soa_field_usage_info_helper import SoaFieldUsageInfo

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, Collect, Bind, NodePattern

from casys.dsl._core.ir_metadata_specs.md_kernels_base import (
    MDK_ALIASES,
    MDK_SOA_FIELD_USAGE_INFO,
)

from casys.dsl._core.soa_field_usage_info_helper import UnfinishedSoaFieldUsageInfo

class AnalyzeSoaFieldsUsage(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Analyzing kernel SoA fields usage')

        aliases: dict[str, ast.AST]

        field_usage_info: SoaFieldUsageInfo

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

        ptrn_soa_field_ref = [
            Collect(
                pattern=NodePattern(
                    node_type=ast.Subscript,
                    value=NodePattern(casys_ast.Cs_SoaFieldRef, field=Bind('fld'), ctx=Bind('ctx')),
                    slice=Bind('slice')
                ),
                key='subscript'
            ),
        ]

        for name, kernel in ir.kernels.items():
            aliases = kernel.metadata.get(MDK_ALIASES)
            field_usage_info = UnfinishedSoaFieldUsageInfo(ir)

            (finder:=FindGuaranteedSoaFieldAccesses()).visit(kernel.ir_ast)
            guaranteed_field_writes, guaranteed_field_reads = finder.get_guaranteed_reads_and_writes()

            (finder:=PatternFinder(ptrn_soa_field_ref)).visit(kernel.ir_ast)

            for m in finder.matches:
                fld = m['fld']

                f = fld.name

                slice_tuple: ast.Tuple = m['slice']

                is_local = check_local(slice_tuple)
                casys_ast.get_meta(m['subscript']).local_access = is_local
                casys_ast.get_meta(slice_tuple).local_access = is_local

                match m['ctx']:
                    case ast.Load():
                        field_usage_info.add_read(f, is_local=is_local)

                    case ast.Store():
                        is_guaranteed = f in guaranteed_field_writes
                        field_usage_info.add_write(f, guaranteed=is_guaranteed)

            kernel.metadata.set(MDK_SOA_FIELD_USAGE_INFO, field_usage_info.finalized())

            if finder.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                    metadata=kernel.metadata
                )

        trkr.exit_phase()

class FindGuaranteedSoaFieldAccesses(ast.NodeVisitor):
    writes: set[str]
    writes_before_return: dict[int, set[str]]
    conditional_writes: list[set[str]]

    reads: set[str]
    reads_before_return: dict[int, set[str]]
    conditional_reads: list[set[str]]

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

    def get_guaranteed_reads_and_writes(self) -> tuple[set[str], set[str]]:
        intersected_set_writes: set[str] | None = None
        intersected_set_reads: set[str] | None = None

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

        intersected_set_writes: set[str] | None = intersected_set_writes if intersected_set_writes else set()
        intersected_set_reads: set[str] | None = intersected_set_reads if intersected_set_reads else set()

        return intersected_set_writes, intersected_set_reads # type: ignore
    
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, casys_ast.Cs_SoaFieldRef):
            fld = node.value.field.name
            match node.ctx:
                case ast.Load():
                    if self.conditional_depth == 0:
                        self.reads.add(fld)
                    else:
                        self.conditional_reads[-1].add(fld)
                case ast.Store():
                    if self.conditional_depth == 0:
                        self.writes.add(fld)
                    else:
                        self.conditional_writes[-1].add(fld)

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

        