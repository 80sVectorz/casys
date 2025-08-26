from __future__ import annotations

from typing import TYPE_CHECKING

from casys.dsl._core.core_transpiler import TranspilerModule

if TYPE_CHECKING:
    from casys.dsl._core.descriptors import BufferUsageInfo
    from casys.dsl._core.descriptors import KernelCallDescriptor
    from casys.dsl._core.ir import Ir_CaSys

from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_KCALL_BUFFER_USAGE_INFO 

from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import Collect, PatternFinder, Bind, NodePattern 

class AnalyzeKCallBufferUsage(TranspilerModule):

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Analyzing buffer usage per kernel call in step function')

        ptrn_kcalls = [
            Collect(
            NodePattern(
                node_type=casys_ast.Cs_KernelCall,
                desc=Bind('desc')
            ),'kcall')
        ]

        (finder:=PatternFinder(ptrn_kcalls)).visit(ir.step_func.ir_ast)

        bounds_access_map: dict[KernelCallDescriptor, BufferUsageInfo] = {}
        
        for m in finder.matches:
            desc: KernelCallDescriptor = m['desc']
            bounds_access_map[desc] = desc.instantiate_access(ir)

        ir.step_func.metadata.set(MDK_KCALL_BUFFER_USAGE_INFO, bounds_access_map)

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC, f_tag_transpiler_module(self)),
            metadata=ir.step_func.metadata
        )
        
        trkr.exit_phase()

            