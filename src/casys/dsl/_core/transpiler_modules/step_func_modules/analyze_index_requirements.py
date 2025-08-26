from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_DEDICATED_IDX_IDS
from casys.dsl._core.kernel_values import KV_WR_IDX

from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, Bind, NodePattern 

class AnalyzeIndexRequirements(TranspilerModule):

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Analyzing double index requirements')

        ptrn_swaps = [
            NodePattern(
                node_type=casys_ast.Cs_DoubleBufferSwaps,
                buffers=Bind('buffers')
            ),
        ]

        swapped_buffers_merged: set[str] = set()
        (finder:=PatternFinder(ptrn_swaps)).visit(ir.step_func.ir_ast)

        for m in finder.matches:
            swapped_buffers_merged.update(m['buffers'])
        
        dedicated_index_ids = {
            buffer:f'{KV_WR_IDX}_{buffer}' for buffer in swapped_buffers_merged
        }

        ir.step_func.metadata.set(MDK_DEDICATED_IDX_IDS, dedicated_index_ids)

        if finder.matches:
            trkr.add_snapshot(
                tags=(TAG_STEP_FUNC, f_tag_transpiler_module(self)),
                metadata=ir.step_func.metadata
            )
        
        trkr.exit_phase()

            