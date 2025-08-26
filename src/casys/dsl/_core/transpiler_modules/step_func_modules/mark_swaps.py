from typing import Any, Sequence, cast

from numba.cuda import target
from casys._ast_pattern_utils.ast_pattern_templates import match_func_call, match_in_expr
from casys.dsl import kernel_utils
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker

from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_SWAP_TARGETS

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore

from casys.dsl._core import core_macros

class MarkSwaps(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Marking double buffer swaps in simulation step function')

        buffers = ir.step_func.base.buffers
        ir_ast = ir.step_func.ir_ast

        ptrn_swap_calls = [
            OneOrMore(pattern=Collect(match_in_expr(
                    pattern=match_func_call(kernel_utils.step_func_swap),
                ), 'swap_calls')
            )
        ]

        swapped_buffers_merged: set[str] = set()

        def handle_swap_calls(m: dict[str, Any]) -> list[ast.AST]:
            swap_calls: list[ast.Expr] = m['swap_calls']
            
            target_buffers: set[str] = set()
            for expr in swap_calls:
                swap_call: ast.Call = expr.value # type: ignore
                args: dict[str, Any] = core_macros.MacroSpec.parse_and_validate(kernel_utils.step_func_swap,swap_call)
                if 'buffers' in args:
                    arg_buffers: list[Any] = cast(ast.List, args['buffers']).elts
                    for node in arg_buffers:
                        if not isinstance(node, ast.Name):
                            raise TranspileError(f'Invalid arguments for {kernel_utils.step_func_swap.__name__} call', node)
                        if node.id not in buffers:
                            raise TranspileError(f"No buffer named '{node.id}'", node)
                    cast(list[ast.Name], arg_buffers)

                    target_buffers.update(n.id for n in arg_buffers)

            swapped_buffers_merged.update(target_buffers)

            new_node = ast.Expr(casys_ast.Cs_DoubleBufferSwaps(list(target_buffers))) # type: ignore
            return [new_node]            

        (tf:=PatternTransformer(
            ptrn_swap_calls,
            {'swap_calls': handle_swap_calls}
        )).visit(ir_ast)

        ir.step_func.metadata.set(MDK_SWAP_TARGETS, swapped_buffers_merged)
        
        if tf.matches:
            trkr.add_snapshot(
                tags=(TAG_STEP_FUNC,),
                ast_node=ir_ast
            )

        trkr.exit_phase()