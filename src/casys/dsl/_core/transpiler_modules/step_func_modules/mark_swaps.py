from typing import Any, Sequence, cast

from numba.cuda import target
from casys._ast_pattern_utils.ast_pattern_templates import match_func_call, match_in_expr
from casys.dsl import kernel_utils
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_SOA_LAYOUT
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_SWAP_TARGETS

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore

from casys.dsl._core import core_macros

class MarkSwaps(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Marking double buffer swaps in simulation step function')

        world_schema = ir.world_schema
        soa_layout = ir.metadata.get(MDK_SOA_LAYOUT)
        ir_ast = ir.step_func.ir_ast

        ptrn_swap_calls = [
            OneOrMore(pattern=Collect(match_func_call(kernel_utils.step_func_swap)
            , 'swap_calls'))
        ]

        swapped_layers_merged: set[str] = set()

        def handle_swap_calls(m: dict[str, Any]) -> list[ast.AST]:
            swap_calls: list[casys_ast.Cs_Macro] = m['swap_calls']
            
            target_layers: set[str] = set()
            for swap_call in swap_calls:
                args: dict[str, Any] = core_macros.MacroSpec.parse_and_validate(kernel_utils.step_func_swap,swap_call)
                if 'layers' in args:
                    arg_layers: list[Any] = cast(ast.List, args['layers']).elts
                    for node in arg_layers:
                        if not isinstance(node, ast.Name):
                            raise TranspileError(f'Invalid arguments for {kernel_utils.step_func_swap.__name__} call', node)
                        if node.id not in world_schema.groups:
                            raise TranspileError(f"No layer named '{node.id}'", node)
                    cast(list[ast.Name], arg_layers)
                    target_layers.update(n.id for n in arg_layers)

            swapped_layers_merged.update(target_layers)

            new_node = ast.Expr(casys_ast.Cs_DoubleBufferSwaps(list(target_layers))) # type: ignore
            return [new_node]            

        (tf:=PatternTransformer(
            ptrn_swap_calls,
            {'swap_calls': handle_swap_calls}
        )).visit(ir_ast)

        ir.step_func.metadata.set(MDK_SWAP_TARGETS, swapped_layers_merged)
        
        if tf.matches:
            trkr.add_snapshot(
                tags=(TAG_STEP_FUNC,),
                ast_node=ir_ast
            )

        trkr.exit_phase()