from typing import Any, Callable, Sequence
from casys._ast_pattern_utils.ast_pattern_templates import match_func_call, match_in_expr
from casys.dsl import kernel_utils
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker

import ast
from casys.dsl._core import casys_ast, core_macros
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern, Filter, OneOrMore

class CreateParallelGroups(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Creating parallel groups for simulation step function')

        ir_ast = ir.step_func.ir_ast

        ptrn_split_calls = [
            OneOrMore(pattern=Collect(match_in_expr(
                    pattern=match_func_call(kernel_utils.step_func_split),
                ), 'split_calls')
            )
        ]

        def handle_split_calls(m: dict[str, Any]) -> list[ast.AST]:
            split_calls: list[ast.Expr] = m['split_calls']
            
            target_buffers: set[str] = set()
            for expr in split_calls:
                split_call: ast.Call = expr.value # type: ignore
                args: dict[str, Any] = core_macros.MacroSpec.parse_and_validate(kernel_utils.step_func_split,split_call)

            return [split_calls[0]]

        (tf:=PatternTransformer(
            ptrn_split_calls,
            {'split_calls': handle_split_calls}
        )).visit(ir_ast)

        if tf.matches:
            trkr.add_snapshot(
                tags=(TAG_STEP_FUNC,),
                ast_node=ir_ast
            )

        # -- Form parallel groups phase --

        ptrn_swaps = OneOrMore(Collect(
            Collect(match_in_expr(Collect(
                NodePattern(
                    casys_ast.Cs_DoubleBufferSwaps, 
                    buffers=Bind(name='swaps')
                ),'call')),
            'expr'),'calls')
        )

        ptrn_k_calls = OneOrMore(Collect(
            Collect(
            match_in_expr(
                Filter(lambda n: 
                            isinstance(n,(casys_ast.Cs_KernelCall, casys_ast.Cs_DoubleBufferSwaps)),
                            key='call'
                        )
                    ),
                'expr'),
            'calls')
        )

        ptrn_form_groups = [
            ptrn_k_calls,
            Collect(match_in_expr(match_func_call(kernel_utils.step_func_split)), 'split_call_expr')
        ]

        form_pgroup_from_calls: Callable[[dict[str,Any]], list[ast.AST]] = (lambda m:[
            casys_ast.Cs_ParallelGroup(
                swaps=[
                        buffer
                        for c in m['calls']
                        if isinstance(c['call'],casys_ast.Cs_DoubleBufferSwaps)
                        for buffer in c['swaps']
                    ],
                calls=[c['call'].desc for c in m['calls'] if isinstance(c['call'],casys_ast.Cs_KernelCall)] if 'calls' in m else [],
                sync_r2w=[],
                sync_w2r=[],
            )
        ])

        # We treat make swaps be their own parallel group to preserve user ordering.
        # The transpiler will clean up and optimize the groupings.

        (tf1:=PatternTransformer(
            pattern=[ptrn_swaps],
            actions={
                'calls': form_pgroup_from_calls
            }
        )).visit(ir_ast)
        
        # Create the kernel call parallel groups based on splits.

        (tf2:=PatternTransformer(
            pattern=ptrn_form_groups,
            actions={
                'calls': None,
                'split_call_expr': form_pgroup_from_calls
            }
        )).visit(ir_ast)

        # Ensure groupings of kernel calls without any splits are also grouped.
        # Like tailing calls or calls that were sandwiched between double buffer swaps.

        (tf3:=PatternTransformer(
            pattern=[ptrn_k_calls],
            actions={
                'calls': form_pgroup_from_calls
            }
        )).visit(ir_ast)

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC,),
            ast_node=ir_ast
        )

        trkr.exit_phase()