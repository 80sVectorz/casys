from typing import Any, Sequence, cast
import copy
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._utils.ast_utils import map_call_args_to_kwargs
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, BottomUpPatternTransformer, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore
from casys._ast_pattern_utils.ast_pattern_templates import match_in_expr, match_func_call

class InsertDoubleBufferIndexing(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Inserting double-buffer indexing')

        ptrn_subscript = [
            Collect(
                pattern=NodePattern(
                node_type=ast.Subscript,
                    value=NodePattern(
                        node_type=casys_ast.Cs_BufferRef,
                        b=Bind('buffer'),
                        f=Bind('field'),
                    ),
                    slice=NodePattern(
                        node_type=ast.Tuple,
                        elts=Bind('slices')
                    ),
                    ctx=Bind('ctx')
                ),
                key='subscript'
            ),
        ]

        ptrn_aug_assign = [
            Collect(
                NodePattern(
                    ast.AugAssign,
                    target=Collect(NodePattern(ast.Subscript,value=NodePattern(casys_ast.Cs_BufferRef)), 'target'),
                    op=Bind('op'),
                    value=Bind('value'),
                ),
                'assign'
            )
        ]

        def handle_aug_assign(m: dict[str, Any]) -> list[ast.AST]:
            assign: ast.AnnAssign = m['assign']
            target: ast.Subscript = m['target']
            op: ast.operator = m['op']
            value: ast.expr = m['value']

            target_copy = copy.deepcopy(target)
            target_copy.ctx = ast.Load()

            new_assign = ast.Assign(
                targets=[target],
                value = ast.BinOp(target_copy, op, value),
            )
            casys_ast.copy_meta(new_assign, assign)
            return [new_assign]

        def handle_subscript(m: dict[str, Any]) -> list[ast.AST]:
            node: ast.Subscript = m['subscript']
            slices: list[ast.expr] = m['slices']
            ctx: ast.Load | ast.Store = m['ctx']            

            new_slices = [
                casys_ast.Cs_RdIdx() if isinstance(ctx, ast.Load) else casys_ast.Cs_WrIdx(),
                *slices
            ]

            cast(ast.Tuple,node.slice).elts = new_slices
            return [node]
        
        for name, kernel in ir.kernels.items():

            (tf1:=PatternTransformer(ptrn_aug_assign, {
                'assign':handle_aug_assign
            })).visit(kernel.ir_ast)

            (tf2:=BottomUpPatternTransformer(ptrn_subscript, {
                'subscript':handle_subscript
            })).visit(kernel.ir_ast)

            if tf1.matches or tf2.matches:
                trkr.add_snapshot(
                    ast_node=kernel.ir_ast,
                    tags=(
                        f_tag_kernel(name),
                        f_tag_transpiler_module(self)
                    )
                )

        trkr.exit_phase()


