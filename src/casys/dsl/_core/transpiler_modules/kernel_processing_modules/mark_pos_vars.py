import copy
from typing import Any
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._utils.ast_utils import map_call_args_to_kwargs
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern, Filter, OneOrMore
from casys._ast_pattern_utils.ast_pattern_templates import match_in_expr, match_func_call

from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_POS_VARS
from casys.dsl.kernel_utils import (
    k_mark_pos, k_assure_bounds
)

class MarkPosVars(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Marking positional variables')

        ptrn_mark_pos = [
            Collect(
                pattern=NodePattern(
                    ast.Assign,
                    targets=NodePattern(
                        ast.Tuple,
                        elts=OneOrMore(
                        NodePattern(ast.Name, id=Bind('targets'))
                    )
                    ),
                ), key='assign',
            ),
            Collect(match_in_expr(Collect(match_func_call(
                k_mark_pos
            ),
            key='mark_pos'),), 'expr')
        ]

        ptrn_aug_assign = [
            Collect(
                NodePattern(
                    ast.AugAssign,
                    target=Collect(NodePattern(ast.Name,id=Filter(lambda n: n in pos_vars)), 'target'),
                    op=Bind('op'),
                    value=Bind('value'),
                ),
                'assign'
            )
        ]

        ptrn_assure_bounds = [
            Collect(
                pattern=NodePattern(
                    ast.Assign,
                    targets=NodePattern(
                        ast.Tuple,
                        elts=OneOrMore(
                        NodePattern(ast.Name, id=Bind('targets'))
                    )
                    ),
                ), key='assign',
            ),
            Collect(
            match_in_expr(Collect(match_func_call(
                k_assure_bounds
            ),
            key='assure_bounds'),), 'expr')
        ]

        pos_vars: dict[str,int]

        def handle_mark_pos(m: dict[str, Any]) -> list[ast.AST]:
            targets: list[str] = m['targets']
            assign_node: ast.Assign = m['assign']
            mark_pos: casys_ast.Cs_Macro = m['mark_pos']
            assert isinstance(mark_pos.func, ast.Name)
            
            mark_pos.func.id

            args = map_call_args_to_kwargs(mark_pos, k_mark_pos)

            axes: list[int] = args.get('args', list(range(len(targets))))

            for target, axis in zip(targets[:len(targets)], axes):
                pos_vars[target] = axis

            return [assign_node]
        
        def handle_aug_assign(m: dict[str, Any]) -> list[ast.AST]:
            assign: ast.AnnAssign = m['assign']
            target: ast.Subscript = m['target']
            op: ast.operator = m['op']
            value: ast.expr = m['value']

            target_copy = copy.deepcopy(target)

            new_assign = ast.Assign(
                targets=[target],
                value = ast.BinOp(target_copy, op, value),
            )
            casys_ast.copy_meta(new_assign, assign)
            return [new_assign]
        
        def handle_assure_bounds(m: dict[str, Any]) -> list[ast.AST]:
            targets: list[str] = m['targets']
            assign_node: ast.Assign = m['assign']
            assure_bounds: casys_ast.Cs_Macro = m['assure_bounds']
            assert isinstance(assure_bounds.func, ast.Name)
            
            assure_bounds.func.id

            node_metadata = (
                meta if 
                (meta := casys_ast.get_meta(assign_node)) is not None
                else 
                casys_ast.AstNodeMeta(
                    verified_bounds=True
                )
            )
            node_metadata.verified_bounds = True

            return [assign_node]

        
        for name, kernel in ir.kernels.items():
            pos_vars = kernel.metadata.get(MDK_POS_VARS)

            (tf1:=PatternTransformer(ptrn_mark_pos, {
                'assign':handle_mark_pos,
                'expr':  None
            })).visit(kernel.ir_ast)

            (tf2:=PatternTransformer(ptrn_aug_assign, {
                'assign':handle_aug_assign,
            })).visit(kernel.ir_ast)

            (tf3:=PatternTransformer(ptrn_assure_bounds, {
                'assign':handle_assure_bounds,
                'assure_bounds':  None,
                'expr':  None
            })).visit(kernel.ir_ast)

            if tf1.matches:
                trkr.add_snapshot(
                    ast_node=kernel.ir_ast,
                    tags=(
                        f_tag_kernel(name),
                        f_tag_transpiler_module(self)
                    ),
                    metadata=kernel.metadata
                )

        trkr.exit_phase()