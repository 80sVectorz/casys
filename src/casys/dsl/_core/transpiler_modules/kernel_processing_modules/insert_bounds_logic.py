import copy
import re
from typing import Any, Sequence, cast
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._utils.ast_utils import map_call_args_to_kwargs
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, BottomUpPatternTransformer, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore
from casys._ast_pattern_utils.ast_pattern_templates import match_in_expr, match_func_call

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_ALIASES, MDK_POS_VARS

class InsertBoundsLogic(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Inserting bounds logic')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        pos_vars: dict[str,int]
        aliases: dict[str,ast.AST]

        ptrn_assign = [
            Collect(
                pattern=NodePattern(
                node_type=ast.Assign,
                    targets=Bind('targets_list'),
                    value=Bind('assign_value'),
                ),
                key='assign'
            ),
        ]
        ptrn_targets_name = NodePattern(ast.Name, id=Bind('target'))
        ptrn_targets_tuple = NodePattern(node_type=ast.Tuple,
            elts=OneOrMore(
                pattern=NodePattern(ast.Name, id=Bind('targets'))
            )
        )

        ptrn_assign_values = NodePattern(ast.Tuple, elts=Bind('values'))

        ptrn_aug_assign = [
            Collect(
                NodePattern(
                    ast.AugAssign,
                    target=NodePattern(ast.Name,id=Filter(lambda n: n in pos_vars, 'target')),
                    op=Bind('op'),
                    value=Bind('value'),
                ),
                'assign'
            )
        ]

        ptrn_walrus_assign = [
            Collect(
                NodePattern(
                    ast.NamedExpr,
                    target=NodePattern(ast.Name,id=Filter(lambda n: n in pos_vars, 'target')),
                    op=Bind('op'),
                    value=Bind('value'),
                ),
                'assign'
            )
        ]

        ptrn_subscript = [
            Collect(
                pattern=NodePattern(
                node_type=ast.Subscript,
                    slice=NodePattern(
                        node_type=ast.Tuple,
                        elts=Bind('slices')
                    ),
                ),
                key='subscript'
            ),
        ]

        def infer_bounds_verified(target_ax: int, value: ast.expr) -> bool:
            return bool(
                isinstance(value, ast.Name) and pos_vars.get(value.id) == target_ax
                or
                NodePattern(ast.BinOp,
                    op=NodePattern(ast.Mod),
                    right=Filter(
                    lambda n: (
                        getattr(node2:=aliases.get(getattr(n,'id',''),n), 'ax', None) == target_ax
                        and isinstance(node2,casys_ast.Cs_AxisSize)
                    ),
                    'x')
                ).match(value)
            )

        def handle_assign(m: dict[str, Any]) -> list[ast.AST]:
            assign: ast.Assign = m['assign']
            targets_list: list[ast.AST] = m['targets_list']
            assign_value: ast.expr = m['assign_value']

            if casys_ast.get_meta(assign).verified_bounds:
                return [assign]

            if m_targets:=ptrn_targets_tuple.match(targets_list[0]):
                targets: list[str] = m_targets['targets']

                if not any(t in pos_vars for t in targets): return [assign]
                
                m_vals = ptrn_assign_values.match(assign_value)

                if not m_vals:
                    raise TranspileError("Couldn't interpret values for position variable assign operation.", assign)

                values: list[ast.expr] = m_vals['values']

                new_values = []

                for target, value in zip(targets, values):
                    target_ax = pos_vars.get(target)
                    if target_ax is None or infer_bounds_verified(target_ax,value):
                        new_values.append(value)
                        continue

                    new_values.append(ast.BinOp(
                        value,
                        ast.Mod(),
                        casys_ast.Cs_AxisSize(target_ax)
                    ))

                cast(ast.Tuple,assign.value).elts = new_values
                casys_ast.get_meta(assign).verified_bounds = True
            
            elif m_target:=ptrn_targets_name.match(targets_list[0]):
                target = m_target['target']
                value = assign.value

                target_ax = pos_vars.get(target)
                if target_ax is None or infer_bounds_verified(target_ax,value):
                    casys_ast.get_meta(assign).verified_bounds = True
                    return [assign]

                assign.value = ast.BinOp(assign.value, ast.Mod(), casys_ast.Cs_AxisSize(target_ax))
                casys_ast.get_meta(assign).verified_bounds = True

            return [assign]

        def handle_aug_assign(m: dict[str, Any]) -> list[ast.AST]:
            assign: ast.AugAssign = m['assign']
            target: str = m['target']
            op: ast.operator = m['op']
            value: ast.expr = m['value']

            if casys_ast.get_meta(assign).verified_bounds: return [assign]

            target_ax = pos_vars[target]

            new_assign = ast.Assign(
                targets=[assign.target],
                value = ast.BinOp(
                    ast.BinOp(ast.Name(target), op, value),
                    ast.Mod(),
                    casys_ast.Cs_AxisSize(target_ax)
                ),
            )
            meta = casys_ast.copy_meta(new_assign, assign)
            meta.verified_bounds = True
            return [new_assign]

        def handle_walrus_assign(m: dict[str, Any]) -> list[ast.AST]:
            assign: ast.AugAssign = m['assign']
            target: str = m['target']
            value: ast.expr = m['value']

            if casys_ast.get_meta(assign).verified_bounds: return [assign]

            target_ax = pos_vars[target]

            assign.value = ast.BinOp(value,ast.Mod(),casys_ast.Cs_AxisSize(target_ax))
            casys_ast.get_meta(assign).verified_bounds = True
            return [assign]

        def handle_subscript(m: dict[str, Any]) -> list[ast.AST]:
            node: ast.Subscript = m['subscript']
            slices: list[ast.expr] = m['slices']

            if (
                (casys_ast.get_meta(node).verified_bounds)
            ): return [node]

            new_slices = []

            for i, islice in enumerate(slices):
                if infer_bounds_verified(i, islice):
                    new_slices.append(islice)
                    continue

                new_slices.append(ast.BinOp(
                    islice,
                    ast.Mod(),
                    casys_ast.Cs_AxisSize(i)
                ))

            if len(new_slices) < len(dims):
                for _ in range(len(dims)-len(new_slices)):
                    new_slices.append(casys_ast.Cs_KPos(len(new_slices)))

            casys_ast.get_meta(node).verified_bounds = True

            cast(ast.Tuple,node.slice).elts = new_slices
            return [node]
        
        for name, kernel in ir.kernels.items():
            pos_vars = kernel.metadata.get(MDK_POS_VARS)
            aliases = kernel.metadata.get(MDK_ALIASES)

            transformers = (
                PatternTransformer(ptrn_assign, {
                    'assign':handle_assign
                }),

                PatternTransformer(ptrn_aug_assign, {
                    'assign':handle_aug_assign
                }),

                PatternTransformer(ptrn_walrus_assign, {
                    'assign':handle_walrus_assign
                }),

                BottomUpPatternTransformer(ptrn_subscript, {
                    'subscript':handle_subscript
                }),
            )

            for tf in transformers: tf.visit

            if any(tf.matches for tf in transformers):
                trkr.add_snapshot(
                    ast_node=kernel.ir_ast,
                    tags=(
                        f_tag_kernel(name),
                        f_tag_transpiler_module(self)
                    )
                )

        trkr.exit_phase()


