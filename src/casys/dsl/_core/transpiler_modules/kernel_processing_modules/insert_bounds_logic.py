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

        ptrn_assign = [
            Collect(
                pattern=NodePattern(
                node_type=ast.Assign,
                    targets=NodePattern(
                        node_type=ast.Tuple,
                        elts=OneOrMore(
                            pattern=NodePattern(ast.Name, id=Bind('targets'))
                        )
                    ),
                    value=NodePattern(
                        ast.Tuple,
                        elts=Bind('values')
                    ),
                ),
                key='assign'
            ),
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

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        pos_vars: dict[str,int]
        aliases: dict[str,ast.AST]

        def handle_assign(m: dict[str, Any]) -> list[ast.AST]:
            targets: list[str] = m['targets']
            node: ast.Assign = m['assign']
            values: list[ast.expr] = m['values']

            if (
                (casys_ast.get_meta(node).verified_bounds) or
                not any(t in pos_vars for t in targets)
            ): return [node]

            new_values = []

            for target, value in zip(targets, values):
                target_ax = pos_vars.get(target)
                if (
                    target_ax is None
                    or (isinstance(value, ast.Name) and pos_vars.get(value.id) == target_ax)
                    or (NodePattern(ast.BinOp, op=NodePattern(ast.Mod), right=Filter(
                        lambda n: getattr(n2:=aliases.get(getattr(n,'id',''),n), 'ax', None) == target_ax and isinstance(n2,casys_ast.Cs_AxisSize),
                        'x')
                        ).match(value, {})
                    )
                ):
                    new_values.append(value)
                    continue

                new_values.append(ast.BinOp(
                    value,
                    ast.Mod(),
                    casys_ast.Cs_AxisSize(target_ax)
                ))

            casys_ast.get_meta(node).verified_bounds = True

            cast(ast.Tuple,node.value).elts = new_values
            return [node]

        def handle_subscript(m: dict[str, Any]) -> list[ast.AST]:
            node: ast.Subscript = m['subscript']
            slices: list[ast.expr] = m['slices']

            if (
                (casys_ast.get_meta(node).verified_bounds)
            ): return [node]

            new_slices = []

            for i, islice in enumerate(slices):
                if (
                    NodePattern(ast.Name, id=Filter(lambda n : pos_vars.get(n) == i, 'x')).match(islice, {})
                    or (NodePattern(ast.BinOp, op=NodePattern(ast.Mod), right=Filter(
                        lambda n: getattr(n2:=aliases.get(getattr(n,'id',''),n), 'ax', None) == i and isinstance(n2,casys_ast.Cs_AxisSize),
                        'x')
                        ).match(islice, {})
                    )
                    or aliases.get(getattr(islice,'id',''),islice)
                ):
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

            (tf1:=PatternTransformer(ptrn_assign, {
                'assign':handle_assign
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


