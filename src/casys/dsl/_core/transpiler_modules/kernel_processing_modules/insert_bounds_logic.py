from __future__ import annotations

from typing import Any, Sequence, cast

import ast
import numpy as np

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core import casys_ast
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS, MDK_DIMS_UNSIGNED_NB_TYPES
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_ALIASES, MDK_POS_VARS

from casys._ast_pattern_utils.assign_rewrite_template import AssignLikeRewriter, LhsEdit, SliceEdit
from casys._ast_pattern_utils.ast_pattern_engine import (
    BottomUpPatternTransformer,
    NodePattern,
    Filter,
    Collect,
    Bind,
)

class InsertBoundsLogic(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Inserting bounds logic')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)

        def infer_bounds_verified(value: ast.expr, target_ax: int) -> bool:
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

        def snippet_int_cast(e: ast.expr) -> ast.expr:
            if isinstance(e, ast.Name) and e.id in aliases:
                return e  # Ensure that local only read validation checks don't get false flagged by int cast
            return ast.Call(func=ast.Attribute(value=ast.Name(id='numpy'), attr='uint'), args=[e], keywords=[])

        def snippet_wrap_logic(value: ast.expr, ax: int) -> ast.expr:
            return snippet_int_cast(ast.BinOp(value,ast.Mod(),casys_ast.Cs_AxisSize(ax)))

        def _rhs_bounds_handler(rhs: ast.expr, *, ctx: ast.AST, meta: dict[str, Any]) -> ast.expr:
            """RHS handler used by AssignLikeRewriter to unify Assign/AugAssign/Walrus/AnnAssign."""

            if isinstance(ctx, ast.Assign):
                targets = ctx.targets
            elif isinstance(ctx, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                targets = [ctx.target]
            else:
                return rhs # Not expected

            # Tuple target: (x, y, ...) = (v1, v2, ...)
            if len(targets) == 1 and isinstance(targets[0], ast.Tuple):
                t_elts = targets[0].elts
                t_names: list[str] = []
                for t in t_elts:
                    if isinstance(t, ast.Name):
                        t_names.append(t.id)
                        continue
                    t_names.append('')

                if not any(name and name in pos_vars for name in t_names):
                    return rhs  # no positional variables

                if not isinstance(rhs, ast.Tuple):
                    raise TranspileError(
                        "Couldn't interpret values for position variable assign operation.",
                        ctx,
                    )

                r_elts = list(rhs.elts)
                new_vals: list[ast.expr] = []
                changed = False

                for name, val in zip(t_names, r_elts):
                    if not name or name not in pos_vars:
                        new_vals.append(val)
                        continue
                    ax = pos_vars[name]
                    if infer_bounds_verified(val,ax):
                        new_vals.append(val)
                    else:
                        new_vals.append(snippet_wrap_logic(val, ax))
                        changed = True

                if changed:
                    meta['changed_assigns'] = True
                casys_ast.get_meta(ctx).verified_bounds = True

                return ast.Tuple(elts=new_vals, ctx=ast.Load())

            # Single-name target: x = rhs, x += y, x := y, etc.
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                tname = targets[0].id
                ax = pos_vars.get(tname)
                if ax is None:
                    casys_ast.get_meta(ctx).verified_bounds = True
                    return rhs

                if infer_bounds_verified(rhs, ax):
                    casys_ast.get_meta(ctx).verified_bounds = True
                    return rhs

                casys_ast.get_meta(ctx).verified_bounds = True
                meta['changed_assigns'] = True
                return snippet_wrap_logic(rhs, ax)

            # Any other LHS shape: do nothing
            return rhs


        def handle_subscript(m: dict[str, Any]) -> list[ast.AST]:
            node: ast.Subscript = m['subscript']
            slices: list[ast.expr] = m['slices']

            if (
                (casys_ast.get_meta(node).verified_bounds)
            ): return [node]

            new_slices = []

            for i, islice in enumerate(slices):
                if infer_bounds_verified(islice, i):
                    new_slices.append(snippet_int_cast(islice))
                    continue

                new_slices.append(snippet_wrap_logic(islice,i))

            if len(new_slices) < len(dims):
                for _ in range(len(dims)-len(new_slices)):
                    new_slices.append(casys_ast.Cs_KPos(len(new_slices)))

            casys_ast.get_meta(node).verified_bounds = True

            cast(ast.Tuple,node.slice).elts = new_slices
            return [node]
        
        ptrn_subscript = [
            Collect(
                pattern=NodePattern(
                    node_type=ast.Subscript,
                    slice=NodePattern(node_type=ast.Tuple, elts=Bind('slices')),
                ),
                key='subscript',
            )
        ]

        for kname, kernel in ir.kernels.items():
            pos_vars: dict[str, int] = kernel.metadata.get(MDK_POS_VARS)
            aliases: dict[str, ast.AST] = kernel.metadata.get(MDK_ALIASES)

            meta: dict[str, Any] = {
                'changed_assigns': False,
            }
            assign_rw = AssignLikeRewriter(rhs_handlers=[_rhs_bounds_handler], meta=meta)
            assign_rw.visit(kernel.ir_ast)

            (subscript_tf:=BottomUpPatternTransformer(ptrn_subscript, {
                'subscript':handle_subscript
            })).visit(kernel.ir_ast)

            if meta['changed_assigns'] or subscript_tf.matches:
                trkr.add_snapshot(
                    ast_node=kernel.ir_ast,
                    tags=(f_tag_kernel(kname), f_tag_transpiler_module(self)),
                )

        trkr.exit_phase()