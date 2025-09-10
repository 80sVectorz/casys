from __future__ import annotations
import copy
from typing import Any

import ast

import numpy as np
from casys._ast_pattern_utils.assign_rewrite_template import AssignLikeRewriter
from casys.dsl._core import casys_ast
from casys.dsl._core.errors import TranspileError
from casys._ast_pattern_utils.ast_pattern_engine import (
    BottomUpPatternTransformer, 
    Collect, 
    NodePattern, 
    Filter,
)
from casys.spec.virtual_types import VirtualBoolField

def vbool_read_handler(m: dict[str, Any]) -> list[ast.AST]:
    ref_node: casys_ast.Cs_SchemaRef[VirtualBoolField] = m['ref']
    subscript_node: ast.Subscript = m['subscript']

    bit_idx = ref_node.s.bit_idx
    soa_field = ref_node.s.bit_plane_soa_field

    assert soa_field is not None

    soa_ref = casys_ast.Cs_SoaFieldRef(soa_field, ref_node.ctx)
    casys_ast.copy_meta(soa_ref, ref_node)

    subscript_node.value = soa_ref

    # x >> {bit_idx} & 1
    new_node = ast.BinOp(
        ast.BinOp(subscript_node, ast.RShift(), ast.Constant(bit_idx)),
        ast.BitAnd(),
        ast.Constant(0b1)
    )

    casys_ast.copy_meta(new_node, subscript_node).evaluates_to_uint_bool = True

    return [new_node]

def snippet_cast_u1(expr: ast.expr, dtype: type[np.generic]) -> ast.Call:
    word_bits = dtype().nbytes * 8

    # Builds: numpy.uint{word_bits}(expr != 0)
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id='numpy', ctx=ast.Load()),
                           attr=f'uint{word_bits}', ctx=ast.Load()),
        args=[ast.Compare(left=expr, ops=[ast.NotEq()], comparators=[ast.Constant(0)])],
        keywords=[]
    )

def transform_vbool_assign_value(subscript_node: ast.Subscript, value: ast.expr) -> ast.BinOp:
        ref_node: casys_ast.Cs_SchemaRef[VirtualBoolField] = subscript_node.value # type: ignore
        bit_idx = ref_node.s.bit_idx
        soa_field = ref_node.s.bit_plane_soa_field

        assert soa_field is not None

        soa_ref = casys_ast.Cs_SoaFieldRef(soa_field, ref_node.ctx)
        casys_ast.copy_meta(soa_ref, ref_node)

        subscript_node.value = soa_ref

        # Validate slice
        slice_node = subscript_node.slice

        if not isinstance(slice_node, ast.Tuple):
            raise TranspileError('Invalid subscript slice for virtual bool field', subscript_node)

        # Ensure the reference value uses the write buffer since a single bit change updates the whole uint
        if not isinstance(slice_node.elts[0], (casys_ast.Cs_RdIdx, casys_ast.Cs_WrIdx)):
            slice_node.elts.insert(0,casys_ast.Cs_WrIdx())
        else:
            slice_node.elts.pop(0)
            slice_node.elts.insert(0,casys_ast.Cs_WrIdx())

        # Transform value
        if isinstance(value,ast.Constant):
            if value.value != 0:
                value.value = 1
            casys_ast.get_meta(value).evaluates_to_uint_bool = True

        if not casys_ast.get_meta(value).evaluates_to_uint_bool:
            u1 = snippet_cast_u1(value, soa_field.data_type)
            casys_ast.copy_meta(u1, value).evaluates_to_uint_bool = True
        else:
            u1 = value

        ref_subscript_node = ast.Subscript(casys_ast.Cs_SoaFieldRef(soa_field, ast.Load()), copy.deepcopy(subscript_node.slice))
        casys_ast.copy_meta(ref_node, subscript_node)

        # x = x & {~(1 << bit_idx)} | {u1} << {bit_idx}
        new_val = ast.BinOp(
            ast.BinOp(ref_subscript_node, ast.BitAnd(), ast.Constant(~(0b1 << bit_idx))),
            ast.BitOr(),
            ast.BinOp(u1, ast.LShift(), ast.Constant(bit_idx))
        )

        casys_ast.copy_meta(new_val, value).evaluates_to_uint_bool = False
        
        return new_val


def vbool_assign_rhs_handler(rhs: ast.expr, *, ctx: ast.AST, meta) -> ast.expr:
    if isinstance(ctx, ast.AugAssign):
        raise TranspileError('Augmented assign not supported for virtual boolean field', ctx)

    if isinstance(ctx, ast.Assign):
        targets = ctx.targets
    elif isinstance(ctx, (ast.AnnAssign, ast.NamedExpr)):
        targets = [ctx.target]
    else:
        return rhs # Not expected

    # Tuple target
    if len(targets) == 1 and isinstance(targets[0], ast.Tuple):
        t_elts = targets[0].elts
        targets_filtered: list[int] = []
        for i,t in enumerate(t_elts):
            if (
                isinstance(t, ast.Subscript)
                and isinstance(t.value, casys_ast.Cs_SchemaRef)
                and t.value.s in meta['schema_refs']
            ):
                targets_filtered.append(i)

        if len(targets_filtered) == 0:
            return rhs 

        if not isinstance(rhs, ast.Tuple):
            raise TranspileError(
                "Couldn't interpret values for virtual boolean field assign operation.",
                ctx,
            )

        v_elts = list(rhs.elts)
        new_vals: list[ast.expr] = []

        for i in range(len(t_elts)):
            if i in targets_filtered:
                new_val = transform_vbool_assign_value(t_elts[i], v_elts[i]) # type: ignore
                new_vals.append(new_val)
            else:
                new_vals.append(v_elts[i])            

        casys_ast.get_meta(ctx).verified_bounds = True
        return ast.Tuple(elts=new_vals, ctx=ast.Load())

    # Single-name target: x = rhs, x += y, x := y, etc.
    if (
        len(targets) == 1
        and isinstance(targets[0], ast.Subscript) 
        and isinstance(targets[0].value, casys_ast.Cs_SchemaRef)
        and targets[0].value in meta['schema_refs']
    ):
        return transform_vbool_assign_value(targets[0], rhs)

    # Any other LHS shape: do nothing
    return rhs

def handle_vbool_field_schema_refs(
    schema_refs: list[casys_ast.Cs_SchemaRef[VirtualBoolField]], 
    ir_ast: ast.AST,
):
    ptrn_read = [
        Collect(
        NodePattern(
            ast.Subscript,
            value = Filter(
                lambda n:
                    isinstance(n, casys_ast.Cs_SchemaRef) and n in schema_refs
                    and isinstance(n.ctx, ast.Load), 
                'ref'
            )
        ),
        'subscript'
        )
    ]
    (tf:=BottomUpPatternTransformer(ptrn_read, {
        'subscript': vbool_read_handler
    })).visit(ir_ast)

    meta: dict[str, Any] = {
        'schema_refs': schema_refs,
    }
    assign_rw = AssignLikeRewriter(rhs_handlers=[vbool_assign_rhs_handler], meta=meta)
    assign_rw.visit(ir_ast)

