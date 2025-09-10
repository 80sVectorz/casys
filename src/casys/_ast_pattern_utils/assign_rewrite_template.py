from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Callable

import ast

from casys.dsl._core import casys_ast

@dataclass
class SliceEdit:
    """Represents an edit to an individual slice expression."""
    index: int
    new_expr: ast.expr


@dataclass
class LhsEdit:
    """Represents an edit applied to a LHS target node.

    If `replace_target` is set, the whole LHS node is replaced.
    If `slice_edits` is provided and the LHS is a Subscript with a Tuple slice,
    those indexed slice elements are replaced.
    """
    replace_target: ast.expr | None = None
    slice_edits: list[SliceEdit] | None = None


class RhsHandler(Protocol):
    def __call__(self, rhs: ast.expr, *, ctx: ast.AST, meta: dict[str, Any]) -> ast.expr:
        """Transform an RHS expression.

        Args:
            rhs: The original RHS expression.
            ctx: The AST node where rhs occurs (Assign, AugAssign normalized, etc.).
            meta: Arbitrary metadata bag from the caller.

        Returns:
            The transformed RHS expression.
        """
        ...


class LhsHandler(Protocol):
    def __call__(self, lhs: ast.expr, *, ctx: ast.AST, meta: dict[str, Any]) -> LhsEdit | None:
        """Transform a LHS target expression.

        Args:
            lhs: The original LHS target node (Name, Subscript, Attribute, Tuple, etc.).
            ctx: The assign-ish node where the target occurs.
            meta: Arbitrary metadata bag from the caller.

        Returns:
            A LhsEdit if an edit should be applied, otherwise None.
        """


@dataclass
class AssignLike:
    """Unified view over assign-like nodes.

    After normalization:
    - `targets` is a list of LHS expressions (single target or tuple-unpack).
    - `value` is the RHS expression.
    - `node` is the original AST node being rewritten.
    """
    node: ast.AST
    targets: list[ast.expr]
    value: ast.expr


def _normalize_assign_like(node: ast.AST) -> AssignLike | None:
    """Produce a unified AssignLike view for Assign, AugAssign, AnnAssign, NamedExpr.

    - AugAssign(target, op, value) becomes Assign([target], BinOp(Name(target), op, value)).
    - AnnAssign is treated as Assign when `value` exists.
    - NamedExpr(target, value) becomes Assign([target], value) for transformation purposes.
    """

    match node:
        case ast.Assign():
            # Multiple targets possible: a = b = c, or tuple targets
            return AssignLike(node=node, targets=list(node.targets), value=node.value)

        case ast.AugAssign():
            # x += y -> x = x + y
            binop = ast.BinOp(left=ast.copy_location(ast.copy_location(node.target, node.target), node),
                            op=node.op,
                            right=node.value)
            return AssignLike(node=node, targets=[node.target], value=binop)

        case ast.AnnAssign():
            if node.value is None: return None
            return AssignLike(node=node, targets=[node.target], value=node.value)

        case ast.NamedExpr():
            return AssignLike(node=node, targets=[node.target], value=node.value)

    return None


def _apply_lhs_edit(lhs: ast.expr, edit: LhsEdit) -> ast.expr:
    if edit.replace_target is not None:
        return edit.replace_target

    if edit.slice_edits and isinstance(lhs, ast.Subscript) and isinstance(lhs.slice, ast.Tuple):
        new_elts = list(lhs.slice.elts)
        for se in edit.slice_edits:
            new_elts[se.index] = se.new_expr
        new_slice = ast.Tuple(elts=new_elts, ctx=lhs.slice.ctx)
        return ast.Subscript(value=lhs.value, slice=new_slice, ctx=lhs.ctx)

    return lhs


def _rewrite_assign_like(
    al: AssignLike,
    *,
    lhs_handlers: list[LhsHandler],
    rhs_handlers: list[RhsHandler],
    meta: dict[str, Any],
) -> ast.AST:
    """Apply handlers and rebuild a valid AST node with transformed LHS and RHS."""

    # LHS pass
    new_targets: list[ast.expr] = []
    for t in al.targets:
        edited = False
        cur = t
        for h in lhs_handlers:
            out = h(cur, ctx=al.node, meta=meta)
            if out:
                cur = _apply_lhs_edit(cur, out)
                edited = True
        new_targets.append(cur if edited else t)

    # RHS pass
    rhs = al.value
    for rh in rhs_handlers:
        rhs = rh(rhs, ctx=al.node, meta=meta)

    # Rebuild node, preserving original node type and metadata.
    if isinstance(al.node, ast.Assign):
        new_node = ast.Assign(targets=new_targets, value=rhs, type_comment=al.node.type_comment)
        casys_ast.copy_meta(new_node, al.node)
        return new_node

    if isinstance(al.node, ast.AugAssign):
        # We normalized AugAssign to a BinOp RHS already; emit plain Assign.
        new_node = ast.Assign(targets=[new_targets[0]], value=rhs)
        casys_ast.copy_meta(new_node, al.node)
        return new_node

    if isinstance(al.node, ast.AnnAssign):
        new_node = ast.AnnAssign(
            target=new_targets[0],
            annotation=al.node.annotation,
            value=rhs,
            simple=al.node.simple,
        )
        casys_ast.copy_meta(new_node, al.node)
        return new_node

    if isinstance(al.node, ast.NamedExpr):
        # Re-emit as NamedExpr for fidelity if no structure changed on LHS.
        # If LHS changed shape, fall back to Assign for validity.

        if len(new_targets) == 1 and isinstance(new_targets[0], type(al.node.target)):
            new_node = ast.NamedExpr(target=new_targets[0], value=rhs)
        else:
            new_node = ast.Assign(targets=new_targets, value=rhs)
        casys_ast.copy_meta(new_node, al.node)
        return new_node

    return al.node


class AssignLikeRewriter(ast.NodeTransformer):
    """Reusable helper that allows for easy transformation of any assign like nodes."""

    def __init__(
        self,
        *,
        lhs_handlers: Sequence[LhsHandler] = (),
        rhs_handlers: Sequence[RhsHandler] = (),
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._lhs_handlers = list(lhs_handlers)
        self._rhs_handlers = list(rhs_handlers)
        self._meta = meta or {}

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        al = _normalize_assign_like(node)
        assert al is not None
        return _rewrite_assign_like(al, lhs_handlers=self._lhs_handlers, rhs_handlers=self._rhs_handlers, meta=self._meta)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        al = _normalize_assign_like(node)
        assert al is not None
        return _rewrite_assign_like(al, lhs_handlers=self._lhs_handlers, rhs_handlers=self._rhs_handlers, meta=self._meta)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None:
            return self.generic_visit(node)
        al = _normalize_assign_like(node)
        assert al is not None
        return _rewrite_assign_like(al, lhs_handlers=self._lhs_handlers, rhs_handlers=self._rhs_handlers, meta=self._meta)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        al = _normalize_assign_like(node)
        assert al is not None
        return _rewrite_assign_like(al, lhs_handlers=self._lhs_handlers, rhs_handlers=self._rhs_handlers, meta=self._meta)
