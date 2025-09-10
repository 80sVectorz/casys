from __future__ import annotations

from casys._utils.debug_utils import ast_recursive_dump

"""casys debug helpers: AST -> python source + unified‑diff

This module provides two public helpers:

* ``ast_to_source`` – Safely converts an :pyclass:`ast.AST` instance back to
  Python source.  *Non‑builtin* (``casys_ast``) nodes are replaced with
  clearly‑identifiable placeholders so that ``ast.unparse`` can succeed.
* ``diff_ast`` – Returns a *unified diff* between two ASTs by comparing their
  ``ast_to_source`` output.

The placeholders let humans eyeball where custom IR constructs were inserted
without wading through the full *repr* of every node.  Combined with
``diff_ast`` you can swap the heavyweight *full‑snapshot* JSON currently used
by ``ast_timeline_tracking`` for a much lighter, line‑oriented diff.
"""

import copy
import ast

# ---------------- #
# Helper utilities #
# ---------------- #

def _placeholder_str(node: ast.AST) -> str:
    """Return a str ast dump, human-friendly placeholder for *node*."""

    return '<<'+ast_recursive_dump(node).replace("'", "\'").replace('"','\"').replace("\n", "").replace("    ","")+'>>'

def _make_placeholder(node: ast.AST) -> ast.AST:
    """Convert *node* to a valid AST subtree that unparses as a placeholder."""
    placeholder = ast.Constant(value=_placeholder_str(node))

    # Expression context is easy - replace directly.
    if isinstance(node, ast.expr):
        return ast.copy_location(placeholder, node)

    # Statement context - wrap in an Expr so we stay syntactically valid.
    return ast.copy_location(ast.Expr(value=placeholder), node)


class _ReplaceCustomNodes(ast.NodeTransformer):
    """Swap out non-builtin nodes for string placeholders."""

    def generic_visit(self, node: ast.AST):
        # Only *traverse* genuine builtin AST nodes.  Any subclass from a
        # different module (e.g. casys_ast) gets replaced.
        if node.__class__.__module__ != 'ast':
            return ast.fix_missing_locations(_make_placeholder(node))
        return super().generic_visit(node)


# ——————————————
# Public API
# ——————————————

def ast_to_source(tree: ast.AST) -> str:
    """Return *tree* as pretty-printed Python source.

    ``casys_ast`` nodes are rendered as placeholders so the output is always
    valid, unparsable code.  Location info is preserved where possible to keep
    diffs readable.
    """
    clean = _ReplaceCustomNodes().visit(ast.fix_missing_locations(copy.deepcopy(tree)))  # type: ignore[attr-defined]
    return ast.unparse(clean)