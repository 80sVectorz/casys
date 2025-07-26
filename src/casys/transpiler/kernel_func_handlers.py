import ast
import copy
from typing import Callable

from .kernel_values import KV_HEIGHT, KV_PX, KV_PY, KV_TIMESTAMP, KV_WIDTH, KV_WR_IDX
from .kernel_ast import K_META, KernelASTNodeMeta, set_meta

from .kernel_utils import (
    Spec, k_get_pos, k_get_dims, k_get_const,
    k_get_timestamp, k_get_wr_idx, k_patch_op,
    k_neighbor_mask
)

# —————————————————————————————————————————————————————————————————————————————
# Registry of all k-func handlers
# —————————————————————————————————————————————————————————————————————————————
_KFUNC_HANDLERS: dict[str, Callable[[ast.Call], ast.AST]] = {}

def kfunc_handler(name: str):
    """
    Decorator to register an AST transformer for a kernel-util function.
    """
    def decorator(fn: Callable[[ast.Call], ast.AST]) -> Callable[[ast.Call], ast.AST]:
        _KFUNC_HANDLERS[name] = fn
        return fn
    return decorator


# —————————————————————————————————————————————————————————————————————————————
# kernel_utils handlers
# —————————————————————————————————————————————————————————————————————————————

@kfunc_handler(k_get_pos.__name__)
def _handle_get_pos(call: ast.Call) -> ast.Tuple:
    # validate no arguments passed
    Spec.parse(k_get_pos, call)
    # k_get_pos() -> (kval_x, kval_y)
    return ast.Tuple(
        elts=[
            ast.Name(id=KV_PX, ctx=ast.Load()),
            ast.Name(id=KV_PY, ctx=ast.Load()),
        ],
        ctx=ast.Load()
    )

@kfunc_handler(k_get_timestamp.__name__)
def _handle_get_timestamp(call: ast.Call) -> ast.Name:
    # validate no arguments passed
    Spec.parse(k_get_timestamp, call)
    # k_get_pos() -> kval_timestamp
    return ast.Name(id=KV_TIMESTAMP, ctx=ast.Load())

@kfunc_handler(k_get_dims.__name__)
def _handle_get_dims(call: ast.Call) -> ast.Tuple:
    Spec.parse(k_get_dims, call)
    # k_get_dims() -> (kval_w, kval_h)
    return ast.Tuple(
        elts=[
            ast.Name(id=KV_WIDTH, ctx=ast.Load()),
            ast.Name(id=KV_HEIGHT, ctx=ast.Load()),
        ],
        ctx=ast.Load()
    )


@kfunc_handler(k_get_wr_idx.__name__)
def _handle_get_wr_idx(call: ast.Call) -> ast.Name:
    # validate no arguments passed
    Spec.parse(k_get_wr_idx, call)
    # k_get_pos() -> kval_wr_idx
    return ast.Name(id=KV_WR_IDX, ctx=ast.Load())


@kfunc_handler(k_get_const.__name__)
def _handle_get_const(call: ast.Call) -> ast.Name:
    args = Spec.parse(k_get_const, call)
    # k_get_const(type, "NAME") -> NAME
    name_arg = args['name']
    assert isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)
    return ast.Name(id=name_arg.value, ctx=ast.Load())


@kfunc_handler(k_patch_op.__name__)
def _handle_patch_op(call: ast.Call) -> ast.AST:
    """
    Inlines k_patch_op(...) -> a big BinOp over a buffer field over the mask.
    """
    args_dict = Spec.parse(k_patch_op, call)

    # op must be a string constant
    op_const = args_dict['op']
    if not isinstance(op_const, ast.Constant) or not isinstance(op_const.value, str):
        raise ValueError('k_patch_op requires a string constant for "op"')
    op = op_const.value
    if op not in ('sum', 'mean', 'product', 'bit_or'):
        raise ValueError(f'Unsupported op {op!r}')

    # width/height must be int constants
    w_node, h_node = args_dict['width'], args_dict['height']
    if not (isinstance(w_node, ast.Constant) and isinstance(w_node.value, int)):
        raise ValueError('k_patch_op width must be an int constant')
    if not (isinstance(h_node, ast.Constant) and isinstance(h_node.value, int)):
        raise ValueError('k_patch_op height must be an int constant')
    w, h = w_node.value, h_node.value

    # mask: optional list[list[0|1]]
    if 'mask' in args_dict:
        mask_ast = args_dict['mask']
        if not isinstance(mask_ast, ast.List) or len(mask_ast.elts) != h:
            raise ValueError('Bad mask shape')
        mask: list[list[int]] = []
        for row in mask_ast.elts:
            if not isinstance(row, ast.List) or len(row.elts) != w:
                raise ValueError('Bad mask shape')
            row_vals: list[int] = []
            for b in row.elts:
                if not (isinstance(b, ast.Constant) and b.value in (0, 1)):
                    raise ValueError('Mask must be 0 or 1')
                row_vals.append(int(b.value))
            mask.append(row_vals)
    else:
        mask = [[1] * w for _ in range(h)]

    buf = args_dict['buffer']
    x_ast, y_ast = args_dict['x'], args_dict['y']

    # flatten coordinates with wrap
    raw_coords: list[ast.expr] = []
    mod_coords: list[ast.expr] = []

    for i in range(h):
        for j in range(w):
            if mask[i][j] == 1:
                ox,oy = (
                    j - w // 2,
                    i - h // 2
                )
                ast_x_op, ast_y_op = (
                    ast.BinOp(left=x_ast, op=ast.Add(), right=ast.Constant(j - w // 2)),
                    ast.BinOp(left=y_ast, op=ast.Add(), right=ast.Constant(i - h // 2)),
                )

                raw_coords.append(
                    raw_c := ast.Subscript(
                        value=buf,
                        slice=ast.Tuple(elts=[
                            ast_x_op if ox != 0 else x_ast,
                            ast_y_op if oy != 0 else y_ast,
                        ]),
                        ctx=ast.Load()
                    )
                )

                mod_coords.append(
                    mod_c := ast.Subscript(
                        value=buf,
                        slice=ast.Tuple(elts=[
                            ast.BinOp(
                                left=ast_x_op,
                                op=ast.Mod(),
                                right=ast.Name(KV_WIDTH, ctx=ast.Load())
                            ) if ox != 0 else x_ast,
                            ast.BinOp(
                                left=ast_y_op,
                                op=ast.Mod(),
                                right=ast.Name(KV_HEIGHT, ctx=ast.Load())
                            ) if oy != 0 else y_ast,
                        ]),
                        ctx=ast.Load()
                    )
                )

                set_meta(raw_c, KernelASTNodeMeta(verified_bounds=True))
                set_meta(mod_c, KernelASTNodeMeta(verified_bounds=True))

    # pick op AST
    op_map = {'sum': ast.Add(), 'mean': ast.Add(), 'product': ast.Mult(), 'bit_or': ast.BitOr()}
    binop = op_map[op]
    # build a left-deep BinOp tree

    ast_raw_sum = ast.BinOp(left=raw_coords.pop(), op=binop, right=raw_coords.pop())
    ast_mod_sum = ast.BinOp(left=mod_coords.pop(), op=binop, right=mod_coords.pop())
    for raw_c,mod_c in zip(raw_coords,mod_coords):
        ast_raw_sum = ast.BinOp(left=ast_raw_sum, op=binop, right=raw_c)
        ast_mod_sum = ast.BinOp(left=ast_mod_sum, op=binop, right=mod_c)


    node = ast.IfExp(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=ast.Constant(w//2),
                    ops=[ast.Lt(),ast.Lt()],
                    comparators=[
                        x_ast,
                        ast.BinOp(ast.Name(KV_WIDTH),ast.Sub(), ast.Constant(w//2))
                    ]
                ),
                ast.Compare(
                    left=ast.Constant(h//2),
                    ops=[ast.Lt(),ast.Lt()],
                    comparators=[
                        y_ast,
                        ast.BinOp(ast.Name(KV_HEIGHT),ast.Sub(), ast.Constant(h//2))
                    ]
                )
            ]
        ),
        body=ast_raw_sum,
        orelse=ast_mod_sum,
    )

    # if mean, divide by total count
    if op == 'mean':
        total = sum(sum(mask, []))
        node = ast.BinOp(left=node, op=ast.Div(), right=ast.Constant(total))

    return node


class NameReplacer(ast.NodeTransformer):
    """Replace Name nodes according to a mapping."""
    def __init__(self, mapping: dict[str, ast.expr]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return self.mapping.get(node.id, node)

@kfunc_handler(k_neighbor_mask.__name__)
def _handle_neighbor_mask(call: ast.Call) -> ast.AST:
    """
    Inlines k_neighbor_mask(buffer, x, y, d=…) into an 8-bit mask
    by evaluating the user's Boolean `expr` expression at each Moore neighbor.
    Marks every Subscript in those expressions as verified_bounds.
    """
    args = Spec.parse(k_neighbor_mask, call)
    test_expr = args['expr']
    x_ast = args['x']
    y_ast = args['y']

    # figure out what name the user bound d to (via walrus or simple name)
    d_arg = args.get('d')
    if isinstance(d_arg, ast.NamedExpr) and isinstance(d_arg.target, ast.Name):
        d_name = d_arg.target.id
    elif isinstance(d_arg, ast.Name):
        d_name = d_arg.id
    else:
        d_name = 'd'

    # (dx, dy, bit) for the 8 Moore neighbors
    neighbors: list[tuple[int,int,int]] = [
        ( 0,  1, 0),  # N
        ( 1,  1, 1),  # NE
        ( 1,  0, 2),  # E
        ( 1, -1, 3),  # SE
        ( 0, -1, 4),  # S
        (-1, -1, 5),  # SW
        (-1,  0, 6),  # W
        (-1,  1, 7),  # NW
    ]

    raw_bits: list[ast.expr] = []
    mod_bits: list[ast.expr] = []

    for idx, (dx, dy, bit) in enumerate(neighbors):
        # raw coords (no wrap)
        x_raw = ast.BinOp(left=x_ast, op=ast.Add(), right=ast.Constant(dx))
        y_raw = ast.BinOp(left=y_ast, op=ast.Add(), right=ast.Constant(dy))
        # wrapped coords
        x_mod = ast.BinOp(left=x_raw, op=ast.Mod(),
                          right=ast.Name(id=KV_WIDTH, ctx=ast.Load()))
        y_mod = ast.BinOp(left=y_raw, op=ast.Mod(),
                          right=ast.Name(id=KV_HEIGHT, ctx=ast.Load()))

        # 1) build & mark the raw‐test
        raw_map = {'x': x_raw, 'y': y_raw, d_name: ast.Constant(0b1 << bit)}
        raw_test = NameReplacer(raw_map).visit(copy.deepcopy(test_expr))
        # mark every Subscript as verified
        for sub in ast.walk(raw_test):
            if isinstance(sub, ast.Subscript):
                set_meta(sub, KernelASTNodeMeta(verified_bounds=True))
        raw_bits.append(
            ast.Call(
                ast.Attribute(ast.Name('numpy'),'uint8'),
                [ast.BinOp(left=raw_test, op=ast.LShift(), right=ast.Constant(bit))]
            )
        )

        # 2) build & mark the mod‐test
        mod_map = {'x': x_mod, 'y': y_mod, d_name: ast.Constant(0b1 << bit)}
        mod_test = NameReplacer(mod_map).visit(copy.deepcopy(test_expr))
        for sub in ast.walk(mod_test):
            if isinstance(sub, ast.Subscript):
                set_meta(sub, KernelASTNodeMeta(verified_bounds=True))
        mod_bits.append(
            ast.Call(
                ast.Attribute(ast.Name('numpy'),'uint8'),
                [ast.BinOp(left=mod_test, op=ast.LShift(), right=ast.Constant(bit))]
            )
        )

    # OR-reduce the raw bits into one expr
    raw_mask = raw_bits.pop()
    for b in raw_bits:
        raw_mask = ast.BinOp(left=raw_mask, op=ast.BitOr(), right=b)

    # OR-reduce the mod bits
    mod_mask = mod_bits.pop()
    for b in mod_bits:
        mod_mask = ast.BinOp(left=mod_mask, op=ast.BitOr(), right=b)

    # interior test: 1 <= x < width-1  and 1 <= y < height-1
    interior = ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Compare(
                left=ast.Constant(1),
                ops=[ast.LtE(), ast.Lt()],
                comparators=[
                    x_ast,
                    ast.BinOp(
                        left=ast.Name(id=KV_WIDTH, ctx=ast.Load()),
                        op=ast.Sub(),
                        right=ast.Constant(1)
                    )
                ]
            ),
            ast.Compare(
                left=ast.Constant(1),
                ops=[ast.LtE(), ast.Lt()],
                comparators=[
                    y_ast,
                    ast.BinOp(
                        left=ast.Name(id=KV_HEIGHT, ctx=ast.Load()),
                        op=ast.Sub(),
                        right=ast.Constant(1)
                    )
                ]
            ),
        ]
    )

    # choose raw_mask on interior, else mod_mask
    mask_expr = ast.IfExp(test=interior, body=raw_mask, orelse=mod_mask)

    # final cast to uint8
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                           attr='uint8', ctx=ast.Load()),
        args=[mask_expr],
        keywords=[]
    )

# —————————————————————————————————————————————————————————————————————————————
# The dispatcher transformer
# —————————————————————————————————————————————————————————————————————————————
class NT_KFuncParser(ast.NodeTransformer):
    """
    Replaces calls to k_get_pos, k_get_dims, k_get_const, k_patch_op
    by looking up the handlers in our registry.
    """
    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name):
            handler = _KFUNC_HANDLERS.get(node.func.id)
            if handler:
                return handler(node)
        return self.generic_visit(node)