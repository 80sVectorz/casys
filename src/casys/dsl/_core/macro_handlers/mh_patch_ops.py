import ast
import copy
from logging import Filter
from typing import Sequence

from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, NodePattern, Filter

from casys._utils.ast_utils import parse_literal_expr
from casys.dsl._core import casys_ast
from casys.dsl._core.core_macros import MacroSpec, macro_handler
from casys.dsl._core.errors import TranspileError
from casys.dsl.kernel_utils import (
    k_patch_op, k_neighbor_mask
)

@macro_handler(k_patch_op.__name__)
def mh_k_patch_op(call: ast.Call, _) -> list[ast.AST]:
    args_dict = MacroSpec.parse_and_validate(k_patch_op, call)

    op_map = {
        'sum': ast.Add(), 
        'mean': ast.Add(), 
        'product': ast.Mult(), 
        'bit_or': ast.BitOr(),

        'logical_or': ast.Or(),
        'logical_and': ast.And(),
    }

    # op must be a string constant
    op = parse_literal_expr(args_dict['op'], str)
    if op not in op_map:
        raise TranspileError(f'Unsupported op {op!r}', args_dict['op'])

    # width/height must be int constants
    w, h = (
        parse_literal_expr(args_dict['width'], int),
        parse_literal_expr(args_dict['height'], int),
    )

    weights = (
        parse_literal_expr(args_dict['weights'], Sequence[Sequence[int | float]])
        if args_dict.get('weights') else
        [[1 if (y!=h//2 or x!=w//2) else 0 for x in range(w)] for y in range(h)]
    )

    # weights: optional list[list[int | float]]
    if 'weights' in args_dict:
        if len(weights) != h or any(len(row)!=w for row in weights):
            raise TranspileError('Bad weights shape', call)

    buf = args_dict['buffer']
    x_ast, y_ast = args_dict['x'], args_dict['y']

    # flatten coordinates with wrap
    raw_coords: list[ast.expr] = []
    mod_coords: list[ast.expr] = []
    coord_weights: list[int | float] = []

    for i in range(h):
        for j in range(w):
            if (weight:=weights[i][j]) != 0:
                coord_weights.append(weight)

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
                                right=casys_ast.Cs_AxisSize(0)
                            ) if ox != 0 else x_ast,
                            ast.BinOp(
                                left=ast_y_op,
                                op=ast.Mod(),
                                right=casys_ast.Cs_AxisSize(1)
                            ) if oy != 0 else y_ast,
                        ]),
                        ctx=ast.Load()
                    )
                )

                casys_ast.get_meta(raw_c).verified_bounds = True

    # pick op AST
    binop = op_map[op]

    if op not in ('logical_or', 'logical_and'):
        # Build weighted term lists so weights align with coords
        terms_raw: list[ast.expr] = []
        terms_mod: list[ast.expr] = []
        use_weight = op in ('sum', 'mean')

        for raw_c, mod_c, weight in zip(raw_coords, mod_coords, coord_weights):
            if use_weight and weight != 1:
                wconst = ast.Constant(weight)
                terms_raw.append(ast.BinOp(left=raw_c, op=ast.Mult(), right=wconst))
                terms_mod.append(ast.BinOp(left=mod_c, op=ast.Mult(), right=wconst))
            else:
                terms_raw.append(raw_c)
                terms_mod.append(mod_c)

        def _reduce(exprs: list[ast.expr]) -> ast.expr:
            # Assume at least 1 term is present for normal use; handle edge cases anyway.
            if not exprs:
                # Neutral elements per op
                return ast.Constant(0 if binop.__class__ is ast.Add else (1 if binop.__class__ is ast.Mult else 0))
            node = exprs[0]
            for e in exprs[1:]:
                node = ast.BinOp(left=node, op=binop, right=e)
            return node

        ast_raw_sum = _reduce(terms_raw)
        ast_mod_sum = _reduce(terms_mod)
    else:
        ast_raw_sum = ast.BoolOp(binop, [raw_c for raw_c in raw_coords])
        ast_mod_sum = ast.BoolOp(binop, [mod_c for mod_c in mod_coords])

    rx = int(w // 2)
    ry = int(h // 2)

    node = ast.IfExp(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=ast.Constant(rx),
                    ops=[ast.Lt(),ast.Lt()],
                    comparators=[
                        x_ast,
                        ast.BinOp(casys_ast.Cs_AxisSize(0),ast.Sub(), ast.Constant(rx))
                    ]
                ),
                ast.Compare(
                    left=ast.Constant(ry),
                    ops=[ast.Lt(),ast.Lt()],
                    comparators=[
                        y_ast,
                        ast.BinOp(casys_ast.Cs_AxisSize(1),ast.Sub(), ast.Constant(ry))
                    ]
                )
            ]
        ),
        body=ast_raw_sum,
        orelse=ast_mod_sum,
    )

    # if mean, divide by total count
    if op == 'mean':
        total = len(coord_weights)
        node = ast.BinOp(left=node, op=ast.Div(), right=ast.Constant(total))

    return [node]


@macro_handler(k_neighbor_mask.__name__)
def mh_k_neighbor_mask(call: ast.Call, _) -> list[ast.AST]:
    """
    Inlines `k_neighbor_mask(buffer, x, y, d=...)` into an 8-bit mask
    by evaluating the user's Boolean `expr` expression at each Moore neighbor.
    """
    args = MacroSpec.parse_and_validate(k_neighbor_mask, call)

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
        x_raw = ast.BinOp(left=x_ast, op=ast.Add(), right=ast.Constant(dx)) if dx != 0 else x_ast
        y_raw = ast.BinOp(left=y_ast, op=ast.Add(), right=ast.Constant(dy)) if dy != 0 else y_ast
        # wrapped coords
        x_mod = ast.BinOp(left=x_raw, op=ast.Mod(),
                          right=casys_ast.Cs_AxisSize(0))
        y_mod = ast.BinOp(left=y_raw, op=ast.Mod(),
                          right=casys_ast.Cs_AxisSize(1))

        # build & mark the raw‐test
        raw_map = {'x': x_raw, 'y': y_raw, d_name: ast.Constant(0b1 << bit)}
        ptrn_replace_names = [Collect(NodePattern(ast.Name, id=Filter(lambda n: n in raw_map)), 'name')]
        raw_test = copy.deepcopy(test_expr)
        (tf:=PatternTransformer(ptrn_replace_names, {'name': lambda m: [raw_map[m['name'].id]]})).visit(raw_test)

        # mark every Subscript as verified
        for sub in ast.walk(raw_test):
            if isinstance(sub, ast.Subscript):
                casys_ast.get_meta(sub).verified_bounds = True

        raw_bits.append(
            ast.Call(
                ast.Attribute(ast.Name('numpy'),'uint8'),
                [ast.BinOp(left=raw_test, op=ast.LShift(), right=ast.Constant(bit))]
            )
        )

        # build & mark the mod‐test
        mod_map = {'x': x_mod, 'y': y_mod, d_name: ast.Constant(0b1 << bit)}

        mod_test = copy.deepcopy(test_expr)
        (tf:=PatternTransformer(ptrn_replace_names, {'name': lambda m: [mod_map[m['name'].id]]})).visit(mod_test)

        for sub in ast.walk(mod_test):
            if isinstance(sub, ast.Subscript):
                casys_ast.get_meta(sub).verified_bounds = True

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
                        left=casys_ast.Cs_AxisSize(0),
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
                        left=casys_ast.Cs_AxisSize(1),
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
    return [ast.Call(
        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                           attr='uint8', ctx=ast.Load()),
        args=[mask_expr],
        keywords=[]
    )]