import ast
from casys._utils.ast_utils import parse_literal_expr
from casys.dsl._core import casys_ast
from casys.dsl._core.core_macros import MacroSpec, macro_handler
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys, Ir_CaKernel, Ir_SimStepFunc
from casys.dsl.kernel_utils import (
    k_eval
)

@macro_handler(k_eval.__name__)
def mh_k_eval(call: ast.Call, ir: None | Ir_CaSys) -> list[ast.AST]:
    assert ir != None
    args = MacroSpec.parse_and_validate(k_eval, call)
    expr: ast.expr = args['expr']

    node_ir_source = casys_ast.get_meta(call).source_ir
    if node_ir_source is None:
        raise TranspileError("Failed to handle k_eval. Call node does not have ir_source linked", call)

    if not isinstance(node_ir_source, (Ir_CaKernel, Ir_SimStepFunc)):
        raise TranspileError("Failed to handle k_eval. Incompatible environment", call)

    nspace = node_ir_source.base.func.__globals__.copy()
    result = eval(ast.unparse(expr), nspace) # type: ignore

    return [ast.Constant(result)]