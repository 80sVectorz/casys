import ast
from casys._utils.ast_utils import parse_literal_expr
from casys.dsl._core import casys_ast
from casys.dsl._core.core_macros import MacroSpec, macro_handler
from casys.dsl.kernel_utils import (
    k_get_pos, k_get_dims, k_get_timestamp,
)

from casys.dsl._core.kernel_values import (
    KV_TIMESTAMP
)

@macro_handler(k_get_timestamp.__name__)
def mh_k_get_timestamp(call: ast.Call, _) -> list[ast.AST]:
    MacroSpec.parse_and_validate(k_get_timestamp, call)

    return [ast.Name(KV_TIMESTAMP)]