import ast
import inspect
from typing import cast
from casys._utils.ast_utils import map_call_args_to_kwargs
from casys._ast_pattern_utils.ast_pattern_engine import Bind, Collect, Filter, NodePattern, PatternTransformer
from casys.dsl._core import casys_ast
from casys.dsl._core.core_macros import MacroSpec, macro_handler
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys, Ir_CaKernel, Ir_SimStepFunc
from casys.dsl.kernel_utils import (
    k_snippet
)

@macro_handler(k_snippet.__name__)
def mh_k_snippet(call: ast.Call, ir: None | Ir_CaSys) -> list[ast.AST]:
    assert ir != None

    args = MacroSpec.parse_and_validate(k_snippet, call)
    snippet_call_expr: ast.expr = args['snippet_call']

    node_ir_source = casys_ast.get_meta(call).source_ir
    if not isinstance(node_ir_source, (Ir_CaKernel, Ir_SimStepFunc)):
        raise TranspileError(f"Failed to handle {k_snippet.__name__}. Incompatible environment", call)

    call_match = NodePattern(ast.Call,func=NodePattern(ast.Name,id=Bind('snippet_func_name'))).match(snippet_call_expr)

    if call_match is None or 'snippet_func_name' not in call_match:
        raise TranspileError(f"Failed to handle {k_snippet.__name__}. Couldn't infer snippet function", call)

    snippet_call: ast.Call = cast(ast.Call,snippet_call_expr)
    snippet_func = node_ir_source.base.func.__globals__.get(call_match['snippet_func_name'])
    if snippet_func is not None and callable(snippet_func):
        snippet_call_kwargs = map_call_args_to_kwargs(snippet_call, snippet_func)

        snippet_func_src = inspect.getsource(snippet_func)
        snippet_func_ast: ast.FunctionDef = cast(ast.FunctionDef, ast.parse(snippet_func_src).body[0])

        ptrn_find_and_replace = [Collect(NodePattern(ast.Name, id=Filter(lambda n: n in snippet_call_kwargs)), 'node')]

        PatternTransformer(ptrn_find_and_replace, {
            'node': lambda m: [snippet_call_kwargs[m['node'].id]]
        }).visit(snippet_func_ast)

        return [*snippet_func_ast.body]
    else:
        raise TranspileError(f"Failed to handle {k_snippet.__name__}. Couldn't find snippet function in the kernel's namespace", call)