import ast
import inspect
from typing import Any, Callable, Type
from typeguard import check_type
import json
from pathlib import Path
from casys.dsl._core.errors import TranspileError

def type_to_ast_node(t: type) -> ast.expr:
    if hasattr(t, '__module__') and hasattr(t, '__name__'):
        return ast.Attribute(
            value=ast.Name(id=t.__module__.split('.')[-1], ctx=ast.Load()),
            attr=t.__name__,
            ctx=ast.Load()
        )
    raise ValueError(f"Cannot convert type {t} to AST")

def map_call_args_to_kwargs(call_node: ast.Call, func_obj: Callable) -> dict[str, Any]:
    """
    Map an AST Call node's arguments to keyword arguments based on the function signature.

    :param call_node: The AST Call node representing the function call.
    :type call_node: ast.Call

    :param func_obj: The function object whose signature is used for mapping arguments.
    :type func_obj: Callable

    :return: A dictionary mapping parameter names to the corresponding AST expression nodes.
    :rtype: dict[str, Any]

    :raises TypeError: If call_node is not an ast.Call node.
    :raises ValueError: If there are too many positional arguments or duplicate keywords.
    """

    if not isinstance(call_node, ast.Call):
        raise TypeError("Expected call_node to be an ast.Call node")
    
    sig = inspect.signature(func_obj)
    params = list(sig.parameters.values())

    # Track where we are in the positional arguments
    mapped_args = {}
    used_kwargs = set()
    pos_index = 0

    # Map positional arguments
    for arg in call_node.args:
        if pos_index >= len(params):
            raise ValueError("Too many positional arguments provided for function")
        
        # Skip parameters already filled by keyword-only params
        while pos_index < len(params) and params[pos_index].kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD
        ):
            pos_index += 1

        if pos_index < len(params):
            param = params[pos_index]
            mapped_args[param.name] = arg
            used_kwargs.add(param.name)
            pos_index += 1
        else:
            break  # No more params to map

    # Map keyword arguments
    for kw in call_node.keywords:
        if kw.arg is None:
            # \**kwargs, skip dynamic unpacking
            continue
        if kw.arg in mapped_args:
            raise ValueError(f"Duplicate argument for parameter '{kw.arg}'")
        mapped_args[kw.arg] = kw.value
        used_kwargs.add(kw.arg)

    # Fill in defaults if needed
    for param in params:
        if param.name not in mapped_args and param.default is not param.empty and param.default is not None:
            mapped_args[param.name] = ast.Constant(value=param.default)

    return mapped_args


def parse_literal_expr(expr: ast.expr, expected_ann: Any, *, transpile_error: bool = True):
    try:
        value = ast.literal_eval(expr)  # Safe: only literal nodes
    except Exception as e:
        msg = f"Not a pure literal: {e}"
        if transpile_error: raise TranspileError(msg, expr)
        raise

    try:
        check_type(value, expected_ann)  # Validates the literal's type based on expected annotation
    except TypeError as e:
        if transpile_error: raise TranspileError(str(e), expr)
        raise
    return value

# def parse_literal_expr[T](expr: ast.expr, expected_type: Type[T], transpile_error: bool = True) -> T:
#     """
#     Parses a literal AST expression into its Python value.
    
#     :param expr: An ast.expr node (e.g., Constant, Tuple, List, etc.)
#     :param expected_type: The expected Python type for validation
#     :return: The literal value, cast to expected_type
#     :raises TypeError: If the AST node is not literal or type doesn't match
#     """

#     try:
#         value = ast.literal_eval(expr)  # safe: only literal nodes
#     except Exception as e:
#         msg = f"Not a pure literal: {e}"
#         if transpile_error: raise TranspileError(msg, expr)
#         raise

#     try:
#         check_type("value", value, expected_ann)  # validates list[int], tuple[int,...], dict[str,float], Union, etc.
#     except TypeError as e:
#         if transpile_error: raise TranspileError(str(e), expr)
#         raise
#     return value
#     if isinstance(expr, ast.Constant):
#         value = expr.value
#     elif isinstance(expr, ast.Tuple):
#         value = tuple(parse_literal_expr(e, expected_type.__args__[0]) if hasattr(expected_type, '__args__') else parse_literal_expr(e, Any) for e in expr.elts)
#     elif isinstance(expr, ast.List):
#         value = [parse_literal_expr(e, expected_type.__args__[0]) if hasattr(expected_type, '__args__') else parse_literal_expr(e, Any) for e in expr.elts]
#     elif isinstance(expr, ast.Dict):
#         key_type, val_type = expected_type.__args__ if hasattr(expected_type, '__args__') else (Any, Any)
#         value = {
#             parse_literal_expr(k, key_type): parse_literal_expr(v, val_type)
#             for k, v in zip(expr.keys, expr.values)
#         }
#     else:
#         raise TypeError(f"Unsupported expression type: {type(expr).__name__}")

#     if not isinstance(value, expected_type):
#         msg = f"Expected type {expected_type.__name__}, got {type(value).__name__}"
#         if transpile_error:
#             raise TranspileError(msg, expr)
#         else:
#             raise TypeError(msg)

#     return value

def ast_to_dict(node) -> dict | list | str | int | float | bool | None:
    """Convert a AST node into a fully JSON-serializable structure."""
    if isinstance(node, ast.AST):
        result = {'_type': node.__class__.__name__}
        for field in node._fields:
            result[field] = ast_to_dict(getattr(node, field))
        return result
    elif isinstance(node, list):
        return [ast_to_dict(el) for el in node]
    elif isinstance(node, (str, int, float, bool)) or node is None:
        return node
    else:
        raise TypeError(f"Unsupported AST value type: {type(node).__name__}")
    

def dump_ast_to_json(path: str | Path, node: ast.AST, indent: int = 2) -> None:
    """Dump a AST to a JSON file as a clean, serializable structure."""
    as_dict = ast_to_dict(node)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(as_dict, f, indent=indent, ensure_ascii=False)