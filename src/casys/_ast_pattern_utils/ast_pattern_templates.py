from .ast_pattern_engine import *
from typing import Callable

def match_in_expr(pattern: Pattern) -> NodePattern:
    return NodePattern(
        node_type=ast.Expr,
        value=pattern,
    )

def match_func_call(func: Callable, **kwargs) -> NodePattern:
    return NodePattern(
        node_type=ast.Call,
        func=NodePattern(
            node_type=ast.Name, 
            id=Filter(lambda n: n == func.__name__),
            **kwargs
        )
    )