from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from typing import Any, Callable
from casys.dsl._core.core_transpiler import Ir_CaSys
from casys.dsl._core.errors import TranspileError

from casys._utils.ast_utils import map_call_args_to_kwargs

import ast
import copy
from typing import Callable

# ———————————————————————————————— #
#  Macro specification components  #
# ———————————————————————————————— #

# Registry of all macros
_MACROS: set[str] = set()

@dataclass
class MacroSpec:
    """Decorator to attach a simple signature spec (required and optional args) to macro dummy functions.

    :param required: tuple of required argument names
    :param optional: tuple of optional argument names
    """
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def __call__[T: Callable](self, fn: T) -> T:
        sig = inspect.signature(fn)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
        fn._spec_required = self.required # type: ignore
        fn._spec_optional = self.optional # type: ignore
        fn._spec_has_kwargs = has_kwargs # type: ignore

        _MACROS.add(fn.__name__)

        if (name:=fn.__name__) not in _MACRO_HANDLERS:
            _MACRO_HANDLERS[name] = get_default_handler(fn)
        return fn

    @classmethod
    def parse_and_validate(cls, fn: Callable[..., Any], call: ast.Call) -> dict[str, ast.expr]:
        """ Parse an AST Call node into a dict mapping argument names to AST expr nodes.  
        And ensure the required arguments have been given.

        :param fn: function previously decorated with @MacroSpec
        :param call: AST Call node
        :return: mapping from argument names to AST expression nodes
        :raises ValueError: if there are missing required args, too many positionals, unexpected or duplicated keywords
        """

        required = getattr(fn, '_spec_required', ())
        optional = getattr(fn, '_spec_optional', ())
        has_kwargs = getattr(fn, '_has_kwargs', False)
        names = list(required) + list(optional)
        args_dict: dict[str, ast.expr] = map_call_args_to_kwargs(call, fn)

        if not has_kwargs:
            for kw in args_dict:
                if kw not in names:
                    raise TranspileError(f"Unexpected keyword argument '{kw!r}'", call)

        # ensure all required args are present
        missing = [n for n in required if n not in args_dict]
        if missing:
            raise TranspileError(f"Missing required arguments '{missing}'", call)

        return args_dict

# ——————————————————————————— #
#  Macro handling components  #
# ——————————————————————————— #

type t_macro_handler = Callable[[ast.Call, None | Ir_CaSys], list[ast.AST]]

# Registry of all macro handlers
_MACRO_HANDLERS: dict[str, t_macro_handler] = {}

def macro_handler(name: str):
    """
    Decorator to register an AST transformer for a kernel-util function.
    """
    def decorator(fn: t_macro_handler) -> t_macro_handler:
        _MACRO_HANDLERS[name] = fn
        return fn
    return decorator

def get_default_handler(fn: Callable[...,Any]) -> t_macro_handler:
    def f(call: ast.Call, ir: None | Ir_CaSys = None) -> list[ast.AST]:
        MacroSpec.parse_and_validate(fn,call)
        return [call]
    setattr(f,'_is_default_handler', True)
    return f