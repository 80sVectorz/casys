from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Callable, cast, get_type_hints
import ast

if TYPE_CHECKING:
    from casys._step_func import KernelCallDescriptor

from casys.dsl._core.debug.ast_origin_tracking import build_origin_map, get_origin_map
from .dsl._core.errors import TranspileError
from ._utils.ast_utils import map_call_args_to_kwargs
from .dsl._core import casys_ast
from ._utils.debug_utils import header
from casys.dsl import kernel_utils

from ._ast_pattern_utils.ast_pattern_engine import Collect, PatternTransformer
from ._ast_pattern_utils.ast_pattern_templates import match_func_call

from .dsl._core.descriptors import CactBufferDescriptor

@dataclass(frozen=True)
class CaKernel:
    func: Callable[..., None]
    func_ast: ast.FunctionDef
    buffers: dict[str, CactBufferDescriptor]
    req_constants: list[str]
    calls: list[KernelCallDescriptor] = field(default_factory=list) # Populated by SimStepFunc

    @classmethod
    def from_func(cls, func: Callable[...,None]) -> CaKernel:
        header(f'[Creating CAKernel: {func.__name__}]')

        hints = get_type_hints(func, include_extras=True)

        buffers = {
            name: CactBufferDescriptor(name,hint.__cac_type__)
            for name,hint in hints.items()
            if hasattr(hint, '__cac_type__')
        }

        src_lines, start_line = inspect.getsourcelines(func)
        src_text = ''.join(src_lines)
        func_ast = cast(ast.FunctionDef, ast.parse(src_text).body[0])

        fname = func.__code__.co_filename or f'<{func.__name__}>'

        local_map = build_origin_map(
            tree=func_ast, 
            filename=fname, 
            source=src_text, 
            line_offset=start_line-1
        )
        get_origin_map().update(local_map.items())     # merge into global singleton

        # -- Handle k_get_const macro --

        req_constants = []

        constant_request_pattern = [
            Collect(
                pattern=match_func_call( kernel_utils.k_get_const),
                key='call'
            )
        ]

        def handle_constant_request(m: dict[str,Any]) -> list[ast.AST]:
            assert 'call' in m and isinstance(m['call'], ast.Call)
            call = m['call']

            args = map_call_args_to_kwargs(call, kernel_utils.k_get_const)
            if not all(k in args for k in ['name','scalar_type']):
                raise TranspileError('k_get_const did not receive required arguments', call)
            
            name = args['name'].value
            req_constants.append(name)
            return [casys_ast.Cs_Constant(name)]

        (tf:=PatternTransformer(
            pattern=constant_request_pattern, 
            actions={'call': handle_constant_request}
        )).visit(func_ast)
        
        return cls(
            func=func,
            func_ast=func_ast,
            buffers=buffers,
            req_constants=req_constants,
        )
    
    def __repr__(self) -> str:
        return f'<CAKernel {self.func.__name__}>'

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        pass


