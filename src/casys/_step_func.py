from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
from typing import Any, cast, get_type_hints, Callable

from casys.dsl._core import casys_ast
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, TAG_STEP_FUNC
from ._utils.ast_utils import map_call_args_to_kwargs

from ._utils.debug_utils import ast_recursive_dump, preview, header
from ._ast_pattern_utils.ast_pattern_engine import OneOrMore, ZeroOrMore, Bind, Collect, Filter, NodePattern, PatternFinder, PatternTransformer, Pattern
from ._ca_kernel import CaKernel
from casys.dsl._core.debug.ast_origin_tracking import build_origin_map, get_origin_map

from casys.dsl._core.descriptors import CactBufferDescriptor, KernelCallDescriptor
from ._ast_pattern_utils.ast_pattern_templates import match_func_call, match_in_expr

@dataclass(frozen=True)
class SimStepFunc:
    func: Callable
    func_ast: ast.FunctionDef
    buffers: dict[str,CactBufferDescriptor]
    kernels: dict[str,CaKernel]

    @classmethod
    def from_func(cls, func: Callable[...,None]) -> SimStepFunc:
        trkr = get_tracker()
        trkr.enter_phase('Decorating CASysStepFunc')

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
        get_origin_map().update(local_map.items()) # merge into global singleton

        nspace = func.__globals__

        # -- Replace kernel calls with CS_KernelCall --

        kernel_calls_pattern = [
            Collect(
                pattern=match_in_expr(
                    Collect(NodePattern(
                        node_type=ast.Call,
                        func=NodePattern(
                            node_type=ast.Name,
                            id=Filter(lambda n: isinstance(nspace.get(n, None),CaKernel), 'kernel_name')
                        ),
                    ), 'call')),
                key='expr'
            ),
        ]
        
        kernels: dict[str, CaKernel] = {}

        def replace_kernel_call(match: dict[str,Any]) -> list[ast.AST]:
            assert 'call' in match
            assert 'kernel_name' in match

            call: ast.Call = match['call']
            kernel_name: str = match['kernel_name']
            kwargs: dict[str,str] = {k:v.id for k,v in map_call_args_to_kwargs(call, nspace[kernel_name].func).items() if isinstance(v,ast.Name)}

            kernel: CaKernel = nspace[kernel_name]
            for buffer in kernel.buffers:
                if buffer not in kwargs:
                    raise TranspileError(f"Kernel call missing argument: '{buffer}'", match['expr'])

            ca_kernel_obj = nspace[kernel_name]
            desc = KernelCallDescriptor(
                kernel_name,
                {k:v for k,v in kwargs.items()},
            )
            new_node = ast.Expr(casys_ast.Cs_KernelCall(desc)) # type: ignore
            kernels[kernel_name] = nspace[kernel_name]


            kernel.calls.append(desc)
            
            return [new_node]

        PatternTransformer(kernel_calls_pattern, {'expr': replace_kernel_call}).visit(func_ast)

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC,),
            ast_node=func_ast
        )

        header(f'[Finished creating CASysStepFunc: {func.__name__}]\n')

        trkr.exit_phase()

        return cls(
            func,
            func_ast,
            buffers,
            kernels,
        )
        