from __future__ import annotations

import ast
from collections import defaultdict
import copy
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, cast, get_type_hints, Callable

if TYPE_CHECKING:
    from casys.spec.cac_type import CaCellTypeSpec

from casys.dsl._core.schema.base_components import GroupSchema

from casys.dsl._core.schema.world_schema import WorldSchema
from casys.spec.ca_kernel import CaKernel
from casys.dsl._core import casys_ast
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, TAG_STEP_FUNC
from casys._utils.ast_utils import map_call_args_to_kwargs

from casys._utils.debug_utils import header
from casys._ast_pattern_utils.ast_pattern_engine import Collect, Filter, NodePattern, PatternTransformer
from casys.dsl._core.debug.ast_origin_tracking import build_origin_map, get_origin_map

from casys.dsl._core.descriptors import KernelCallDescriptor
from casys._ast_pattern_utils.ast_pattern_templates import match_in_expr

from casys.spec.world_spec import get_bound_world_info, LayerSymbol

@dataclass(frozen=True)
class SimStepFunc:
    func: Callable
    func_ast: ast.FunctionDef
    world_schema: WorldSchema
    kernels: dict[str,CaKernel]
    kcall_permutations: dict[str, dict[str,list[GroupSchema]]]

    @classmethod
    def from_func(cls, fn: Callable[...,None]) -> SimStepFunc:
        trkr = get_tracker()
        trkr.enter_phase('Decorating SimStepFunc')

        bound = get_bound_world_info(fn)

        if bound is not None:
            # Build groups using the GLOBAL symbol names captured by the decorator
            # Each LayerSymbol carries its CAC type via .cac_type
            groups: dict[str, GroupSchema] = {}
            for sym_name, sym in bound.symbol_names.items():
                cact_spec: CaCellTypeSpec = getattr(sym.cac_type, '__cac_type__')
                fields_copy = copy.deepcopy(cact_spec.schema.fields)
                group_schema = GroupSchema(sym_name, fields_copy)
                groups[sym_name] = group_schema

                sym.bind_schema(group_schema)

            world_schema = WorldSchema(groups)
        
        else:
            # Build groups using function argument and argument type hints.
            hints = get_type_hints(fn, include_extras=True)

            groups: dict[str,GroupSchema] = {
                name: GroupSchema(name, copy.deepcopy(getattr(hint, '__cac_type__').schema.fields))
                for name,hint in hints.items()
                if hasattr(hint, '__cac_type__')
            }
            world_schema = WorldSchema(groups)

        src_lines, start_line = inspect.getsourcelines(fn)
        src_text = ''.join(src_lines)
        func_ast = cast(ast.FunctionDef, ast.parse(src_text).body[0])

        fname = fn.__code__.co_filename or f'<{fn.__name__}>'

        local_map = build_origin_map(
            tree=func_ast, 
            filename=fname, 
            source=src_text, 
            line_offset=start_line-1
        )
        get_origin_map().update(local_map.items()) # merge into global singleton

        nspace = fn.__globals__

        # Replace kernel calls with CS_KernelCall --

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
        kcall_permutations: dict[str, dict[str,list]] = defaultdict(lambda: defaultdict(list))

        def replace_kernel_call(match: dict[str,Any]) -> list[ast.AST]:
            call: ast.Call = match['call']
            kernel_name: str = match['kernel_name']

            kwargs: dict[str,str] = {
                k:v.id
                for k,v in map_call_args_to_kwargs(call, nspace[kernel_name].func).items()
                if isinstance(v,ast.Name)
            }

            kernel: CaKernel = nspace[kernel_name]
            for layer in kernel.layer_args:
                if layer not in kwargs:
                    raise TranspileError(f"Kernel call missing argument: '{layer}'", match['expr'])

            ca_kernel_obj = nspace[kernel_name]
            desc = KernelCallDescriptor(
                kernel_name,
                {arg:v for arg,v in kwargs.items()},
            )
            new_node = ast.Expr(casys_ast.Cs_KernelCall(desc)) # type: ignore
            kernels[kernel_name] = nspace[kernel_name]

            for arg,v in kwargs.items():
                kcall_permutations[kernel_name][arg].append(groups[v])

                if (
                    len(kcall_permutations[kernel_name]) > 1
                    and
                    any(groups[l].has_dirty_offspring for l in kcall_permutations[kernel.name])
                ):
                    raise NotImplementedError("Kernel permutations not implemented.")

            kernel.calls.append(desc)
            
            return [new_node]
        
        PatternTransformer(kernel_calls_pattern, {'expr': replace_kernel_call}).visit(func_ast)
        kcall_permutations = {k:dict(v) for k,v in kcall_permutations.items()} # Convert default dicts to normal dicts

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC,),
            ast_node=func_ast
        )

        trkr.exit_phase()

        return cls(
            fn,
            func_ast,
            world_schema,
            kernels,
            kcall_permutations,
        )
        