from __future__ import annotations

from dataclasses import dataclass
import pprint
from typing import Callable, cast
import numpy as np
import numba
import ast, inspect
from typing import get_type_hints

from .soa import SOASchema
from casys.wrappers import CASysStepFunc, CAKernel, CACType
from .utils import namespace_canonicalize_modules

class StepFunctionDescriptor:
    """Holds raw step function data, which will be handed off to the pipeline."""

    def __init__(self, wrapped_func: CASysStepFunc) -> None:
        func = wrapped_func.func
        self.func = func
        self.src = inspect.getsource(func)
        self.globals = func.__globals__
        namespace_canonicalize_modules(self.globals)

        self.hints = get_type_hints(func, include_extras=True)

@dataclass
class KernelCallDescriptor:
    kernel_name: str
    args: list[str]
    kwargs: dict[str]
    func: Callable

@dataclass(frozen=True)
class StepFuncMetadata:
    fndef: ast.FunctionDef
    cact_nspace: dict[str, CACType]
    kernel_nspace: dict[str, CAKernel]
    soa: SOASchema
    calls: tuple[tuple[str,SOASchema],...]

class StepFuncPreprocessor:

    def preprocess(self, desc: StepFunctionDescriptor) -> StepFuncMetadata:
        func = desc.func
        tree = ast.parse(desc.src)    
        fndef = cast(ast.FunctionDef, tree.body[0])

        hints = get_type_hints(func, include_extras=True)

        # astpretty.pprint(fndef)

        soa_input_schema = tuple((
            (k, t.__cac_type__)
            for k,t in hints.items()
            if hasattr(t, "__cac_type__"))
        )
        soa = SOASchema(soa_input_schema)

        cact_nspace = {
            cast(CACType,t.__cac_type__).dclass.__qualname__:t.__cac_type__
            for t in hints.values()
            if hasattr(t, "__cac_type__")
        }

        # —————————————————————————————————————————————
        # Build groups of CA‐kernel calls, splitting at step_func_split()
        # —————————————————————————————————————————————
        grouped: list[list[tuple[str, SOASchema]]] = []
        current: list[tuple[str, SOASchema]] = []
        for node in fndef.body:
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func

            # split marker?
            if isinstance(func, ast.Name) and func.id == 'step_func_split':
                grouped.append(tuple(current))
                current = []
                continue

            # CA‐kernel call?
            if isinstance(func, ast.Name) and func.id in desc.globals \
               and isinstance(desc.globals[func.id], CAKernel):
                schema = SOASchema(input_schema=tuple(
                    (arg.id, hints[arg.id].__cac_type__)
                    for arg in node.value.args if isinstance(arg, ast.Name)
                ))
                current.append((func.id, schema))
        # final group (even if empty)
        grouped.append(tuple(current))
        calls = tuple(grouped)

        kernels_nspace = {
            kernel_name:desc.globals[kernel_name]
            for calls_group in calls
            for kernel_name, _ in calls_group
        }

        return StepFuncMetadata(
            fndef=fndef,
            cact_nspace=cact_nspace,
            kernel_nspace=kernels_nspace,
            soa=soa,
            calls=calls
        )