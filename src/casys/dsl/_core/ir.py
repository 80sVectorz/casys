from __future__ import annotations
import copy
from dataclasses import dataclass, field

import ast
from typing import TYPE_CHECKING, Callable

from casys.dsl._core import casys_ast
from casys.dsl._core.metadata_store import MetadataStore

if TYPE_CHECKING:
    from casys._cac_type import CaCellType, CactField
    from casys._ca_kernel import CaKernel
    from casys._step_func import SimStepFunc

@dataclass(kw_only=True)
class Ir_Base:
    metadata: MetadataStore = field(default_factory=MetadataStore)

@dataclass
class Ir_CaKernel(Ir_Base):
    base: CaKernel
    ir_ast: ast.FunctionDef
    nb_kernel: Callable[...,None] | None = None

    @classmethod
    def from_ca_kernel(cls, kernel: CaKernel) -> Ir_CaKernel:
        result = cls(
            base=kernel,
            ir_ast=copy.deepcopy(kernel.func_ast)
        )

        for node in ast.walk(result.ir_ast):
            casys_ast.get_meta(node).source_ir = result

        return result

@dataclass
class Ir_SimStepFunc(Ir_Base):
    base: SimStepFunc
    ir_ast: ast.FunctionDef
    nb_func: Callable[...,None] | None = None
    soa_fields: dict[str, CactField] | None = None

    @classmethod
    def from_step_func(cls, step_func: SimStepFunc) -> Ir_SimStepFunc:
        result = cls(
            base=step_func,
            ir_ast=copy.deepcopy(step_func.func_ast)
        )

        for node in ast.walk(result.ir_ast):
            casys_ast.get_meta(node).source_ir = result

        return result

@dataclass
class Ir_CaSys(Ir_Base):
    kernels: dict[str,Ir_CaKernel]
    step_func: Ir_SimStepFunc
    cac_types: dict[str,CaCellType]

    @classmethod
    def from_step_func(cls, step_func: SimStepFunc):
        cac_types = {}
        for cact_desc in step_func.buffers.values():
            if cact_desc.name not in cac_types:
                cac_types[cact_desc.name] = cact_desc.cact

        ir_kernels = {}
        for name,kernel in step_func.kernels.items():
            ir_kernels[name] = Ir_CaKernel.from_ca_kernel(kernel)

        return cls(
            kernels=ir_kernels,
            step_func=Ir_SimStepFunc.from_step_func(step_func),
            cac_types=cac_types
        )