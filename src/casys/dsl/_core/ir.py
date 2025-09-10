from __future__ import annotations
import copy
from dataclasses import dataclass, field

import ast
from typing import TYPE_CHECKING, Callable

from casys.dsl._core import casys_ast
from casys.dsl._core.metadata_store import MetadataStore

if TYPE_CHECKING:
    from casys.spec.cac_type import CaCellTypeSpec, CactField
    from casys.spec.ca_kernel import CaKernel
    from casys.spec.step_func import SimStepFunc
    from casys.dsl._core.schema.soa_layout import SoaLayout
    from casys.dsl._core.schema.world_schema import WorldSchema

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
    world_schema: WorldSchema
    kernels: dict[str,Ir_CaKernel]
    step_func: Ir_SimStepFunc

    @classmethod
    def from_step_func(cls, step_func: SimStepFunc):
        ir_kernels = {}
        for name,kernel in step_func.kernels.items():
            ir_kernels[name] = Ir_CaKernel.from_ca_kernel(kernel)

        return cls(
            world_schema=step_func.world_schema,
            kernels=ir_kernels,
            step_func=Ir_SimStepFunc.from_step_func(step_func),
        )