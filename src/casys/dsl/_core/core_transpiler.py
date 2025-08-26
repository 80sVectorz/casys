from __future__ import annotations
from typing import TYPE_CHECKING, Any, Type, TypedDict

from numba.core.typeconv.castgraph import CastSet

if TYPE_CHECKING:
    from casys._step_func import SimStepFunc
    from casys.wrappers import CaSimConstants

from .ir import Ir_CaSys

from casys.dsl._core.metadata_store import MetadataStore
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import (
    CoreConfig,
    MDK_CORE_CONF, MDK_DIMS, MDK_CONSTANTS
)

from casys.dsl._core.ca_system import CaSystem

class TranspilerModule:
    """
    Base class for all transpiler modules.
    """
    module_requirements: list[TranspilerModule] = []
    dirties: list[TranspilerModule] = []

    def process(self, ir: Ir_CaSys) -> None:
        raise NotImplementedError
    
class Transpiler:
    """Base class that full transpiler implementations inherit from"""
    ir_obj: Ir_CaSys
    sim_constants: CaSimConstants | Type[CaSimConstants]

    def __init__(self, step_func: SimStepFunc, constants: CaSimConstants | Type[CaSimConstants]) -> None:
        self.sim_constants = constants

        conf: CoreConfig = {
            'strict_kernels': constants.strict_kernels
        }

        constants_req = {
            c: getattr(constants, c)
            for c in {kc for k in step_func.kernels.values() for kc in k.req_constants}
        }

        self.ir_obj = (ir_obj:=Ir_CaSys.from_step_func(step_func))

        ir_obj.metadata.set(MDK_CORE_CONF, conf)
        ir_obj.metadata.set(MDK_DIMS, tuple(constants.dims))
        ir_obj.metadata.set(MDK_CONSTANTS, constants_req)

    def transpile(self) -> CaSystem:
        raise NotImplementedError