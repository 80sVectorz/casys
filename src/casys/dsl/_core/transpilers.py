from __future__ import annotations
from typing import Sequence, Type, TYPE_CHECKING, TypedDict

from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_DEDICATED_IDX_IDS, MDK_SIGNATURE, MDK_SIGNATURE_BUFFERS

if TYPE_CHECKING:
    from casys._step_func import SimStepFunc
    from casys.wrappers import CaSimConstants

from casys.dsl._core.ca_system import CaSystem
from casys.dsl._core.core_transpiler import Transpiler
from .transpiler_modules.step_func_processor_base import BaseStepFuncProcessor
from .transpiler_modules.kernels_base import BaseKernelsProcessor
from .transpiler_modules.finalizer_base import BaseFinalizer

import json
from casys.logging import log_warning

from casys.config import CASYS_CONFIG
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

class BaseTranspiler(Transpiler):
    
    def __init__(self, step_func: SimStepFunc, sim_constants: CaSimConstants | Type[CaSimConstants]) -> None:
        super().__init__(step_func, sim_constants)

        self.pipeline = [
            BaseKernelsProcessor(self.ir_obj),
            BaseStepFuncProcessor(self.ir_obj),
            BaseFinalizer(self.ir_obj),
        ]

    def transpile(self) -> CaSystem:
        ir = self.ir_obj

        trkr = get_tracker()
        trkr.enter_phase('Transpilation')

        try:
            for module in self.pipeline:
                module.process(ir)
        finally:
            if CASYS_CONFIG.debug_ast_timeline:
                try:
                    with open(CASYS_CONFIG.debug_timeline_file, 'w', encoding='utf-8') as f:
                        json.dump(trkr.to_json(), f, indent=True, ensure_ascii=False)
                except:
                    log_warning('Saving AST timeline failed')
                    

        sys = CaSystem(
            step_func=self.ir_obj.step_func.base,
            sim_constants=self.sim_constants,
            nb_step_func=ir.step_func.nb_func, # type: ignore
            signature_buffers=ir.step_func.metadata.get(MDK_SIGNATURE_BUFFERS),
            dedicated_idx_ids=ir.step_func.metadata.get(MDK_DEDICATED_IDX_IDS)
        )

        return sys