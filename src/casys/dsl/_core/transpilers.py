from __future__ import annotations
import time
from typing import Sequence, Type, TYPE_CHECKING, TypedDict

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_SOA_LAYOUT
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_NEEDS_DEDICATED_IDX, MDK_SIGNATURE, MDK_SIGNATURE_BUFFERS
from casys.dsl._core.kernel_values import f_kv_wr_idx

if TYPE_CHECKING:
    from casys.spec.step_func import SimStepFunc
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
            start_time = time.perf_counter()

            for module in self.pipeline:
                module.process(ir)

            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            message = 'Transpilation and Numba compilation completed in'
            if elapsed_time < 1:
                print(message, f"{elapsed_time * 1000:.2f} ms")
            elif elapsed_time < 60:
                print(message, f'{elapsed_time:.2f} s')
            else:
                minutes, seconds = divmod(elapsed_time, 60)
                milliseconds = (seconds - int(seconds)) * 1000
                print(message, f'{int(minutes)}:{int(seconds):02}:{int(milliseconds):03}')
        finally:
            if CASYS_CONFIG.debug_ast_timeline:
                try:
                    with open(CASYS_CONFIG.debug_timeline_file, 'w', encoding='utf-8') as f:
                        json.dump(trkr.to_json(), f, indent=True, ensure_ascii=False)
                except:
                    log_warning('Saving AST timeline failed')
                    
        sys = CaSystem(
            soa_layout=self.ir_obj.metadata.get(MDK_SOA_LAYOUT),
            world_schema=self.ir_obj.world_schema,
            step_func=self.ir_obj.step_func.base,
            sim_constants=self.sim_constants,
            nb_step_func=ir.step_func.nb_func, # type: ignore
            signature_buffers=ir.step_func.metadata.get(MDK_SIGNATURE_BUFFERS),
        )

        return sys