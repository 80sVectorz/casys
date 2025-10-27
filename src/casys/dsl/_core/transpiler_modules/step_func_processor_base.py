from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.transpiler_modules.util_modules import PipelineSection

from casys.dsl._core.transpiler_modules.step_func_modules import (
    handle_macros_step_func,
    mark_swaps,
    create_parallel_groups, 
)

class BaseStepFuncProcessor(PipelineSection):
    section_phase = 'Processing step function'

    @property
    def pipeline(self) -> list[TranspilerModule | None]:
        return [
            handle_macros_step_func.HandleMacrosStepFunc(),
            mark_swaps.MarkSwaps(),
            create_parallel_groups.CreateParallelGroups(),
        ]