from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.transpiler_modules.util_modules import PipelineSection

from casys.dsl._core.transpiler_modules.step_func_modules import (
    mark_swaps,
    create_parallel_groups, 
    # analyze_index_requirements,
    # analyze_kcall_buffer_usage,
)

class BaseStepFuncProcessor(PipelineSection):
    section_phase = 'Processing step function'

    @property
    def pipeline(self) -> list[TranspilerModule | None]:
        return [
            mark_swaps.MarkSwaps(),
            # analyze_index_requirements.AnalyzeIndexRequirements(),
            # analyze_kcall_buffer_usage.AnalyzeKCallLayerUsage(),
            create_parallel_groups.CreateParallelGroups(),
        ]