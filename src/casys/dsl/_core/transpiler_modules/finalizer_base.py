from __future__ import annotations

from typing import TYPE_CHECKING, Sequence


if TYPE_CHECKING:
    from casys.dsl._core.core_transpiler import TranspilerModule

from casys.dsl._core.transpiler_modules.util_modules import PipelineSection

from casys.dsl._core.transpiler_modules.schema_access_modules import resolve_schema_refs

from casys.dsl._core.transpiler_modules.step_func_modules import (
    # analyze_index_requirements,
    bake_step_func_to_python, 
    generate_step_func_signature,
    optimize_parallel_grouping,
)

from casys.dsl._core.transpiler_modules.kernel_processing_modules import (
    bake_kernels_to_python,
    generate_kernel_signatures,
)

class BaseFinalizer(PipelineSection):

    @property
    def pipeline(self) -> list[TranspilerModule | None]:
        pipeline = [
            # Step function stuff
            optimize_parallel_grouping.OptimizeParallelGrouping(),
            
            # Function signatures
            generate_kernel_signatures.GenerateKernelSignatures(),
            generate_step_func_signature.GenerateStepFuncSignature(),

            # Final python conversion
            bake_kernels_to_python.BakeKernelsToPython(),
            bake_step_func_to_python.BakeStepFuncToPython(),

        ]
        return pipeline
