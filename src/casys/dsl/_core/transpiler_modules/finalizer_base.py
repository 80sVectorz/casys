from __future__ import annotations

from typing import TYPE_CHECKING, Sequence


if TYPE_CHECKING:
    from casys.dsl._core.core_transpiler import TranspilerModule

from casys.config import CASYS_CONFIG
from casys.dsl._core.transpiler_modules.util_modules import PipelineSection

from casys.dsl._core.transpiler_modules.schema_access_modules import resolve_schema_refs

from casys.dsl._core.transpiler_modules.step_func_modules import (
    # analyze_index_requirements,
    bake_step_func_to_python, 
    generate_step_func_signature,
    optimize_parallel_grouping,
    cpu_compile_step_func,
)

from casys.dsl._core.transpiler_modules.kernel_processing_modules import (
    bake_kernels_to_python,
    generate_kernel_signatures,
    cpu_compile_kernels,
)

from casys.dsl._core.transpiler_modules.cuda import (
    bake_step_func_to_cuda,
    cuda_compile_kernels,
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

            bake_kernels_to_python.BakeKernelsToPython(),

            # Final python conversion
            *([
                cuda_compile_kernels.CudaCompileKernels(),
                bake_step_func_to_cuda.BakeStepFuncToCUDA(),
            ]
            if CASYS_CONFIG.backend == 'cuda' else
            [
                bake_step_func_to_python.BakeStepFuncToPython(),
                cpu_compile_kernels.CpuCompileKernels(),
                cpu_compile_step_func.CpuCompileStepFunc(),
            ]),

        ]
        return pipeline
