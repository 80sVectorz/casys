from casys.dsl._core.transpiler_modules.util_modules import PipelineSection

from casys.dsl._core.core_transpiler import Ir_CaSys, TranspilerModule
from casys.dsl._core.kernel_values import BASE_RESERVED_NAMES

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import (
    MDK_CORE_CONF
)
from casys.dsl._core.ir_metadata_specs.md_kernels_base import (
    MDK_READONLY
)

from casys.dsl._core.transpiler_modules.kernel_processing_modules import (
    handle_macros_kernels,
    handle_k_gets,
    insert_schema_refs,

    mark_pos_vars,
    insert_bounds_logic,

    insert_double_buffer_indexing,

    validate_readonly,

    analyze_soa_fields_usage,

    validate_strict_kernels,
)
from casys.dsl._core.transpiler_modules.schema_access_modules import resolve_schema_refs

class BaseKernelsProcessor(PipelineSection):
    section_phase = 'Processing kernels'

    @property
    def pipeline(self) -> list[TranspilerModule | None]:
        ir = self.ir

        return [
            handle_macros_kernels.HandleMacrosKernels(),
            insert_schema_refs.InsertSchemaRefs(),

            handle_k_gets.HandleKGets(),
            mark_pos_vars.MarkPosVars(),
            insert_bounds_logic.InsertBoundsLogic(),
            insert_double_buffer_indexing.InsertDoubleBufferIndexing(),

            validate_readonly.ValidateReadonly(),

            resolve_schema_refs.ResolveSchemaRefs(),

            analyze_soa_fields_usage.AnalyzeSoaFieldsUsage(),

            validate_strict_kernels.ValidateStrictKernels()
            if ir.metadata.get(MDK_CORE_CONF)['strict_kernels'] else None,
        ]

    def process(self, ir: Ir_CaSys) -> None:
        for kernel in ir.kernels.values():
            kernel.metadata.set(MDK_READONLY, set(BASE_RESERVED_NAMES))

        super().process(ir)
        
