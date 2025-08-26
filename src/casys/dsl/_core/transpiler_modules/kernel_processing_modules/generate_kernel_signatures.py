import ast
from typing import TYPE_CHECKING, Sequence

from vispy.gloo import buffer
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.descriptors import CactBufferDescriptor
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

if TYPE_CHECKING:
    from casys.dsl._core.ir_metadata_specs.md_kernels_base import BufferUsageInfo

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_BUFFER_USAGE_INFO, MDK_NEEDS_DEDICATED_IDX, MDK_SIGNATURE
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_DEDICATED_IDX_IDS

from casys.dsl._core.kernel_values import KV_TIMESTAMP, KV_WR_IDX, f_kv_pos_ax, f_kv_wr_idx

class GenerateKernelSignatures(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Generating kernel function signatures')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        dedicated_idx_ids: dict[str, str] = ir.step_func.metadata.get(MDK_DEDICATED_IDX_IDS)
        buffer_usage: BufferUsageInfo

        for name, kernel in ir.kernels.items():
            buffer_usage = kernel.metadata.get(MDK_BUFFER_USAGE_INFO)
            needs_dedicated_idx: set[str] = set()

            for kernel_call in kernel.base.calls:
                for k,v in kernel_call.kwargs.items():
                    if v in dedicated_idx_ids:
                        needs_dedicated_idx.add(k)

            kernel.metadata.set(MDK_NEEDS_DEDICATED_IDX,needs_dedicated_idx)
            
            buffer_args = [f'{b}_{fld}' for b,fld in CactBufferDescriptor.get_soa_pairs_multi(
                descriptors=kernel.base.buffers.values(),
                filter_predicate=buffer_usage.check_accesses
            )]

            pos_args = [f_kv_pos_ax(ax) for ax in range(len(dims))]
            # size_args = [f_kv_size_ax(ax) for ax in range(len(dims))]
            idx_args = [KV_WR_IDX, *[f_kv_wr_idx(b) for b in kernel.base.buffers if b in needs_dedicated_idx and buffer_usage.check_accesses(b)]]

            args = [
                *buffer_args,
                *pos_args,
                KV_TIMESTAMP,
                *idx_args,
            ]

            kernel.metadata.set(MDK_SIGNATURE, args)

            kernel.ir_ast = ast.FunctionDef(
                name=kernel.ir_ast.name,
                args=ast.arguments(posonlyargs=[ ast.arg(arg) for arg in args ]),
                body=kernel.ir_ast.body
            )

            trkr.add_snapshot(
                tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                ast_node=kernel.ir_ast,
                metadata=kernel.metadata,
            )
        

        trkr.exit_phase()

            