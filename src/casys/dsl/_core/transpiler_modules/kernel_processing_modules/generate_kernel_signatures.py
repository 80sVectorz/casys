import ast
from os import access
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numba import from_dtype

from casys.dsl._core.core_transpiler import MDK_DIMS_SIGNED_NB_TYPES, TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

if TYPE_CHECKING:
    from casys.dsl._core.soa_field_usage_info_helper import SoaFieldUsageInfo

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS, MDK_DIMS_UNSIGNED_NB_TYPES, MDK_SOA_LAYOUT
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_SOA_FIELD_USAGE_INFO, MDK_NEEDS_DEDICATED_IDX, MDK_SIGNATURE
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_NEEDS_DEDICATED_IDX

from casys.dsl._core.kernel_values import KV_TIMESTAMP, f_kv_pos_ax, f_kv_size_ax

class GenerateKernelSignatures(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Generating kernel function signatures')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        dims_signed_nb_types = ir.metadata.get(MDK_DIMS_SIGNED_NB_TYPES)
        dims_unsigned_nb_types = ir.metadata.get(MDK_DIMS_UNSIGNED_NB_TYPES)

        layer_usage: SoaFieldUsageInfo

        soa_layout = ir.metadata.get(MDK_SOA_LAYOUT)

        for name, kernel in ir.kernels.items():
            layer_usage = kernel.metadata.get(MDK_SOA_FIELD_USAGE_INFO)

            get_field_nb_type = lambda field: from_dtype(field.data_type).__getitem__(args=[slice(None),*[slice(None) for _ in dims]])

            buffer_args = {
                field_name: get_field_nb_type(field)
                for field_name, field in soa_layout.fields.items()
                if layer_usage.check_accesses(field_name)
            }

            pos_args = {
                f_kv_pos_ax(ax):nb_type
                for ax,nb_type in zip(range(len(dims)),dims_signed_nb_types)
            }
            ax_size_args = {
                f_kv_size_ax(ax):nb_type
                for ax,nb_type in zip(range(len(dims)),dims_unsigned_nb_types)
            }

            args = {
                **buffer_args,
                **pos_args,
                **ax_size_args,
                KV_TIMESTAMP: from_dtype(np.uint64),
            }

            kernel.metadata.set(MDK_SIGNATURE, args)

            kernel.ir_ast = ast.FunctionDef(
                name=kernel.ir_ast.name,
                args=ast.arguments(posonlyargs=[ ast.arg(arg) for arg in args ]),
                body=kernel.ir_ast.body,
            )

            trkr.add_snapshot(
                tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                ast_node=kernel.ir_ast,
                metadata=kernel.metadata,
            )
        

        trkr.exit_phase()

            