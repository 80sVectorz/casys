from __future__ import annotations

from typing import TYPE_CHECKING

from numba import from_dtype
import numpy as np

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys

from casys.dsl._core.core_transpiler import MDK_DIMS, MDK_DIMS_SIGNED_NB_TYPES, TranspilerModule
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module

import ast
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_DEDICATED_IDX_IDS, MDK_SIGNATURE, MDK_SIGNATURE_BUFFERS
from casys.dsl._core.kernel_values import (
    KV_N_SIM_STEP_REPEATS, KV_WR_IDX, KV_TIMESTAMP
)

class GenerateStepFuncSignature(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Generating step function signature')
        
        buffers = ir.step_func.base.buffers

        dims = ir.metadata.get(MDK_DIMS)
        dims_signed_nb_types = ir.metadata.get(MDK_DIMS_SIGNED_NB_TYPES)

        get_field_buffer_nb_type = lambda field: from_dtype(field.field_type.true_type).__getitem__(args=[slice(None),*[slice(None) for _ in dims]])

        buffer_args = {
            f'{b}_{fld}':get_field_buffer_nb_type(field)
            for b,buffer in buffers.items()
            for fld, field in buffer.cact._fields.items()
        }

        dedicates_idx_ids = ir.step_func.metadata.get(MDK_DEDICATED_IDX_IDS)
        idx_args = {
            k:from_dtype(np.uint8)
            for k in [KV_WR_IDX, *dedicates_idx_ids.values()]
        }

        args = {
            **buffer_args,
            **idx_args,
            KV_TIMESTAMP: from_dtype(np.uint64),
            # KV_N_SIM_STEP_REPEATS: from_dtype(np.uint64),
        }

        ir.step_func.metadata.set(MDK_SIGNATURE, args)
        ir.step_func.metadata.set(MDK_SIGNATURE_BUFFERS, list(buffer_args.keys()))

        ir.step_func.ir_ast = ast.FunctionDef(
            name=ir.step_func.ir_ast.name,
            args=ast.arguments(
                posonlyargs=[
                    *[ast.arg(arg) for arg in args],
                ],
            ),
            body=ir.step_func.ir_ast.body
        )

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC, f_tag_transpiler_module(self)),
            ast_node=ir.step_func.ir_ast,
            metadata=ir.step_func.metadata,
        )

        trkr.exit_phase()

            