from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys

from casys.dsl._core.core_transpiler import TranspilerModule

from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

from casys.dsl._core.kernel_values import f_kv_pos_ax, f_kv_size_ax
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_SIGNATURE
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_CONSTANTS, MDK_DIMS

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import Filter, PatternTransformer, Collect, Bind, NodePattern

class BakeKernelsToPython(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Baking kernels to final Python code')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        constants: dict[str, Any] = ir.metadata.get(MDK_CONSTANTS)

        ptrn_buffer_refs = [
            NodePattern(
                ast.Subscript,
                value=Collect(NodePattern(
                    casys_ast.Cs_SoaFieldRef,
                    field=Bind('fld'),
                ), 'soa_field_ref'),
                slice=Collect(NodePattern(ast.Tuple),'slice')
            )
        ]

        ptrn_double_buf_idx = [
            Filter(lambda n: isinstance(n, (casys_ast.Cs_RdIdx, casys_ast.Cs_WrIdx)), 'idx')
        ]

        ptrn_axis_pos = [
            Collect( NodePattern(
                casys_ast.Cs_KPos,
                ax=Bind('ax'),
            ), 'kpos')
        ]

        ptrn_axis_size = [
            Collect( NodePattern(
                casys_ast.Cs_AxisSize,
                ax=Bind('ax'),
            ), 'axis_size')
        ]

        ptrn_constants = [
            Collect( NodePattern(
                casys_ast.Cs_Constant,
                constant_id=Bind('id'),
            ), 'constant')
        ]

        for name, kernel in ir.kernels.items():
            signature = kernel.metadata.get(MDK_SIGNATURE)

            transformers = (
                PatternTransformer(ptrn_buffer_refs, {
                    'soa_field_ref': lambda m: [ast.Name(m['fld'].name)],
                }),

                PatternTransformer(ptrn_double_buf_idx, {
                    'idx': lambda m: [
                        ast.Constant(1)
                        if isinstance(m['idx'], casys_ast.Cs_WrIdx) else
                        ast.Constant(0)
                    ],
                }),

                PatternTransformer(ptrn_axis_pos, {
                    'kpos': lambda m: [ast.Name(f_kv_pos_ax(m['ax']))],
                }),

                PatternTransformer(ptrn_axis_size, {
                    'axis_size': lambda m: [ast.Name(f_kv_size_ax(m['ax']))],
                }),

                PatternTransformer(ptrn_constants, {
                    'constant': lambda m: [ast.Constant(constants[m['id']])],
                }),
            )

            for tf in transformers: tf.visit(kernel.ir_ast)

            trkr.add_snapshot(
                tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                ast_node=kernel.ir_ast
            )

        trkr.exit_phase()

            