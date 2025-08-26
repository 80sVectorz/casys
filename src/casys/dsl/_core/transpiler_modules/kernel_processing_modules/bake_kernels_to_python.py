from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

import numba
from numba.np.ufunc import parallel

from casys.config import CASYS_CONFIG
from casys.dsl._core.debug.dynsrc import compile_and_exec

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys

from casys._utils.misc_utils import namespace_canonicalize_modules
from casys.dsl._core.core_transpiler import TranspilerModule

from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

from casys.dsl._core.kernel_values import KV_WR_IDX, f_kv_pos_ax, f_kv_wr_idx
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_NEEDS_DEDICATED_IDX
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_CONSTANTS, MDK_DIMS

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern

class BakeKernelsToPython(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Baking kernels to final Python code')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        constants: dict[str, Any] = ir.metadata.get(MDK_CONSTANTS)
        needs_dedicated_idx: set[str]

        ptrn_buffer_refs = [
            NodePattern(
                ast.Subscript,
                value=Collect(NodePattern(
                    casys_ast.Cs_BufferRef,
                    b=Bind('b'),
                    f=Bind('f'),
                ), 'buffer_ref'),
                slice=Collect(NodePattern(ast.Tuple),'slice')
            )
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
            needs_dedicated_idx = kernel.metadata.get(MDK_NEEDS_DEDICATED_IDX)

            (tf1:=PatternTransformer(ptrn_buffer_refs, {
                'buffer_ref': lambda m: [ast.Name(f"{m['b']}_{m['f']}")],
                'slice': lambda m: [ast.Tuple([
                    (
                        ast.Name(f_kv_wr_idx(m['b']) if m['b'] in needs_dedicated_idx else KV_WR_IDX)
                        if isinstance(m['slice'].elts[0], casys_ast.Cs_WrIdx) else
                        ast.BinOp(
                            left=ast.Constant(1),
                            op=ast.BitXor(),
                            right=ast.Name(f_kv_wr_idx(m['b']) if m['b'] in needs_dedicated_idx else KV_WR_IDX)
                        )
                    ),
                    *m['slice'].elts[1:]
                ])],
            })).visit(kernel.ir_ast)

            (tf2:=PatternTransformer(ptrn_axis_pos, {
                'kpos': lambda m: [ast.Name(f_kv_pos_ax(m['ax']))],
            })).visit(kernel.ir_ast)

            (tf3:=PatternTransformer(ptrn_axis_size, {
                'axis_size': lambda m: [ast.Constant(dims[m['ax']])],
            })).visit(kernel.ir_ast)

            (tf4:=PatternTransformer(ptrn_constants, {
                'constant': lambda m: [ast.Constant(constants[m['id']])],
            })).visit(kernel.ir_ast)

            if tf1.matches or tf2.matches or tf3.matches or tf4.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name), f_tag_transpiler_module(self)),
                    ast_node=kernel.ir_ast
                )

            ast.fix_missing_locations(kernel.ir_ast)
            src = ast.unparse(kernel.ir_ast)

            nspace = kernel.base.func.__globals__
            nspace['numba'] = numba
            namespace_canonicalize_modules(nspace)

            compile_and_exec(
                src,
                nspace,
                virtual_filename=f'{name}__baked.py',
                mirror_kind='kernel',
            )


            if CASYS_CONFIG.debug_disable_jit != 'full':
                nb_func = numba.jit(
                    nspace[name],
                    nopython=CASYS_CONFIG.debug_jit_nopython,
                    inline='always' if CASYS_CONFIG.debug_jit_inline_kernels else 'never'
                )
            else:
                nb_func = nspace[name]
            

            kernel.nb_kernel = nb_func

        trkr.exit_phase()

            