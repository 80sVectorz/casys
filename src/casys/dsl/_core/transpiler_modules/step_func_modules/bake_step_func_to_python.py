from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

from collections import Counter
import numba
import numpy as np
import time

from casys.config import CASYS_CONFIG

from casys._utils.misc_utils import namespace_canonicalize_modules
from casys.dsl._core.source_management import import_from_source, get_assigned_names

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys
    from casys.dsl._core.descriptors import KernelCallDescriptor

from casys.dsl._core.core_transpiler import TranspilerModule

from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module

from casys.dsl._core.kernel_values import KV_I_SIM_STEP_INTERNAL, KV_N_SIM_STEP_REPEATS, KV_RD_IDX, KV_TIMESTAMP, KV_WR_IDX, f_kv_pos_ax, f_kv_rd_idx, f_kv_size_ax, f_kv_wr_idx
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import (
    MDK_NEEDS_DEDICATED_IDX as MDK_NEEDS_DEDICATED_IDX_SF,
    MDK_SIGNATURE
)
from casys.dsl._core.ir_metadata_specs.md_kernels_base import (
    MDK_SOA_FIELD_USAGE_INFO, 
    MDK_NEEDS_DEDICATED_IDX as MDK_NEEDS_DEDICATED_IDX_KR,
    MDK_SIGNATURE as MDK_KERNEL_SIGNATURE
)
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_CONSTANTS, MDK_DIMS, MDK_SOA_LAYOUT

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern

class BakeStepFuncToPython(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Baking step function to final Python code')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)

        def snippet_idx(read=False):
            if read:
                return ast.Constant(0)
            return ast.Constant(1)

        def snippet_buffer_subscript(buffer:str, read=False, index_read=False):
            return ast.Subscript(ast.Name(buffer), ctx=ast.Load() if read else ast.Store(), slice=ast.Tuple(elts=[
                snippet_idx(index_read), *[
                    ast.Name(f_kv_pos_ax(ax)) for ax in range(len(dims))
                ]
            ]))

        def snippet_sync_r2w(buffer):
            return ast.Assign(
                [snippet_buffer_subscript(buffer, index_read=True)],
                snippet_buffer_subscript(buffer,read=True, index_read=False)
            )

        def snippet_sync_w2r(buffer):
            return ast.Assign(
                [snippet_buffer_subscript(buffer, index_read=False)],
                snippet_buffer_subscript(buffer,read=True, index_read=True)
            )
        
        def snippet_loop(ax: int):
            range_function = (
                ast.Attribute(ast.Name('numba'),'prange')
                if not CASYS_CONFIG.debug_disable_cpu_parallelization else
                ast.Name('range')
            )

            return ast.For(ast.Name(f_kv_pos_ax(ax)), ast.Call(range_function,[ast.Constant(dims[ax])]))

        def snippet_parallel_loop(body: list[ast.stmt]):
            top_loop = snippet_loop(0)
            loop_node = top_loop
            for ax in range(1,len(dims)):
                child_loop = snippet_loop(ax)
                loop_node.body.append(child_loop)
                loop_node = child_loop

            loop_node.body = body
            return top_loop
        
        def snippet_kcall(kcall: KernelCallDescriptor):
            kernel = ir.kernels[kcall.kernel_name]

            args = [
                k for k in kernel.metadata.get(MDK_KERNEL_SIGNATURE).keys()
            ]

            args = [
                ast.Name(arg, ast.Load()) for arg in args
            ]
            
            return ast.Expr(ast.Call(ast.Name(kcall.kernel_name, ast.Load()), args)) # type: ignore

        new_body: list[ast.stmt] = []

        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup): continue

            if node.calls or node.sync_r2w or node.sync_w2r:
                new_body.append(snippet_parallel_loop(
                    body=[
                        *[snippet_sync_w2r(soa_field_buffer) for soa_field_buffer in node.sync_w2r],
                        *[snippet_kcall(kcall) for kcall in node.calls],
                        *[snippet_sync_r2w(soa_field_buffer) for soa_field_buffer in node.sync_r2w],
                    ]
                ))

        ir.step_func.ir_ast.body = new_body

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC,f_tag_transpiler_module(self)),
            ast_node=ir.step_func.ir_ast
        )

        ast.fix_missing_locations(ir.step_func.ir_ast)
        src = ast.unparse(ir.step_func.ir_ast)

        nspace = ir.step_func.base.func.__globals__
        nspace['numba'] = numba
        namespace_canonicalize_modules(nspace)

        for kernel_name, kernel in ir.kernels.items():
            nspace[kernel_name] = kernel.nb_kernel

        fn_name = ir.step_func.base.func.__name__
        module = import_from_source(
            src,
            virtual_filename=f'{fn_name}.py',
            mirror_kind='step',
            cache_salt=(
                f'par={not CASYS_CONFIG.debug_disable_cpu_parallelization};'
                f'bc={CASYS_CONFIG.debug_jit_enable_bounds_check}'
            ),
            nspace=nspace,
            dep_mode='explicit',
            inject_into_module=False,
        )

        fn = module.__dict__[fn_name]

        # Inject dependencies into the function globals.
        _defined = get_assigned_names(src)
        deps = {k: v for k, v in nspace.items() if k not in _defined and k != fn_name}
        fn.__globals__.update(deps)

        signature = ir.step_func.metadata.get(MDK_SIGNATURE)
        if CASYS_CONFIG.debug_disable_jit not in ('full', 'step_func'):
            start_time = time.perf_counter()
            nb_func = numba.jit(
                numba.types.void(*signature.values()),
                nopython=CASYS_CONFIG.debug_jit_nopython, 
                parallel=not CASYS_CONFIG.debug_disable_cpu_parallelization,
                boundscheck = CASYS_CONFIG.debug_jit_enable_bounds_check,
                cache=True,
            )(fn)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            message = 'Simulation step function Numba compilation completed in'
            if elapsed_time < 1:
                print(message, f"{elapsed_time * 1000:.2f} ms")
            elif elapsed_time < 60:
                print(message, f'{elapsed_time:.2f} s')
            else:
                minutes, seconds = divmod(elapsed_time, 60)
                milliseconds = (seconds - int(seconds)) * 1000
                print(message, f'{int(minutes)}:{int(seconds):02}:{int(milliseconds):03}')
        else:
            nb_func = fn

        ir.step_func.nb_func = nb_func

        trkr.exit_phase()