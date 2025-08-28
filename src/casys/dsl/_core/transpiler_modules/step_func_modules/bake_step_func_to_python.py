from hmac import new
from typing import TYPE_CHECKING, Any, Sequence

from collections import Counter
import numba

from casys.config import CASYS_CONFIG

from casys._utils.misc_utils import namespace_canonicalize_modules
from casys.dsl._core.debug.dynsrc import compile_and_exec
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_DEDICATED_IDX_IDS

from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.descriptors import KernelCallDescriptor

from casys.dsl._core.core_transpiler import TranspilerModule

from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module

from casys.dsl._core.kernel_values import KV_TIMESTAMP, KV_WR_IDX, f_kv_pos_ax, f_kv_wr_idx
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_BUFFER_USAGE_INFO, MDK_NEEDS_DEDICATED_IDX
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_CONSTANTS, MDK_DIMS

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern

class BakeStepFuncToPython(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Baking step function to final Python code')

        buffers = ir.step_func.base.buffers
        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        dedicated_idx_ids = ir.step_func.metadata.get(MDK_DEDICATED_IDX_IDS)

        def snippet_flip_idx(idx_id: str):
            return ast.AugAssign(
                ast.Name(idx_id,ast.Store()),
                ast.BitXor(),
                ast.Constant(1),
            )

        def snippet_flip_buffer_idx(buffer: str):
            idx_id = dedicated_idx_ids.get(buffer, KV_WR_IDX)
            return snippet_flip_idx(idx_id)

        def snippet_idx(buffer: str, read=False):
            idx_id = dedicated_idx_ids.get(buffer, KV_WR_IDX)
            name_node = ast.Name(idx_id,ast.Load() if read else ast.Store())
            if read:
                return ast.BinOp(
                    name_node,
                    ast.BitXor(),
                    ast.Constant(1),
                )
            return name_node

        def snippet_buffer_subscript(buffer:str, field: str, read=False, index_read=False):
            return ast.Subscript(ast.Name(f'{buffer}_{field}'), ctx=ast.Load() if read else ast.Store(), slice=ast.Tuple(elts=[
                snippet_idx(buffer, index_read), *[
                    ast.Name(f_kv_pos_ax(ax)) for ax in range(len(dims))
                ]
            ]))

        def snippet_sync_r2w(buffer,field):
            return ast.Assign(
                [snippet_buffer_subscript(buffer,field, index_read=True)],
                snippet_buffer_subscript(buffer,field,read=True, index_read=False)
            )

        def snippet_sync_w2r(buffer,field):
            return ast.Assign(
                [snippet_buffer_subscript(buffer,field, index_read=False)],
                snippet_buffer_subscript(buffer,field,read=True, index_read=True)
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
            buffer_usage = ir.kernels[kcall.kernel_name].metadata.get(MDK_BUFFER_USAGE_INFO) 

            buffer_args = [
                f'{buf}_{fld}'
                for k,v in kcall.kwargs.items()
                for buf,fld in buffers[v].soa_pairs
                if buffer_usage.check_accesses(k,fld)
            ]

            pos_args = [f_kv_pos_ax(ax) for ax in range(len(dims))]

            idx_args = [
                KV_WR_IDX,
                *[
                    dedicated_idx_ids[v]
                    for k,v in kcall.kwargs.items()
                    if buffer_usage.check_accesses(k) and v in dedicated_idx_ids
                ]

            ]

            args = [
                ast.Name(arg, ast.Load()) for arg in [
                    *buffer_args,
                    *pos_args,
                    KV_TIMESTAMP,
                    *idx_args,
                ]
            ]
            
            return ast.Expr(ast.Call(ast.Name(kcall.kernel_name, ast.Load()), args)) # type: ignore

        swap_counts = Counter({KV_WR_IDX:0})
        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup): continue
            for swap in node.swaps:
                swap_counts[dedicated_idx_ids[swap]] += 1
        
        idx_ids = [KV_WR_IDX, *dedicated_idx_ids.values()]

        new_body: list[ast.stmt] = []

        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup): continue

            for swap in node.swaps:
                new_body.append(snippet_flip_buffer_idx(swap))

            if node.calls or node.sync_r2w or node.sync_w2r:
                new_body.append(snippet_parallel_loop(
                    body=[
                        *[snippet_kcall(kcall) for kcall in node.calls],
                        *[snippet_sync_r2w(buffer,field) for buffer,field in node.sync_r2w],
                        *[snippet_sync_w2r(buffer,field) for buffer,field in node.sync_w2r],
                    ]
                ))

        for idx_id in idx_ids:
            new_body.append(snippet_flip_idx(idx_id))

        new_body.append(
            ast.Return(
                ast.Tuple(
                    [ast.Name(idx_id) for idx_id in idx_ids]
                )
            )
        )

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

        compile_and_exec(
            src,
            nspace,
            virtual_filename=f'{ir.step_func.base.func.__name__}.py',
            mirror_kind='step',
        )

        if CASYS_CONFIG.debug_disable_jit not in ('full', 'step_func'):
            nb_func = numba.jit(
                nspace[ir.step_func.base.func.__name__],
                nopython=CASYS_CONFIG.debug_jit_nopython, 
                parallel=not CASYS_CONFIG.debug_disable_cpu_parallelization,
                boundscheck = CASYS_CONFIG.debug_jit_enable_bounds_check,
            )
        else:
            nb_func = nspace[ir.step_func.base.func.__name__]

        ir.step_func.nb_func = nb_func

        trkr.exit_phase()