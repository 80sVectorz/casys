# src/casys/dsl/_core/transpiler_modules/cuda/bake_step_func_to_cuda.py
from __future__ import annotations

import ast
from typing import Any, Sequence

from numba import cuda

from casys.config import CASYS_CONFIG
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.debug.ast_timeline_tracking import (
    TAG_STEP_FUNC,
    f_tag_transpiler_module,
    get_tracker,
)
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_SIGNATURE as MDK_KERNEL_SIGNATURE
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import (
    MDK_SIGNATURE,
    MDK_SIGNATURE_BUFFERS,
)
from casys.dsl._core.source_management import import_from_source
from casys.dsl._core import casys_ast
from casys.dsl._core.kernel_values import f_kv_size_ax


def _pick(names: list[str], present: set[str]) -> str:
    """Return the first present name from candidates or raise."""
    for n in names:
        if n in present:
            return n
    raise KeyError(f'None of {names!r} found in step signature')


class BakeStepFuncToCUDA(TranspilerModule):
    """Builds per-parallel-group CUDA kernels and a host launcher using real arg names."""

    def process(self, ir: Any) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Baking CUDA step kernels + launcher')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)  # not used directly, just here if needed
        step_sig: dict[str, Any] = ir.step_func.metadata.get(MDK_SIGNATURE)
        step_buf_sig: list[str] = list(ir.step_func.metadata.get(MDK_SIGNATURE_BUFFERS))
        arg_names: list[str] = list(step_sig.keys())
        arg_set: set[str] = set(arg_names)


        # Canonical names from the existing signature. No temp locals invented.
        size0_name = _pick([f_kv_size_ax(0), 'size0', 'H'], arg_set)
        size1_name = _pick([f_kv_size_ax(1), 'size1', 'W'], arg_set)

        # Build one CUDA kernel per Cs_ParallelGroup.
        group_names: list[str] = []
        group_funcs: list[Any] = []

        # Helper: build an Assign node buf[idx, y, x] = buf[idx2, y, x]
        def _sync_assign(buf_name: str, left_idx: int, right_idx: int) -> ast.Assign:
            slc_l = ast.Subscript(
                value=ast.Name(id=buf_name, ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[ast.Constant(left_idx),
                          ast.Name(id='x', ctx=ast.Load()),
                          ast.Name(id='y', ctx=ast.Load())],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
            slc_r = ast.Subscript(
                value=ast.Name(id=buf_name, ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[ast.Constant(right_idx),
                          ast.Name(id='x', ctx=ast.Load()),
                          ast.Name(id='y', ctx=ast.Load())],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
            return ast.Assign(targets=[slc_l], value=slc_r)

        # Walk the step AST in order and emit a global kernel per parallel group.
        group_idx = 0
        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup):
                continue

            gname = f'__step_group_{group_idx}'
            group_idx += 1

            # Build the body statements in per-cell order
            stmts: list[ast.stmt] = []

            # 1) Thread coords and bounds guard using real arg names
            stmts.append(
                    ast.Assign(
                        targets=[ast.Tuple(elts=[ast.Name('x', ast.Store()), ast.Name('y', ast.Store())], ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name('cuda', ast.Load()), attr='grid', ctx=ast.Load()),
                        args=[ast.Constant(2)],
                        keywords=[],
                    ),
                )
            )
            stmts.append(
                ast.If(
                    test=ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.Compare(left=ast.Name('x', ast.Load()), ops=[ast.GtE()], comparators=[ast.Name(size1_name, ast.Load())]),
                            ast.Compare(left=ast.Name('y', ast.Load()), ops=[ast.GtE()], comparators=[ast.Name(size0_name, ast.Load())]),
                        ],
                    ),
                    body=[ast.Return(value=None)],
                    orelse=[],
                )
            )

            # 2) Pre-kernel syncs
            for bname in getattr(node, 'sync_w2r', []):
                stmts.append(_sync_assign(bname, 1, 0))

            # 3) Kernel calls in order, mapping their signatures to current per-thread args
            for kcall in getattr(node, 'calls', []):
                kmeta = ir.kernels[kcall.kernel_name]
                ksig: dict[str, Any] = kmeta.metadata.get(MDK_KERNEL_SIGNATURE)
                call_args: list[ast.expr] = []
                for pname in ksig.keys():
                    if pname in step_buf_sig:
                        call_args.append(ast.Name(pname, ast.Load()))
                    elif pname in (size0_name, size1_name):
                        call_args.append(ast.Name(pname, ast.Load()))
                    elif pname.endswith('kval_p_ax0'):
                        call_args.append(ast.Name('x', ast.Load()))
                    elif pname.endswith('kval_p_ax1'):
                        call_args.append(ast.Name('y', ast.Load()))
                    else:
                        # Pass through constants or extra scalars by name
                        call_args.append(ast.Name(pname, ast.Load()))
                stmts.append(
                    ast.Expr(
                        value=ast.Call(func=ast.Name(id=kcall.kernel_name + '_dev', ctx=ast.Load()), args=call_args, keywords=[])
                    )
                )


            # 3) Post-kernel syncs
            for bname in getattr(node, 'sync_r2w', []):
                stmts.append(_sync_assign(bname, 0, 1))

            # Function def with the step signature order
            fn = ast.FunctionDef(
                name=gname,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=n) for n in arg_names],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=stmts,
                decorator_list=[],
                returns=None,
            )

            # Wrap into a module with an explicit cuda import
            import_cuda = ast.ImportFrom(module='numba', names=[ast.alias(name='cuda')], level=0)
            mod_ast = ast.Module(body=[import_cuda, fn], type_ignores=[])
            ast.fix_missing_locations(mod_ast)
            src = ast.unparse(mod_ast)

            # Build namespace: provide device fns as globals the kernel can call
            nspace = ir.step_func.base.func.__globals__
            for kname, k in ir.kernels.items():
                nspace[kname + '_dev'] = k.cuda_device

            module = import_from_source(
                src,
                virtual_filename=f'{gname}.py',
                mirror_kind='step',
                cache_salt='cuda_step_group',
                nspace=nspace,
                dep_mode='explicit',
                depends=[kname+'_dev' for kname in ir.kernels.keys()],
                inject_into_module=True,
            )

            cuda_kernel = cuda.jit(getattr(module, gname), fastmath=CASYS_CONFIG.cuda_fastmath, cache=True, debug=CASYS_CONFIG.debug_jit_enable_bounds_check, opt = not CASYS_CONFIG.debug_jit_enable_bounds_check)
            setattr(ir.step_func, gname, cuda_kernel)
            group_names.append(gname)
            group_funcs.append(cuda_kernel)

        # Expose metadata for runtime and helpers for clean precompile
        ir.step_func.arg_names = arg_names
        ir.step_func.buffer_arg_names = step_buf_sig
        ir.step_func.cuda_group_names = group_names

        def step_cuda_host(*args) -> None:
            bx, by, _ = CASYS_CONFIG.cuda_block
            size0 = args[arg_names.index(size0_name)]
            size1 = args[arg_names.index(size1_name)]
            grid = ((size0 + bx - 1) // bx, (size1 + by - 1) // by)
            for g in ir.step_func.cuda_group_names:
                getattr(ir.step_func, g)[grid, (bx, by, 1)](*args)

        ir.step_func.nb_func = step_cuda_host

        trkr.add_snapshot(tags=(TAG_STEP_FUNC, f_tag_transpiler_module(self)))
        trkr.exit_phase()
