from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
import time, ast
import numba
import numpy as np
from numba import cuda

from casys.config import CASYS_CONFIG
from casys._utils.misc_utils import namespace_canonicalize_modules
from casys.dsl._core.source_management import import_from_source, get_assigned_names
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS, MDK_CONSTANTS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_SIGNATURE as MDK_KERNEL_SIGNATURE

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys

class CudaCompileKernels(TranspilerModule):
    """Emit @cuda.jit(device=True) functions for each kernel."""

    def process(self, ir: Ir_CaSys) -> None:

        for name, kernel in ir.kernels.items():

            ast.fix_missing_locations(kernel.ir_ast)
            src = ast.unparse(kernel.ir_ast)

            nspace = kernel.base.func.__globals__
            nspace['np'] = np
            nspace['cuda'] = cuda
            namespace_canonicalize_modules(nspace)

            mod = import_from_source(
                src,
                virtual_filename=f'{name}__cuda_dev.py',
                mirror_kind='kernel',
                cache_salt=f'cuda_device',
                nspace=nspace,
                dep_mode='scan',
                inject_into_module=False,
            )

            fn = mod.__dict__[name]

            _defined = get_assigned_names(src)
            deps = {k: v for k, v in nspace.items() if k not in _defined and k != name}
            fn.__globals__.update(deps)
            
            sig = kernel.metadata.get(MDK_KERNEL_SIGNATURE)

            start_time = time.perf_counter()
            # dev_func = cuda.jit(numba.void(*sig.values()), device=True, cache=True, inline=False, debug=CASYS_CONFIG.debug_jit_enable_bounds_check, opt = not CASYS_CONFIG.debug_jit_enable_bounds_check)(fn)
            dev_func = cuda.jit(device=True, cache=True, inline=True, debug=CASYS_CONFIG.debug_jit_enable_bounds_check, opt = not CASYS_CONFIG.debug_jit_enable_bounds_check)(fn)
            end_time = time.perf_counter()

            message = f"Kernel '{name}' Numba CUDA compilation completed in"
            elapsed_time = end_time - start_time
            if elapsed_time < 1:
                print(message, f'{elapsed_time * 1000:.2f} ms')
            elif elapsed_time < 60:
                print(message, f'{elapsed_time:.2f} s')
            else:
                minutes, seconds = divmod(elapsed_time, 60)
                milliseconds = (seconds - int(seconds)) * 1000
                print(message, f'{int(minutes)}:{int(seconds):02}:{int(milliseconds):03}')

            kernel.cuda_device = dev_func # type: ignore