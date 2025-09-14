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

from casys.dsl._core.core_transpiler import TranspilerModule


from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import (
    MDK_SIGNATURE
)

import ast

class CpuCompileStepFunc(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
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
            depends=[kernel_name for kernel_name in ir.kernels.keys()],
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