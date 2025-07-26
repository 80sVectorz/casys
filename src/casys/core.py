from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Self, cast
from numba.np.ufunc import parallel
import numpy as np
import numba as nb
import ast, inspect
import astpretty
from typing import get_type_hints

from .debug_utils import preview, header
from casys.kernel_values import KV_TIMESTAMP, KV_HEIGHT, KV_WIDTH, KV_LD_IDX, KV_PX, KV_PY, KV_WR_IDX

from .wrappers import CASysStepFunc, CAKernel, CACType
from .step_func_preprocessing import StepFuncMetadata, StepFunctionDescriptor, StepFuncPreprocessor
from .kernel_processing import KernelDescriptor, KernelMetadata, KernelPreprocessor, KernelProcessor

class CASystem:
    """Handles processing and bookkeeping of a simulation step function and the required kernels."""

    step_fn_desc: StepFunctionDescriptor
    step_fn_meta: StepFuncMetadata

    kernels_meta: dict[str,KernelMetadata]
    kernels_processor: KernelProcessor

    req_consts: list[tuple[str,ast.expr]]

    nb_step_fn: Callable

    def __init__(self, step_fn: CASysStepFunc) -> None:
        self.step_fn_desc = (desc := StepFunctionDescriptor(step_fn))
        self.step_fn_meta = StepFuncPreprocessor().preprocess(desc)

        header(f'[preprocessing kernels]')
        self.preprocess_kernels()
        header(f'[Processing kernels]')
        self.process_kernels()

        header(f'[Creating final step function]')
        self.create_step_function()

    def preprocess_kernels(self) -> None:
        kernel_nspace = self.step_fn_meta.kernel_nspace
        preprocessor = KernelPreprocessor()

        self.kernels_meta = {
            name:preprocessor.preprocess(KernelDescriptor(kernel))
            for name,kernel in kernel_nspace.items()
        }

    def process_kernels(self) -> None:
        self.kernels_processor = (processor := KernelProcessor())

        self.req_consts = []
        for meta in self.kernels_meta.values():
            header(f'[Processing kernel: {meta.fndef.name}]')
            processor.process(meta)
            self.req_consts += [c for c in meta.consts if c not in self.req_consts]    

        processor.jit_kernels()

    def create_step_function(self):
        step_fn_meta = self.step_fn_meta
        assert step_fn_meta.soa.output_schema is not None

        fndef = self.step_fn_meta.fndef
        arg_names = [
            *[b[0] for b in step_fn_meta.soa.output_schema],
            *[c for c,_ in self.req_consts]
        ]
        step_fn_args = ast.arguments( args=[
            *[ast.arg(n) for n in [
                KV_TIMESTAMP,
                KV_WIDTH, KV_HEIGHT,
                KV_LD_IDX, KV_WR_IDX,
            ]],
            *[ast.arg(n) for n in arg_names]
        ])

        # —————————————————————————————————————————————
        # Build `nb_step_fn` with one nested prange‐loop per call‐group
        # —————————————————————————————————————————————
        fn_def = ast.FunctionDef(
            name='nb_step_fn',
            args=step_fn_args,
            body=[],
            decorator_list=[]
        )

        for group in step_fn_meta.calls:
            # outer loop: for px in prange(width)
            px_for = ast.For(
                target=ast.Name(KV_PX, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Attribute(value=ast.Name('nb', ctx=ast.Load()),
                                       attr='prange', ctx=ast.Load()),
                    args=[ast.Name(KV_WIDTH, ctx=ast.Load())],
                    keywords=[]
                ),
                body=[],
                orelse=[]
            )
            # inner loop: for py in prange(height)
            py_for = ast.For(
                target=ast.Name(KV_PY, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Attribute(value=ast.Name('nb', ctx=ast.Load()),
                                       attr='prange', ctx=ast.Load()),
                    args=[ast.Name(KV_HEIGHT, ctx=ast.Load())],
                    keywords=[]
                ),
                body=[],
                orelse=[]
            )

            # append each kernel‐call in this group
            for kernel_name, call_soa in group:
                # buffer args (from output_schema)
                buf_args = [
                    ast.Name(fld, ctx=ast.Load())
                    for fld, _ in call_soa.output_schema
                ]
                # extra args (ld_idx, wr_idx, constants…)
                extra = [
                    ast.Name(arg.arg, ctx=ast.Load())
                    for arg in self.kernels_processor
                                       .processed_kernels[kernel_name]
                                       .args.args[len(call_soa.output_schema):]
                ]
                call_node = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(kernel_name, ctx=ast.Load()),
                        args=buf_args + extra,
                        keywords=[]
                    )
                )
                py_for.body.append(call_node)

            px_for.body.append(py_for)
            fn_def.body.append(px_for)

        ast.fix_missing_locations(fn_def)
        module = ast.Module(body=[fn_def], type_ignores=[])
        src = ast.unparse(module)
        nspace = {
            **self.step_fn_desc.globals,
            **self.kernels_processor.compiled_kernels,
            'nb': nb,
        }

        header('[Final step function finished]')
        preview(txt=src)

        exec(src, nspace)
        self.nb_step_fn = nb.jit(nspace['nb_step_fn'], nopython=True, nogil=True, parallel=True, fastmath=True)

from typing import Sequence, Any
import numpy as np

class CASim:
    """
    Handles CA simulation and buffer management, including user-supplied constants.

    :param system: CASystem containing the compiled step function and metadata.
    :param dims: Tuple[int, ...] of the simulation grid dimensions.
    :param consts: An object whose attributes match each constant name required by the system.
    """
    system: CASystem
    dims: tuple[int, int]
    consts: object
    _const_names: list[str]
    _const_vals: list[Any]

    ld_idx: int = 0
    wr_idx: int = 1
    buffers: dict[str, np.ndarray]
    timestamp: int = 0

    def __init__(self, system: CASystem, dims: tuple[int, int], consts: object | None = None) -> None:
        """
        Initialize simulation buffers for a given CASystem and cache constants.

        :param system: CASystem containing the step function and required metadata.
        :param dims: Grid dimensions, e.g. (width, height).
        :param consts: Class or instance providing attributes for each name in system.req_consts.
        """
        self.system = system
        self.dims = dims
        self.consts = consts

        if self.system.req_consts and consts is None:
            raise ValueError("CASim received CASystem with one or more required constants but no consts object was given.")

        # Extract constant names in the order the step-fn expects :contentReference[oaicite:0]{index=0}
        self._const_names = [name for name, _ in self.system.req_consts]


        # Read & cache their values once
        self._cache_consts()

        assert self.system.step_fn_meta.soa.output_schema is not None

        # Create a double buffer (2 × width × height) for each field
        self.buffers = {}
        for name, dtype in self.system.step_fn_meta.soa.output_schema:
            self.buffers[name] = np.zeros((2, *dims), dtype=dtype)

        for buffer_name, cact in self.system.step_fn_meta.soa.input_schema:
            for k,v in cact.fields.items():
                if default_val := v.default_value is not None:
                    field_buffer = self.system.step_fn_meta.soa.cvt(buffer_name,k)
                    self.buffers[field_buffer][:,:,:] = default_val

    def _cache_consts(self) -> None:
        """
        Read constant values from self.consts and store them for fast access.

        :raises AttributeError: if a required constant is missing.
        """
        vals: list[Any] = []
        for name in self._const_names:
            if not hasattr(self.consts, name):
                raise AttributeError(f"Missing constant '{name}' on {self.consts!r}")
            vals.append(getattr(self.consts, name))
        self._const_vals = vals

    def refresh_consts(self) -> None:
        """
        Public API to re-cache constants (e.g. if you’ve mutated self.consts).
        """
        self._cache_consts()

    def step(self) -> None:
        """Perform one CA step: run the JIT step-fn and swap read/write buffers."""

        w, h = self.dims
        buf_list = [self.buffers[name] for name, _ in self.system.step_fn_meta.soa.output_schema] # type: ignore
        # Pass width, height, ld_idx, wr_idx, all buffers, then the cached consts
        for buf in self.buffers.values():
            np.copyto(buf[self.wr_idx], buf[self.ld_idx])

        self.timestamp+=1

        self.system.nb_step_fn(
            self.timestamp,
            w, h,
            self.ld_idx, self.wr_idx,
            *buf_list,
            *self._const_vals
        )
        # Swap for next iteration
        self.ld_idx, self.wr_idx = self.wr_idx, self.ld_idx

    def load_state(self, t: int, ld_idx: int, buffers_snapshot: dict[str, np.ndarray]):
        """load_state loads a given state snapshot and timestamp.

        Args:
            t (int): Timestamp of the latest buffer (idx: 0)
            ld_idx: which double buffer side is the latest
            buffers_snapshot (dict[str, np.ndarray]): The double-buffered state snapshot to load
        """
        self.ld_idx, self.wr_idx = ld_idx, 1-ld_idx
        self.timestamp = t
        self.buffers = buffers_snapshot


    def edit_cells(
        self,
        buffer_idx: int,
        edits: Sequence[tuple[int, int, tuple[tuple[str, Any], ...]]]
    ) -> None:
        """
        Apply ad-hoc edits to individual cells in one of the two buffers.
        
        :param buffer_idx: 0 for the current read buffer, 1 for the write buffer.
        :param edits: A sequence of (x, y, ((field_name, value), …)) tuples.
        """
        for x, y, field_updates in edits:
            for field_buffer_name, value in field_updates:
                if field_buffer_name not in self.buffers:
                    raise ValueError(f"No field buffer called '{field_buffer_name}'")
                self.buffers[field_buffer_name][buffer_idx, x, y] = value
