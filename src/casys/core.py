from __future__ import annotations

from collections import deque
import logging
from typing import TYPE_CHECKING, Any, Sequence, Type, cast

try:
    from numba import cuda
    cuda_lg = logging.getLogger('numba.cuda.cudadrv.driver')
    cuda_lg.setLevel(logging.WARNING)
    cuda_lg.propagate = False
except:
    cuda = None

import numpy as np
from functools import lru_cache

if TYPE_CHECKING:
    from casys.dsl._core.ca_system import CaSystem
    from casys.dsl._core.schema.world_schema import WorldSchema
    from casys.dsl._core.schema.soa_layout import SoaLayout

from casys.buffer_access_handling import BuffersAccessor, GpuIoManager
from casys.config import CASYS_CONFIG
from casys.spec.cac_type import CactField
from casys.dsl._core.schema.base_components import FieldSchema
from casys.wrappers import CaSimConstants, DefaultCaSimConstants, dataclass

from casys.dsl._core.schema.virtual_bools import VirtualBoolField

from .dsl._core.transpilers import BaseTranspiler

class CaSim:
    """
    Handles CA simulation and buffer management, including user-supplied constants.
    """

    system: CaSystem
    dims: Sequence[int]
    consts: object

    world_schema: WorldSchema
    soa_layout: SoaLayout

    field_schemas_lut: dict[str, FieldSchema]

    _buffers: dict[str, np.ndarray]
    _buffers_accessor: BuffersAccessor

    _edit_queue: deque[tuple[int, int, tuple[tuple[str, Any], ...]]]

    timestamp: int = 0

    def __init__(self, system: CaSystem) -> None:
        """
        Initialize simulation buffers for a given CASystem and cache constants.

        :param system: CASystem containing the step function and required metadata.
        :param dims: Grid dimensions, e.g. (width, height).
        """

        self.system = system
        self.world_schema = system.world_schema
        self.soa_layout = system.soa_layout
        self.consts = system.sim_constants
        self.dims = system.sim_constants.dims

        self._edit_queue = deque()
        self.io = GpuIoManager(self, initial_capacity=64, use_default_stream=True)

        # Create a double buffer (2 * width (ax-0) * height (ax-1) * depth (ax-2), etc) for each SoA field
        self._buffers = {}
        for soa_field in self.soa_layout.fields.values():
            dtype = soa_field.data_type
            fill_value = soa_field.default_value if soa_field.default_value is not None else 0

            self._buffers[soa_field.name] = np.full((2, *self.dims), fill_value, dtype=dtype)

        # -- GPU: device mirrors (same order as self._buffers) --
        self._dev_buffers: dict[str, object] | None = None
        if CASYS_CONFIG.backend == 'cuda':
            if cuda is None or not cuda.is_available():
                raise RuntimeError('CUDA backend selected but CUDA is not available.')
            self._dev_buffers = {name: cuda.to_device(arr) for name, arr in self._buffers.items()}

        self._buffers_accessor = BuffersAccessor(self, self._buffers)

        self.field_schemas_lut = { # type: ignore
            field.canonical_name():field
            for field in self.world_schema.get_flattened_tree(lambda n: isinstance(n, FieldSchema))
        }

    @classmethod
    def from_step_func(cls, step_func, sim_constants: CaSimConstants | Type[CaSimConstants] = DefaultCaSimConstants) -> CaSim:
        transpiler = BaseTranspiler(step_func, sim_constants)
        system = transpiler.transpile()

        return CaSim(system)

    # def _cache_consts(self) -> None:
    #     """
    #     Read constant values from self.consts and store them for fast access.

    #     :raises AttributeError: if a required constant is missing.
    #     """
    #     vals: list[Any] = []
    #     for name in self._const_names:
    #         if not hasattr(self.consts, name):
    #             raise AttributeError(f"Missing constant '{name}' on {self.consts!r}")
    #         vals.append(getattr(self.consts, name))
    #     self._const_vals = vals

    # def refresh_consts(self) -> None:
    #     """
    #     Public API to re-cache constants (e.g. if you’ve mutated self.consts).
    #     """
    #     self._cache_consts()

    @property
    def buffers(self) -> BuffersAccessor:
        return self._buffers_accessor

    def step(self) -> None:
        """Perform one CA step"""
        self._apply_pending_edits()
        self.timestamp += 1

        # Pick buffers based on backend; order matches insertion order in __init__
        bufs = self._dev_buffers if (CASYS_CONFIG.backend == 'cuda' and self._dev_buffers is not None) else self._buffers

        self.system.nb_step_func(
            *bufs.values(),
            *self.dims,
            self.timestamp,
        )

        self.io.mark_after_swap_event()
        self.buffers.clear_cache()

    def ensure_host_synced(self) -> None:
        """Copy device buffers back into host arrays when using the CUDA backend."""
        if self._dev_buffers is None:
            return
        for name, d_arr in self._dev_buffers.items():
            # d_arr: DeviceNDArray; self._buffers[name]: np.ndarray (2, H, W)
            d_arr.copy_to_host(self._buffers[name])
        # Keep accessor pointing at current host buffers and clear any cached derived views
        self._buffers_accessor.buffers = self._buffers
        self.buffers.clear_cache()

    def ensure_host_synced_for(self, fields: Sequence[str | object]) -> None:
        """Copy only the requested SoA fields (both buffers) device->host when on CUDA."""
        if self._dev_buffers is None:
            return
        # Resolve any virtual fields to their underlying SoA names
        to_copy: set[str] = set()
        for f in fields:
            if isinstance(f, str):
                if f in self._buffers:
                    to_copy.add(f)
                else:
                    to_copy.add(self.field_schemas_lut[f].resolve_field().name)
            elif isinstance(f, FieldSchema):
                to_copy.add(f.resolve_field().name)

        for name in to_copy:
            self._dev_buffers[name].copy_to_host(self._buffers[name])
        self._buffers_accessor.buffers = self._buffers
        self.buffers.clear_cache()

    def _resolve_field_for_device_read(self, field: str | FieldSchema) -> tuple[str, int | None]:
        """Resolve field name or FieldSchema to an SoA field name and optional vbool bit index.

        Args:
            field: Field canonical name or schema object.

        Returns:
            Tuple of (soa_field_name, bit_idx). bit_idx is None for non-virtual-bool fields.
        """
        if isinstance(field, str):
            if field in self._buffers:
                return field, None
            fs = self.field_schemas_lut[field]
        else:
            fs = field

        base = fs.resolve_field()
        bit_idx = getattr(fs, 'bit_idx', None)
        return base.name, bit_idx

    def _mirror_edit_to_device(self, soa_field_name: str, x: int, y: int) -> None:
        """Apply the just-written host values at (x, y) to CUDA device buffers.

        Works for arrays shaped (2, H, W). Avoids non-contiguous copies by launching
        a single-thread kernel that writes the two scalar values.
        """
        if self._dev_buffers is None:
            return

        d_arr = self._dev_buffers[soa_field_name]   # DeviceNDArray (2, H, W)
        h_arr = self._buffers[soa_field_name]       # np.ndarray     (2, H, W)

        # Read back the two scalar values we just wrote on host.
        v0 = h_arr[0, x, y]
        v1 = h_arr[1, x, y]

        k = _point_edit_kernel(np.dtype(h_arr.dtype).str)
        k[1, 1](d_arr, int(x), int(y), v0, v1)

    def _flush_edit_batch_to_device(self, soa_field_name: str, batch: list[tuple[int, int]]) -> None:
        """Batch-apply many host edits for a single SoA field to CUDA."""
        if self._dev_buffers is None or not batch:
            return

        d_arr = self._dev_buffers[soa_field_name]
        h_arr = self._buffers[soa_field_name]

        xs = np.fromiter((x for x, _ in batch), dtype=np.int32, count=len(batch))
        ys = np.fromiter((y for _, y in batch), dtype=np.int32, count=len(batch))
        v0s = h_arr[0, xs, ys].astype(h_arr.dtype, copy=False)
        v1s = h_arr[1, xs, ys].astype(h_arr.dtype, copy=False)

        k = _batched_edits_kernel(np.dtype(h_arr.dtype).str)
        n = xs.shape[0]
        threads = 128
        blocks = (n + threads - 1) // threads

        # Transfer small vectors once, write everything in one kernel.
        d_xs = cuda.to_device(xs)
        d_ys = cuda.to_device(ys)
        d_v0s = cuda.to_device(v0s)
        d_v1s = cuda.to_device(v1s)
        k[blocks, threads](d_arr, d_xs, d_ys, d_v0s, d_v1s, n)

    def load_state(self, t: int, buffers_snapshot: dict[str, np.ndarray]):
        """load_state loads a given state snapshot and timestamp.

        Args:
            t (int): Timestamp of the latest buffer (idx: 0)
            buffers_snapshot (dict[str, np.ndarray]): The double-buffered state snapshot to load
        """
        self.timestamp = t
        self._buffers = buffers_snapshot
        self._buffers_accessor.buffers = buffers_snapshot
        # self.buffer_args = [b for n,b in self.buffers.items() if n in self.system.signature_buffers]

    def _apply_pending_edits(self) -> None:
        while len(self._edit_queue):
            x, y, field_updates = self._edit_queue.pop()
            for field_name, value in field_updates:
                if field_name in self.soa_layout.fields:
                    self._buffers[field_name][:, x, y] = value
                    self._mirror_edit_to_device(field_name, x, y)
                else:
                    field_schema = self.field_schemas_lut[field_name]
                    soa_field = field_schema.resolve_field()
                    soa_field_name = field_schema.resolve_field().name

                    match field_schema.__class__.__name__:
                        case FieldSchema.__name__:
                            self._buffers[soa_field_name][:,x,y] = value

                        case VirtualBoolField.__name__:
                            field_schema = cast(VirtualBoolField, field_schema)
                            dt = soa_field.data_type
                            
                            value_u1 = dt(1 if value else 0)
                            bit_idx = field_schema.bit_idx
                            self._buffers[soa_field_name][:,x,y] = (
                                self._buffers[soa_field_name][:,x,y] 
                                & ~(dt(0b1) << bit_idx) # type: ignore
                                | value_u1 << bit_idx # type: ignore
                            )
                    self._mirror_edit_to_device(soa_field_name, x, y)

    def edit_cells(
        self,
        edits: Sequence[tuple[int, int, tuple[tuple[str | object, Any], ...]]],
    ) -> None:
        """
        Enqueue ad-hoc edits to individual cells.
        
        :param edits: A sequence of (x, y, ((field_name, value), …)) tuples.
        """
        dims = self.dims

        for x, y, input_field_updates in edits:
            field_updates = []
            
            for field, target_value in input_field_updates:
                if isinstance(field, str):
                    field_name = field
                elif isinstance(field, FieldSchema):
                    field_name = field.canonical_name()   
                else:
                    raise ValueError(f"Received invalid field value '{field}'")

                if (
                    field_name not in self._buffers
                    and field_name not in self.field_schemas_lut
                ):
                    raise ValueError(f"No SoA or virtual field called '{field}'")
                
                field_updates.append((field_name, target_value))

            self._edit_queue.append((x%dims[0], y%dims[1], tuple(field_updates)))

        if self.timestamp == 0:
            self._apply_pending_edits()


@lru_cache(maxsize=None)
def _point_edit_kernel(dtype_str: str):
    """Return a tiny kernel that writes both buffers at (x, y) for a given dtype."""
    @cuda.jit
    def _k(d_arr, x: int, y: int, v0, v1):
        d_arr[0, x, y] = v0
        d_arr[1, x, y] = v1
    return _k

@lru_cache(maxsize=None)
def _batched_edits_kernel(dtype_str: str):
    @cuda.jit
    def _k(d_arr, xs, ys, v0s, v1s, n: int):
        i = cuda.grid(1)
        if i < n:
            x = xs[i]
            y = ys[i]
            d_arr[0, x, y] = v0s[i]
            d_arr[1, x, y] = v1s[i]
    return _k