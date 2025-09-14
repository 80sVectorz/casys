from __future__ import annotations

from typing import TYPE_CHECKING, cast, Any, Literal, Sequence

if TYPE_CHECKING:
    from casys.core import CaSim

from dataclasses import dataclass
import dataclasses
import numpy as np

try:
    from numba import cuda
except Exception:
    cuda = None  # type: ignore

from casys.dsl._core.schema.base_components import FieldSchema
from casys.dsl._core.schema.virtual_bools import VirtualBoolField

@dataclass(slots=True)
class BuffersAccessor:
    sim: CaSim
    buffers: dict[str, np.ndarray]
    cache: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def clear_cache(self):
        self.cache.clear()

    def __getitem__(self, field: str | object) -> np.ndarray:
        if isinstance(field, str):
            field_name = field
        elif isinstance(field, FieldSchema):
            field_name = field.canonical_name()   
        else:
            raise ValueError(f"Received invalid field value '{field}'")

        sim = self.sim
        
        result = None

        if field_name in self.buffers:
            result = self.buffers[field_name]
        elif field_name in self.cache:
            result =  self.cache[field_name]
        elif field_name in sim.field_schemas_lut:
            field_schema = sim.field_schemas_lut[field_name]
            soa_field_name = field_schema.resolve_field().name

            match field_schema.__class__.__name__:
                case FieldSchema.__name__:
                    result = self.buffers[soa_field_name]

                case VirtualBoolField.__name__:
                    field_schema = cast(VirtualBoolField, field_schema)
                    
                    bit_idx = field_schema.bit_idx
                    result = (self.buffers[soa_field_name] >> bit_idx & np.uint8(0b1)).astype(np.uint8)

            self.cache[field_name] = result # type: ignore

        if result is None:
            raise ValueError(f"No buffer for key '{field}'")

        return result


# ------------------------------ CUDA kernels ------------------------------

if cuda is not None:
    @cuda.jit
    def k_gather_points2d(src, xs, ys, buf, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = src[buf, xs[i], ys[i]]

    @cuda.jit
    def k_gather_points2d_vbool(src, xs, ys, buf, bit_idx, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = (src[buf, xs[i], ys[i]] >> bit_idx) & 1

    @cuda.jit
    def k_scatter_points2d(dst, xs, ys, buf, vals):
        i = cuda.grid(1)
        if i < vals.size:
            dst[buf, xs[i], ys[i]] = vals[i]

    @cuda.jit
    def k_scatter_points2d_vbool(dst, xs, ys, buf, bit_idx, vals01):
        i = cuda.grid(1)
        if i < vals01.size:
            old = dst[buf, xs[i], ys[i]]
            if vals01[i] != 0:
                dst[buf, xs[i], ys[i]] = old | (1 << bit_idx)
            else:
                dst[buf, xs[i], ys[i]] = old & ~(1 << bit_idx)


@dataclass
class _PointCache:
    """Reusable device/host buffers for point I/O."""
    capacity: int
    dtype: Any
    d_xs: Any | None
    d_ys: Any | None
    d_out: Any | None
    h_out: np.ndarray
    d_vals: Any | None


class GpuIoManager:
    """
    Batched high-throughput reads/writes for scattered points and small ROIs.

    Usage:

        >>> sim.io = GpuIoManager(sim)  # once in CaSim.__init__`
        >>> vals = sim.io.gather_points(my_cact.my_field, xs, ys) # read
        >>> sim.io.scatter_points(my_cact.my_field, xs, ys, new_values) # write
        >>> spk = sim.io.gather_points(my_cact.my_vbool_field, xs, ys) # vbool read (0/1)
        >>> sim.io.scatter_points_vbool(my_cact.my_vbool_field, xs, ys, bits01) # vbool write
        >>> roi = sim.io.copy_roi(my_cact.my_vbool_field, x0, y0, w, h) # rectangular window

    Notes:
    - Default uses the CUDA default stream, which serializes sim step (safe).
    - After you wire a post-swap event, call set_stream(cuda.stream()) and
      mark_after_swap_event() each step to overlap probe I/O with compute.
    """

    def __init__(self, sim: Any, initial_capacity: int = 64, use_default_stream: bool = True) -> None:
        self.sim = sim
        self.W, self.H = sim.dims
        self._caches: dict[tuple[str, Any], _PointCache] = {}
        self._stream = None
        self._after_swap_event = None

        if cuda is not None and getattr(sim, '_dev_buffers', None) is not None and not use_default_stream:
            self._stream = cuda.stream()

        self._min_cap = max(16, initial_capacity)

    # ------------------------------ stream control ------------------------------

    def set_stream(self, stream: Any | None) -> None:
        """Set the CUDA stream to use for I/O. None = default stream."""
        self._stream = stream

    def mark_after_swap_event(self) -> None:
        """Record an event on the default stream to order subsequent I/O."""
        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            return
        ev = cuda.event()
        ev.record()  # default stream
        self._after_swap_event = ev

    def _wait_after_swap(self) -> None:
        if self._after_swap_event is not None and self._stream is not None:
            self._stream.wait_event(self._after_swap_event)

    # --------------------------------- helpers ----------------------------------

    def _resolve_soa_and_bit(self, field: str | Any) -> tuple[str, int | None, Any]:
        fs = field
        if isinstance(field, str):
            if field in self.sim._buffers:
                return field, None, self.sim._buffers[field].dtype
            fs = self.sim.field_schemas_lut[field]
        base = fs.resolve_field()
        bit_idx = getattr(fs, 'bit_idx', None)
        dtype = self.sim._buffers[base.name].dtype
        return base.name, bit_idx, dtype

    def _get_or_grow_cache(self, key: tuple[str, Any], n: int, dtype: Any) -> _PointCache:
        cap = self._min_cap
        cache = self._caches.get(key)
        if cache is not None:
            if n <= cache.capacity and cache.dtype == dtype:
                return cache
            cap = cache.capacity
        new_cap = cap
        while new_cap < n:
            new_cap *= 2

        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            h_out = np.empty(new_cap, dtype=dtype)
            cache = _PointCache(new_cap, dtype, None, None, None, h_out, None)
            self._caches[key] = cache
            return cache

        stream = self._stream
        d_xs = cuda.device_array(new_cap, dtype=np.int32, stream=stream)
        d_ys = cuda.device_array(new_cap, dtype=np.int32, stream=stream)
        d_out = cuda.device_array(new_cap, dtype=dtype, stream=stream)
        h_out = cuda.pinned_array(new_cap, dtype=dtype)
        d_vals = cuda.device_array(new_cap, dtype=dtype, stream=stream)
        cache = _PointCache(new_cap, dtype, d_xs, d_ys, d_out, h_out, d_vals)
        self._caches[key] = cache
        return cache

    # --------------------------------- reads ------------------------------------

    def gather_points(
        self,
        field: str | Any,
        xs: np.ndarray | Sequence[int],
        ys: np.ndarray | Sequence[int],
        buf: int = 0,
    ) -> np.ndarray:
        """Gather values at scattered points.

        Args:
            field: Field name or schema. Virtual-bool returns 0/1.
            xs: x positions.
            ys: y positions.
            buf: 0 read buffer, 1 write buffer.

        Returns:
            1D array of values in probe order.
        """
        soa, bit_idx, dtype = self._resolve_soa_and_bit(field)
        xs = np.asarray(xs, dtype=np.int32) % self.W
        ys = np.asarray(ys, dtype=np.int32) % self.H
        n = xs.size

        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            arr = self.sim._buffers[soa][buf]
            if bit_idx is None:
                return arr[xs, ys].copy()
            vals = (arr[xs, ys].astype(np.int64) >> int(bit_idx)) & 1
            return vals.astype(np.uint8)

        out_dtype = np.uint8 if bit_idx is not None else dtype
        cache = self._get_or_grow_cache((f'R:{soa}:{bit_idx}', out_dtype), n, out_dtype)
        stream = self._stream
        self._wait_after_swap()

        cuda.to_device(xs, to=cache.d_xs[:n], stream=stream)
        cuda.to_device(ys, to=cache.d_ys[:n], stream=stream)

        threads = 128
        blocks = (n + threads - 1) // threads
        d_src = self.sim._dev_buffers[soa]
        if bit_idx is None:
            k_gather_points2d[blocks, threads, stream](d_src, cache.d_xs, cache.d_ys, buf, cache.d_out)
        else:
            k_gather_points2d_vbool[blocks, threads, stream](d_src, cache.d_xs, cache.d_ys, buf, int(bit_idx), cache.d_out)

        cache.d_out[:n].copy_to_host(cache.h_out[:n], stream=stream)
        if stream is not None:
            stream.synchronize()
        return cache.h_out[:n].copy()

    # --------------------------------- writes -----------------------------------

    def scatter_points(
        self,
        field: str | Any,
        xs: np.ndarray | Sequence[int],
        ys: np.ndarray | Sequence[int],
        vals: np.ndarray | Sequence[Any],
        buf: int = 0,
        mode: Literal['set'] = 'set',
    ) -> None:
        """Scatter numeric values at scattered points.

        Args:
            field: Field name or schema (non-virtual-bool).
            xs: x positions.
            ys: y positions.
            vals: values to write.
            buf: 0 read buffer, 1 write buffer.
            mode: only 'set' for now.

        Returns:
            None.
        """
        if mode != 'set':
            raise NotImplementedError('Only mode="set" is implemented.')

        soa, bit_idx, dtype = self._resolve_soa_and_bit(field)
        if bit_idx is not None:
            raise ValueError('Use scatter_points_vbool for virtual-bool fields.')

        xs = np.asarray(xs, dtype=np.int32) % self.W
        ys = np.asarray(ys, dtype=np.int32) % self.H
        vals_arr = np.asarray(vals, dtype=dtype)
        n = xs.size

        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            self.sim._buffers[soa][buf, xs, ys] = vals_arr
            return

        cache = self._get_or_grow_cache((f'W:{soa}', dtype), n, dtype)
        stream = self._stream
        self._wait_after_swap()

        cuda.to_device(xs, to=cache.d_xs[:n], stream=stream)
        cuda.to_device(ys, to=cache.d_ys[:n], stream=stream)
        cuda.to_device(vals_arr, to=cache.d_vals[:n], stream=stream)

        threads = 128
        blocks = (n + threads - 1) // threads
        d_dst = self.sim._dev_buffers[soa]
        k_scatter_points2d[blocks, threads, stream](d_dst, cache.d_xs, cache.d_ys, buf, cache.d_vals)

        if stream is not None:
            stream.synchronize()

    def scatter_points_vbool(
        self,
        field: str | Any,
        xs: np.ndarray | Sequence[int],
        ys: np.ndarray | Sequence[int],
        bits01: np.ndarray | Sequence[int],
        buf: int = 0,
    ) -> None:
        """Scatter write for virtual-bool fields. bits01 must be 0 or 1.

        Args:
            field: Virtual-bool schema or canonical name.
            xs: x positions.
            ys: y positions.
            bits01: 0 clears, 1 sets the bit.
            buf: 0 read buffer, 1 write buffer.

        Returns:
            None.
        """
        soa, bit_idx, dtype = self._resolve_soa_and_bit(field)
        if bit_idx is None:
            raise ValueError('scatter_points_vbool requires a virtual-bool field.')

        xs = np.asarray(xs, dtype=np.int32) % self.W
        ys = np.asarray(ys, dtype=np.int32) % self.H
        bits01 = np.asarray(bits01, dtype=np.uint8)
        n = xs.size

        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            host = self.sim._buffers[soa]
            b = int(bit_idx)
            for i in range(n):
                x = xs[i]
                y = ys[i]
                old = int(host[buf, x, y])
                if bits01[i] != 0:
                    host[buf, x, y] = old | (1 << b)
                else:
                    host[buf, x, y] = old & ~(1 << b)
            return

        cache = self._get_or_grow_cache((f'Wv:{soa}:{bit_idx}', np.uint8), n, np.uint8)
        stream = self._stream
        self._wait_after_swap()

        cuda.to_device(xs, to=cache.d_xs[:n], stream=stream)
        cuda.to_device(ys, to=cache.d_ys[:n], stream=stream)
        cuda.to_device(bits01, to=cache.d_vals[:n], stream=stream)

        threads = 128
        blocks = (n + threads - 1) // threads
        d_dst = self.sim._dev_buffers[soa]
        k_scatter_points2d_vbool[blocks, threads, stream](d_dst, cache.d_xs, cache.d_ys, buf, int(bit_idx), cache.d_vals)

        if stream is not None:
            stream.synchronize()

    # ---------------------------------- ROI -------------------------------------

    def copy_roi(
        self,
        field: str | Any,
        x0: int,
        y0: int,
        w: int,
        h: int,
        buf: int = 0,
    ) -> np.ndarray:
        """Copy a rectangular ROI into a compact host array.

        Args:
            field: Field name or schema. Virtual-bool returns 0/1.
            x0: left index (wrapped).
            y0: top index (wrapped).
            w: width.
            h: height.
            buf: 0 read buffer, 1 write buffer.

        Returns:
            ndarray of shape (w, h).
        """
        soa, bit_idx, dtype = self._resolve_soa_and_bit(field)
        W, H = self.W, self.H
        x0 %= W
        y0 %= H
        x1 = x0 + w
        y1 = y0 + h

        host = self.sim._buffers[soa]
        if cuda is None or getattr(self.sim, '_dev_buffers', None) is None:
            out = np.empty((w, h), dtype=host.dtype)
            for dx in range(w):
                for dy in range(h):
                    out[dx, dy] = host[buf, (x0 + dx) % W, (y0 + dy) % H]
            if bit_idx is None:
                return out
            return ((out.astype(np.int64) >> int(bit_idx)) & 1).astype(np.uint8)

        d = self.sim._dev_buffers[soa]

        def _copy_block(xa: int, xb: int, ya: int, yb: int) -> None:
            d_view = d[buf, xa:xb, ya:yb]
            h_view = host[buf, xa:xb, ya:yb]
            d_view.copy_to_host(h_view, stream=self._stream)

        # up to 4 blocks due to torus wrap
        if x1 <= W and y1 <= H:
            _copy_block(x0, x1, y0, y1)
        elif x1 > W and y1 <= H:
            _copy_block(x0, W, y0, y1)
            _copy_block(0, x1 % W, y0, y1)
        elif x1 <= W and y1 > H:
            _copy_block(x0, x1, y0, H)
            _copy_block(x0, x1, 0, y1 % H)
        else:
            _copy_block(x0, W, y0, H)
            _copy_block(0, x1 % W, y0, H)
            _copy_block(x0, W, 0, y1 % H)
            _copy_block(0, x1 % W, 0, y1 % H)

        if self._stream is not None:
            self._stream.synchronize()

        out = np.empty((w, h), dtype=host.dtype)
        for dx in range(w):
            for dy in range(h):
                out[dx, dy] = host[buf, (x0 + dx) % W, (y0 + dy) % H]
        if bit_idx is None:
            return out
        return ((out.astype(np.int64) >> int(bit_idx)) & 1).astype(np.uint8)
