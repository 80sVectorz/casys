from collections import deque
import copy
import threading
import time
from queue import Full, Queue, Empty
from typing import Any, Callable, Optional, Sequence

import numpy as np

from casys.core import CaSim, BuffersAccessor

class SimManager:
    """
    General-purpose manager for CASim instances, decoupled from other logic such as visualization.

    Provides start/pause/step controls, latest-state retrieval,
    and update subscriptions via dirty-rectangle maps.
    """

    sim: CaSim
    timestep: float
    history_buffer: deque[tuple[int, dict[str,np.ndarray]]] # timestamp, ld_idx, buffers
    dims: Sequence[int]

    _buffers_accessor: BuffersAccessor

    def __init__(
        self,
        sim: CaSim,
        timestep: float = 0.0,
        history_buffer_len: int = 0,
        use_update_queue: bool = False,
        update_queue_maxsize: int = 1,
    ) -> None:
        """
        :param sim: the CASim to drive
        :param timestep: seconds between automatic steps (0 for no delay)
        :param history_buffer_len: Number of simulations snapshots to store. For rewinding
        """
        self.sim = sim
        self.dims = sim.dims
        self.timestep = timestep
        self.history_buffer = deque(maxlen=history_buffer_len)

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[dict[str, list[tuple[int,...]]]], Any]] = []
        self._state_lock = threading.RLock()

        self._use_update_queue = use_update_queue
        self._update_queue: Queue[dict[str, list[tuple[int,...]]]] = Queue(
            maxsize=update_queue_maxsize if use_update_queue else 0
        )

        self._fps_ema_alpha = 0.2
        self._fps_ema: float | None = None
        self._frame_dt_window: deque[float] = deque(maxlen=120)
        self._last_step_ms: float | None = None

        # field names
        self._fields: list[str] = [
            name for name in sim._buffers
        ]
        
        self._buffers_accessor = BuffersAccessor(self.sim, {})


    def start(self) -> None:
        """
        Begin continuous simulation stepping in a background thread.
        Subsequent calls have no effect if already running.
        """
        if self._thread and self._thread.is_alive():
            self._running.set()
            return

        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """
        Stop automatic stepping until resumed or stepped manually.
        """
        self._running.clear()

    def _sim_step(self):
        if self.history_buffer.maxlen:
            self.history_buffer.append((self.sim.timestamp, copy.deepcopy(self.sim._buffers))) # type: ignore

        self._buffers_accessor.clear_cache()
        self.sim.step()

    def step(self) -> None:
        """Perform exactly one simulation step and enqueue a full-frame dirty rect for each field."""
        if self._running.is_set():
            return

        step_start = time.perf_counter()
        self._sim_step()
        step_ms = (time.perf_counter() - step_start) * 1000.0
        self._record_timing(frame_dt=None, step_ms=step_ms)

        self._mark_dirty()

    
    def _mark_dirty(self):
        # mark entire grid dirty for each field
        dirty_map: dict[str, list[tuple[int,...]]] = {}
        full_rect = (*[0 for _ in self.dims], *self.dims)
        for field in self._fields:
            dirty_map[field] = [full_rect]
        self._publish_update(dirty_map)

    @property
    def paused(self) -> bool:
        return not self._running.is_set()


    def rewind(self) -> None:
        """
        Rewind 1 step
        """
        if self._running.is_set() or self.history_buffer.maxlen == 0: return

        if len(self.history_buffer) < 1: return

        self.sim.load_state(*self.history_buffer.pop())

        self._mark_dirty()

    def get_latest_update(
        self, 
        block: bool = False, 
        timeout: float | None = None
    ) -> dict[str, list[tuple[int,...]]] | None:
        """Retrieve next dirty-map, or None if none available."""
        if not self._use_update_queue:
            return None
        try:
            return self._update_queue.get(block, timeout)
        except Empty:
            return None

    def get_current_state(self) -> BuffersAccessor:
        """
        Return the current read buffers for each field.

        :returns: mapping of field_name to 2D NumPy array
        """
        state: dict[str, Any] = {}
        for field in self._fields:
            buf = self.sim.buffers[field]
            state[field] = buf[0]
        self._buffers_accessor.buffers = state

        return self._buffers_accessor

    def subscribe(
        self,
        callback: Callable[[dict[str, list[tuple[int,...]]]], Any]
    ) -> None:
        """
        Register a function to receive each dirty-map update.

        Callbacks run in the simulation thread context.
        """
        self._callbacks.append(callback)

    def _run_loop(self) -> None:
        """Internal loop: steps simulation while running, respecting timestep."""
        while True:
            if self._running.is_set():
                loop_start = time.perf_counter()

                step_start = loop_start
                self._sim_step()
                step_ms = (time.perf_counter() - step_start) * 1000.0

                # full-frame dirty each step
                if self._callbacks or self._use_update_queue:
                    dirty_map = {field: [(*[0 for _ in self.dims], *self.dims)] for field in self._fields}
                    self._publish_update(dirty_map)

                # honor timestep
                if self.timestep > 0:
                    elapsed = time.perf_counter() - loop_start
                    to_sleep = self.timestep - elapsed
                    if to_sleep > 0:
                        time.sleep(to_sleep)

                frame_dt = time.perf_counter() - loop_start
                self._record_timing(frame_dt=frame_dt, step_ms=step_ms)
            else:
                time.sleep(0.01)

    def _publish_update(
        self,
        dirty_map: dict[str, list[tuple[int,...]]]
    ) -> None:
        """
        Push update into queue and notify subscribers.
        """
        if self._use_update_queue:
            try:
                self._update_queue.put_nowait(dirty_map)
            except Full:
                # drop the frame
                pass

        for cb in list(self._callbacks):
            cb(dirty_map)


    def save_state(self, path: str, history_steps: int = 0) -> None:
        """Serialize current sim state and optional history to a compressed .npz."""
        was_running = self._running.is_set()
        if was_running:
            self.pause()

        # Snapshot under lock
        with self._state_lock:
            t = int(self.sim.timestamp)
            buffers = {name: np.array(arr, copy=True) for name, arr in self.sim._buffers.items()}

            # History slice
            hist_list = list(self.history_buffer)
            if history_steps > 0:
                hist_list = hist_list[-history_steps:]

            # Flatten history into serializable pieces
            history_meta = {
                'history_len': np.array(len(hist_list), dtype=np.int32),
                'history_maxlen': np.array(self.history_buffer.maxlen or 0, dtype=np.int32),
            }
            # Compose final dict of arrays for savez
            payload: dict[str, np.ndarray] = {
                't': np.array(t, dtype=np.int64),
                **{f'buffers/{k}': v for k, v in buffers.items()},
                **history_meta,
            }
            for i, (ht, hbufs) in enumerate(hist_list):
                payload[f'history/{i}/t'] = np.array(int(ht), dtype=np.int64)
                for bk, bv in hbufs.items():
                    payload[f'history/{i}/buffers/{bk}'] = np.array(bv, copy=True)

        # Disk I/O outside the lock
        if not path.lower().endswith('.npz'):
            path = path + '.npz'
        np.savez_compressed(path, **payload) # type: ignore

        if was_running:
            self.start()


    def load_state(self, path: str) -> None:
        """Load a previously saved .npz simulation snapshot from *path*."""
        was_running = self._running.is_set()
        if was_running:
            self.pause()

        # Read everything into memory first
        with np.load(path, allow_pickle=False) as npz:
            # Core snapshot
            t_arr = npz['t']
            t = int(t_arr.item() if t_arr.shape == () else t_arr[0])

            # Rebuild buffers dict: pick all keys that start with 'buffers/'
            buffers: dict[str, np.ndarray] = {}
            prefix = 'buffers/'
            for k in npz.files:
                if k.startswith(prefix):
                    name = k[len(prefix):]
                    buffers[name] = np.array(npz[k])

            # Optional history
            history_len = int(npz['history_len']) if 'history_len' in npz.files else 0
            history_maxlen = int(npz['history_maxlen']) if 'history_maxlen' in npz.files else 0
            history_items: list[tuple[int, dict[str, np.ndarray]]] = []
            for i in range(history_len):
                ht = int(npz[f'history/{i}/t'])
                hbufs: dict[str, np.ndarray] = {}
                hpref = f'history/{i}/buffers/'
                for k in npz.files:
                    if k.startswith(hpref):
                        name = k[len(hpref):]
                        hbufs[name] = np.array(npz[k])
                history_items.append((ht, hbufs))

        # Apply under lock
        with self._state_lock:
            self.sim.load_state(t, buffers)
            self.history_buffer = deque(history_items, maxlen=history_maxlen)

        self._mark_dirty()

        if was_running:
            self.start()

    # Performance and timing tracking related logic

    def _record_timing(self, frame_dt: float | None, step_ms: float) -> None:
        """Update timing metrics.

        Args:
            frame_dt: Wall-clock seconds for one full frame, including sleeps. If None, only step_ms is updated.
            step_ms: Milliseconds spent computing the step itself, excluding sleeps.
        """
        self._last_step_ms = float(step_ms)

        if frame_dt is None:
            return

        self._frame_dt_window.append(float(frame_dt))
        inst_fps = (1.0 / frame_dt) if frame_dt > 0 else float('inf')
        if self._fps_ema is None:
            self._fps_ema = inst_fps
        else:
            self._fps_ema = self._fps_ema + self._fps_ema_alpha * (inst_fps - self._fps_ema)

    @property
    def fps_ema(self) -> float | None:
        """Smoothed FPS using an exponential moving average."""
        return self._fps_ema

    @property
    def fps_avg(self) -> float | None:
        """Average FPS over the recent window."""
        if not self._frame_dt_window:
            return None
        avg_dt = sum(self._frame_dt_window) / len(self._frame_dt_window)
        return (1.0 / avg_dt) if avg_dt > 0 else float('inf')

    @property
    def last_step_ms(self) -> float | None:
        """Milliseconds spent computing the last step (no sleep)."""
        return self._last_step_ms

    def reset_fps_stats(self) -> None:
        """Clear accumulated FPS statistics."""
        self._fps_ema = None
        self._frame_dt_window.clear()
        self._last_step_ms = None