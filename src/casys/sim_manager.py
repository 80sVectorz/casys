from collections import deque
import copy
import threading
import time
from queue import Queue, Empty
from typing import Any, Callable, Optional

import numpy as np

from casys.core import CASim

class SimManager:
    """
    General-purpose manager for CASim instances, decoupled from other logic such as visualization.

    Provides start/pause/step controls, latest-state retrieval,
    and update subscriptions via dirty-rectangle maps.
    """

    sim: CASim
    timestep: float
    history_buffer: deque[tuple[int, int, dict[str,np.ndarray]]] # timestamp, ld_idx, buffers

    def __init__(
        self,
        sim: CASim,
        timestep: float = 0.0,
        history_buffer_len: int = 0
    ) -> None:
        """
        :param sim: the CASim to drive
        :param timestep: seconds between automatic steps (0 for no delay)
        :param history_buffer_len: Number of simulations snapshots to store. For rewinding
        """
        self.sim = sim
        self.timestep = timestep
        self.history_buffer = deque(maxlen=history_buffer_len)

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_queue: Queue[dict[str, list[tuple[int, int, int, int]]]] = Queue()
        self._callbacks: list[Callable[[dict[str, list[tuple[int, int, int, int]]]], Any]] = []

        # field names and simulation dimensions
        schema = self.sim.system.step_fn_meta.soa.output_schema or []
        self._fields: list[str] = [name for name, _ in schema]
        self._width, self._height = self.sim.dims


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
            self.history_buffer.append((self.sim.timestamp, self.sim.ld_idx, copy.deepcopy(self.sim.buffers)))

        self.sim.step()

    def step(self) -> None:
        """
        Perform exactly one simulation step and enqueue a full-frame dirty rect for each field.
        """
        if self._running.is_set(): return

        self._running.clear()
        # advance simulation
        self._sim_step()
        self._mark_dirty()
    
    def _mark_dirty(self):
        # mark entire grid dirty for each field
        dirty_map: dict[str, list[tuple[int, int, int, int]]] = {}
        full_rect = (0, 0, self._width,self._height)
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

        # if current_t == s1_t:
        #     s1_t, s1 = self.history_buffer.pop()
        # s2_t, s2 = self.history_buffer.pop()

        # snapshot = { k:np.stack([s1[k],s2[k]]) for k in s1 }
        self.sim.load_state(*self.history_buffer.pop())

        self._mark_dirty()

    def get_latest_update(
        self,
        block: bool = False,
        timeout: Optional[float] = None
    ) -> Optional[dict[str, list[tuple[int, int, int, int]]]]:
        """
        Retrieve the next update map of dirty rects.

        :param block: whether to block waiting for an update
        :param timeout: seconds to wait if blocking
        :returns: dict of field_name -> list of (x, y, width, height), or None
        """
        try:
            return self._update_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def get_current_state(self) -> dict[str, Any]:
        """
        Return the current read buffers for each field.

        :returns: mapping of field_name to 2D NumPy array
        """
        state: dict[str, Any] = {}
        idx = self.sim.ld_idx
        for field in self._fields:
            buf = self.sim.buffers[field]
            state[field] = buf[idx]
        return state

    def subscribe(
        self,
        callback: Callable[[dict[str, list[tuple[int, int, int, int]]]], Any]
    ) -> None:
        """
        Register a function to receive each dirty-map update.

        Callbacks run in the simulation thread context.
        """
        self._callbacks.append(callback)

    def _run_loop(self) -> None:
        """
        Internal loop: steps simulation while running, respecting timestep.
        """
        while True:
            if self._running.is_set():
                start = time.perf_counter()
                self._sim_step()
                # full-frame dirty each step
                dirty_map = {field: [(0, 0, self._width, self._height)] for field in self._fields}
                self._publish_update(dirty_map)
                if self.timestep > 0:
                    elapsed = time.perf_counter() - start
                    to_sleep = self.timestep - elapsed
                    if to_sleep > 0:
                        time.sleep(to_sleep)
            else:
                time.sleep(0.01)

    def _publish_update(
        self,
        dirty_map: dict[str, list[tuple[int, int, int, int]]]
    ) -> None:
        """
        Push update into queue and notify subscribers.
        """
        self._update_queue.put(dirty_map)
        for cb in list(self._callbacks):
            try:
                cb(dirty_map)
            except Exception:
                pass
