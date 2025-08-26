from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Type
import numpy as np
from casys.dsl._core.descriptors import CactBufferDescriptor

if TYPE_CHECKING:
    from casys.dsl._core.ca_system import CaSystem

from casys.dsl._core.kernel_values import KV_WR_IDX

from casys.wrappers import CaSimConstants, DefaultCaSimConstants

from .dsl._core.transpilers import BaseTranspiler

class CaSim:
    """
    Handles CA simulation and buffer management, including user-supplied constants.

    :param system: CASystem containing the compiled step function and metadata.
    :param dims: Tuple[int, ...] of the simulation grid dimensions.
    """
    system: CaSystem
    dims: Sequence[int]
    consts: object

    buffer_descriptors: dict[str, CactBufferDescriptor]
    buffers: dict[str, np.ndarray]

    dedicated_idx_ids: dict[str, str]   
    idx_lut: dict[str, int]
    wr_indices: Sequence[int]

    buffer_args: list[np.ndarray]

    timestamp: int = 0

    def __init__(self, system: CaSystem) -> None:
        """
        Initialize simulation buffers for a given CASystem and cache constants.

        :param system: CASystem containing the step function and required metadata.
        :param dims: Grid dimensions, e.g. (width, height).
        """

        self.system = system
        self.consts = system.sim_constants
        self.dims = system.sim_constants.dims
        self.buffer_descriptors = system.step_func.buffers
        self.dedicated_idx_ids = system.dedicated_idx_ids.copy()
        self.idx_lut = {KV_WR_IDX:0}

        # Create a double buffer (2 * width (ax-0) * height (ax-1) * depth (ax-2), etc) for each field
        self.buffers = {}
        for buffer,field in CactBufferDescriptor.get_soa_pairs_multi(self.buffer_descriptors.values()):
            field_obj = self.buffer_descriptors[buffer].cact._fields[field]
            dtype = field_obj.field_type.true_type
            fill_value = field_obj.default_value if field_obj.default_value is not None else 0

            self.buffers[f'{buffer}_{field}'] = np.full((2, *self.dims), fill_value, dtype=dtype)

            dedicated_idx_id = system.dedicated_idx_ids.get(buffer)
            if dedicated_idx_id:
                self.dedicated_idx_ids[f'{buffer}_{field}'] = dedicated_idx_id
                self.idx_lut[f'{buffer}_{field}'] = list(system.dedicated_idx_ids.keys()).index(buffer)
            else:
                self.dedicated_idx_ids[f'{buffer}_{field}'] = KV_WR_IDX
                self.idx_lut[f'{buffer}_{field}'] = 0

        self.buffer_args = [self.buffers[b] for b in system.signature_buffers]

        self.wr_indices = [1 for _ in [KV_WR_IDX,*system.dedicated_idx_ids]]

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

    def step(self) -> None:
        """Perform one CA step"""
        
        self.timestamp+=1

        self.wr_indices = self.system.nb_step_func(
            *self.buffer_args,
            *self.wr_indices,
            self.timestamp,
        )


    def load_state(self, t: int, wr_indices: Sequence[int], buffers_snapshot: dict[str, np.ndarray]):
        """load_state loads a given state snapshot and timestamp.

        Args:
            t (int): Timestamp of the latest buffer (idx: 0)
            ld_idx: which double buffer side is the latest
            buffers_snapshot (dict[str, np.ndarray]): The double-buffered state snapshot to load
        """
        self.wr_indices = wr_indices
        self.timestamp = t
        self.buffers = buffers_snapshot
        self.buffer_args = [b for n,b in self.buffers.items() if n in self.system.signature_buffers]

    def edit_cells(
        self,
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
                idx = self.wr_indices[self.idx_lut[field_buffer_name]]
                self.buffers[field_buffer_name][1^idx, x, y] = value
