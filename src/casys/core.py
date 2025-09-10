from __future__ import annotations

from collections import deque
import dataclasses
from typing import TYPE_CHECKING, Any, Sequence, Type, cast
import numpy as np

if TYPE_CHECKING:
    from casys.dsl._core.ca_system import CaSystem
    from casys.dsl._core.schema.world_schema import WorldSchema
    from casys.dsl._core.schema.soa_layout import SoaLayout

from casys.spec.cac_type import CactField
from casys.dsl._core.schema.base_components import FieldSchema
from casys.wrappers import CaSimConstants, DefaultCaSimConstants, dataclass

from casys.spec.world_spec import get_bound_world_info
from casys.dsl._core.schema.virtual_bools import VirtualBoolField

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


        # Create a double buffer (2 * width (ax-0) * height (ax-1) * depth (ax-2), etc) for each SoA field
        self._buffers = {}
        for soa_field in self.soa_layout.fields.values():
            dtype = soa_field.data_type
            fill_value = soa_field.default_value if soa_field.default_value is not None else 0

            self._buffers[soa_field.name] = np.full((2, *self.dims), fill_value, dtype=dtype)

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
        self.timestamp+=1
        self.system.nb_step_func(
            *self._buffers.values(),
            *self.dims,
            self.timestamp,
        )
        self.buffers.clear_cache()

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
            soa_field = field_schema.resolve_field()
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