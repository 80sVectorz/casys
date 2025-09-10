from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence, Type


if TYPE_CHECKING:
    import numpy as np
    from casys.spec.cac_type import CactField
    from casys.wrappers import SimStepFunc, CaSimConstants
    from casys.dsl._core.schema.soa_layout import SoaLayout
    from casys.dsl._core.schema.world_schema import WorldSchema

@dataclass
class CaSystem:
    sim_constants: CaSimConstants | Type[CaSimConstants]
    step_func: SimStepFunc
    world_schema: WorldSchema
    soa_layout: SoaLayout
    nb_step_func: Callable[..., Sequence[int]]
    signature_buffers: list[str]