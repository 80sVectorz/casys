from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence, Type


if TYPE_CHECKING:
    import numpy as np
    from casys._cac_type import CactField
    from casys.wrappers import SimStepFunc, CaSimConstants

@dataclass
class CaSystem:
    sim_constants: CaSimConstants | Type[CaSimConstants]
    step_func: SimStepFunc
    nb_step_func: Callable[..., Sequence[int]]
    signature_buffers: list[str]
    dedicated_idx_ids: dict[str,str]