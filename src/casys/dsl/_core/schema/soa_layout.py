from __future__ import annotations
from typing import TYPE_CHECKING, Any

from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np
    from casys.dsl._core.schema.schema_base import Schema

@dataclass(frozen=True, slots=True)
class SoaField:
    name: str
    data_type: type[np.generic]
    default_value: Any = None

@dataclass(frozen=True, slots=True)
class SoaLayout:
    fields: dict[str,SoaField]