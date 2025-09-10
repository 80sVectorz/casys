from typing import Protocol, runtime_checkable, get_origin, get_args, TypeIs
import numpy as np

from casys.dsl._core.schema.base_components import FieldSchema
from casys.dsl._core.schema.virtual_bools import VirtualBoolField


@runtime_checkable
class VirtualType[T_true: np.generic, T_schema: FieldSchema](Protocol):
    ...

type virtual_type[T_true: np.generic, T_schema: FieldSchema] = VirtualType[T_true,T_schema]

vbool = virtual_type[np.uint8, VirtualBoolField]

def is_virtual_type_annotation(tp: object) -> TypeIs[virtual_type]:
    """Return True if 'tp' is a virtual_type[...] / VirtualType[...] annotation."""
    origin = get_origin(tp)
    if origin is not None:
        return origin is virtual_type
    # Handle bare, unsubscripted alias/protocol (unlikely but safe)
    return tp is VirtualType or tp is virtual_type

def unwrap_virtual_type(tp: object) -> tuple[type[np.generic], type[FieldSchema]]:
    """If 'tp' is virtual_type[T_true, T_schema], return (T_true, T_schema)."""
    if not is_virtual_type_annotation(tp):
        raise TypeError('Not a virtual_type[...] annotation')
    t_true, t_schema = get_args(tp)
    return t_true, t_schema
