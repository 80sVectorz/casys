from operator import truediv
from numba.core.types import Object
import numpy as np
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast, Callable, Protocol, TYPE_CHECKING, get_args
from ._cac_type import CACType

# CACType wrapper components

def cac_type[T](cls: T) -> T:
    dclass = cls
    if not is_dataclass(dclass):
        dclass = dataclass(cls) # type: ignore
    cls.__cac_type__ = CACType(dclass,cls) # type: ignore
    return cls

# if TYPE_CHECKING:
#     # “pretend” each cact_field is an index-able grid object
#     type cact_field[G: np.generic, D: int] = _FieldProto[D]
# else:  # --------------------------------------------------------------
#     # At runtime keep the behaviour you already have
#     cact_field = cact_type            # same object CACType hands out

'''ReadOnlyBuffer
Denotes a buffer reference that is read only.
Instead of reading from the last buffer state it'll read from the current buffer state.
This is useful for multi state simulation step functions,
where one kernel function call might want to read from the new values written by the call before it.
'''
type ReadOnlyBuffer[T:CACType] = T

# CAKernel wrapper components

class CAKernel:
    func: Callable[..., None]

    def __init__(self, func: Callable[..., None]) -> None:
        self.func = func
    
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        pass


def ca_kernel(func: Callable[..., None]) -> CAKernel:
    return CAKernel(func)


# CASysStepFunc wrapper components

@dataclass(frozen=True)
class CASysStepFunc:
    func: Callable[...,None]

def ca_sys_step_func(func: Callable[...,None]) -> CASysStepFunc:
    return CASysStepFunc(func)