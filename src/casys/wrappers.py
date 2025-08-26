from dataclasses import dataclass, is_dataclass
from typing import Callable, Sequence

from ._cac_type import CaCellType
from ._step_func import SimStepFunc
from ._ca_kernel import CaKernel

# -- User facing wrappers -- #

def cac_type[T](cls: T) -> T:
    dclass = cls
    if not is_dataclass(dclass):
        dclass = dataclass(cls) # type: ignore
    cls.__cac_type__ = CaCellType(dclass,cls) # type: ignore
    return cls

def ca_kernel(func: Callable[..., None]) -> CaKernel:
    return CaKernel.from_func(func)

def ca_sys_step_func(func: Callable[...,None]) -> SimStepFunc:
    return SimStepFunc.from_func(func)

# -- Constants base class -- #

class CaSimConstants:
    """
     attributes:
        strict_kernels (bool): Ensure that kernels can only write to their own cell position.
        dims (Sequence[int]): The simulation grid dimensions
    """

    strict_kernels: bool
    dims: Sequence[int]

class DefaultCaSimConstants(CaSimConstants):
    """ Can be subclassed to specify custom constants """
    strict_kernels: bool = True
    dims: Sequence[int] = (2**6,2**6)