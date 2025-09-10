from dataclasses import dataclass, is_dataclass
from typing import Any, Callable, Sequence, dataclass_transform

from .spec.cac_type import CaCellTypeSpec
from .spec.step_func import SimStepFunc
from .spec.ca_kernel import CaKernel

from .spec.world_spec import WorldSpec, LayerSymbol, BoundWorldInfo, _BOUND_ATTR

# -- User facing wrappers -- #

@dataclass_transform()
def cac_type[T](cls: T) -> T:
    dclass = cls
    if not is_dataclass(dclass):
        dclass = dataclass(slots=True)(cls) # type: ignore
    cls.__cac_type__ = CaCellTypeSpec(dclass,cls) # type: ignore
    return cls

def ca_kernel(fn: Callable[..., None]) -> CaKernel:
    return CaKernel.from_func(fn)

def _attach_worldspec_binding(fn: Callable[..., Any], world: WorldSpec) -> None:
    """Attach WorldSpec binding metadata to a zero-arg step function."""

    if fn.__code__.co_argcount != 0:
        raise TypeError('Step function must have zero parameters when using world=WorldSpec')

    layers = world.as_name_map()
    gbls: dict[str, Any] = fn.__globals__

    symbol_names = {
        name:s
        for name,s in layers.items()
        if gbls.get(name) is s
    } # type: ignore

    if not symbol_names:
        raise ValueError('No LayerSymbols from the provided WorldSpec were found in step function globals')
    setattr(fn, _BOUND_ATTR, BoundWorldInfo(spec=world, symbol_names=symbol_names))


def ca_sys_step_func(*, world: WorldSpec | None = None, **opts: Any) -> Callable[[Callable[..., Any]], SimStepFunc]:
    """Decorator for step functions. Supports zero-arg steps when world is provided.

    Legacy behavior is preserved when world is None.
    """
    def _decorator(fn: Callable[..., Any]) -> SimStepFunc:
        if world is not None:
            _attach_worldspec_binding(fn, world)
        return _legacy_ca_sys_step_func_apply(fn)
    return _decorator

def _legacy_ca_sys_step_func_apply(fn: Callable[...,None]) -> SimStepFunc:
    return SimStepFunc.from_func(fn)

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