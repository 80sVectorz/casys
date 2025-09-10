from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Callable
import inspect

from casys.dsl._core.schema.base_components import GroupSchema
from casys.spec.cac_type import CaCellTypeSpec

__all__ = [
    'layer',
    'LayerSymbol',
    'WorldSpec',
    'BoundWorldInfo',
    'get_bound_world_info',
    '_BOUND_ATTR',
]


_AUTO_COUNTER = count(1)
_BOUND_ATTR = '__casys_worldspec_binding__'


@dataclass(slots=True)
class LayerSymbol[T]:
    """Symbolic handle for a layer.

    Attributes:
        name: Bound later by WorldSpec.add(...) if not set.
        cac_type: The CAC type spec for this layer.
        schema: Concrete GroupSchema bound later by SimStepFunc.
    """
    cac_type: CaCellTypeSpec[T]
    name: str | None = None
    schema: GroupSchema | None = None

    def bind_schema(self, schema: GroupSchema) -> None:
        """Attach the concrete schema instance for this layer."""
        self.schema = schema

    def ensure_schema(self, factory: Callable[[CaCellTypeSpec[T]], GroupSchema]) -> GroupSchema:
        """Return schema, constructing it via factory if still missing."""
        if self.schema is None:
            self.schema = factory(self.cac_type)
        return self.schema

    def __getattribute__(self, name: str) -> Any:
        slots = object.__getattribute__(self, '__slots__')
        if name in slots:
            return object.__getattribute__(self, name)

        if self.schema is not None:
            if name in self.schema.fields:
                return self.schema.fields[name]

        return object.__getattribute__(self, name)


class _LayerFactory:
    """Factory supporting bracket syntax: layer[T]()."""

    @classmethod
    def __getitem__[T](cls, item: type[T]) -> Callable[[], T]:
        def _make() -> T:
            return LayerSymbol[T](cac_type=item)  # type: ignore[return-value]
        return _make


layer = _LayerFactory()

class WorldSpec:
    """Collection of LayerSymbols with implicit names."""
    def __init__(self) -> None:
        self._layers: list[LayerSymbol[object]] = []
        self._name_map: dict[str, LayerSymbol[object]] = {}

    def _infer_name(self, sym: LayerSymbol[object], caller_locals: dict[str, Any]) -> str:
        for n, v in caller_locals.items():
            if v is sym:
                return n
        # Fallback in rare cases where no binding exists in locals
        return f'layer_{next(_AUTO_COUNTER)}'

    def add(self, *layers: object) -> WorldSpec:
        """Add the desired layers to the WorldSpec.

        Despite what the type annotation says `*layers` only takes in `LayerSymbol` objects.
        """
        caller = inspect.currentframe().f_back  # type: ignore[assignment]
        caller_locals = caller.f_locals if caller is not None else {}

        for sym in layers:
            if not isinstance(sym, LayerSymbol):
                raise ValueError(f"Received non LayerSymbol object '{sym}'")

            if sym.name is None:
                sym.name = self._infer_name(sym, caller_locals)

            if sym.name in self._name_map and self._name_map[sym.name] is not sym:
                raise ValueError(f'Duplicate layer name: {sym.name}')

            self._layers.append(sym)
            self._name_map[sym.name] = sym

        return self

    def as_name_map(self) -> dict[str, LayerSymbol[object]]:
        return dict(self._name_map)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)


@dataclass(slots=True)
class BoundWorldInfo:
    """Binding metadata attached to the step function by the decorator."""
    spec: WorldSpec
    symbol_names: dict[str, LayerSymbol[object]]


def get_bound_world_info(fn: Any) -> BoundWorldInfo | None:
    """Return decorator-attached binding info or None."""
    return getattr(fn, _BOUND_ATTR, None)
