from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


def _json_coerce(x: object) -> object:
    """Coerce common Python types to JSON-safe values.

    Args:
      x: Any Python object.

    Returns:
      A JSON-serializable object. Sets and tuples become lists.
    """
    if isinstance(x, dict):
        return {k: _json_coerce(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted(_json_coerce(v) for v in x) # type: ignore
    if isinstance(x, tuple):
        return [_json_coerce(v) for v in x]
    if isinstance(x, list):
        return [_json_coerce(v) for v in x]
    return x


@dataclass(frozen=True)
class MetadataKey[T]:
    """Typed metadata key.

    Attributes:
      namespace: Logical group, e.g. 'kernel', 'sys', 'step'. Empty means top-level.
      name: Field name, e.g. 'dims'.
      default: Default value used if no factory is provided.
      factory: Callable to lazily build a default value (e.g. set, dict).
      doc: Short description for debugging.
    """
    namespace: str
    name: str
    default: T | None = None
    factory: Callable[[], T] | None = None
    doc: str = ''

    def storage_key(self) -> str:
        """Return the actual dict key used by the store.

        Returns:
          When namespace is empty, returns name. Otherwise 'namespace:name'.
        """
        return self.name if self.namespace == '' else f'{self.namespace}:{self.name}'

class MetadataStore:
    """Dict-like metadata store with typed key support.

    Backwards compatible with string keys so legacy code keeps working.
    """

    def __init__(self, backing: dict[str, object] | None = None) -> None:
        self._data: dict[str, object] = dict(backing or {})

    def __getitem__(self, k: str | MetadataKey[object]) -> object:
        return self._data[self._sk(k)]

    def __setitem__(self, k: str | MetadataKey[object], v: object) -> None:
        self._data[self._sk(k)] = v

    def get[T](self, key: MetadataKey[T], fallback: T | None = None) -> T:
        """Get a value by typed key, materializing defaults if needed.

        Args:
          key: MetaKey for the value.
          fallback: Optional value used when key has no default.

        Returns:
          The value for the key, creating and storing a default if missing.
        """
        sk = key.storage_key()
        if sk in self._data:
            return self._data[sk]  # type: ignore[return-value]
        if key.factory is not None:
            v = key.factory()
        elif key.default is not None:
            v = key.default
        elif fallback is not None:
            v = fallback
        else:
            raise KeyError(f'Meta key missing and no default: {sk}')
        self._data[sk] = v
        return v  # type: ignore[return-value]

    def set[T](self, key: MetadataKey[T], value: T | None = None) -> None:
        """Set a value by typed key.

        Args:
          key: MetaKey for the value.
          value: The value to store.
        """
        if value is None:
            self._data[key.storage_key()] = key.default
        self._data[key.storage_key()] = value

    def update[T](self, key: MetadataKey[T], fn: Callable[[T], T]) -> T:
        """Update a value by applying a function.

        Args:
          key: MetaKey for the value.
          fn: Function from old value to new value.

        Returns:
          The updated value.
        """
        cur = self.get(key)
        new = fn(cur)
        self.set(key, new)
        return new

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe nested dict grouped by namespace."""
        out: dict[str, dict[str, object]] = {}
        for sk, v in self._data.items():
            if ':' in sk:
                ns, name = sk.split(':', 1)
            else:
                ns, name = '', sk
            group = out.setdefault(ns, {})
            group[name] = _json_coerce(v)
        return out # type: ignore

    def as_flat_dict(self) -> dict[str, object]:
        """Return the flat internal dict (legacy compatibility)."""
        return dict(self._data)

    @staticmethod
    def _sk(k: str | MetadataKey[object]) -> str:
        return k if isinstance(k, str) else k.storage_key()
