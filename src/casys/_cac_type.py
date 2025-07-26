from __future__ import annotations
import inspect
from casys.logging import log_debug, DefinitionException, display_error
from dataclasses import _MISSING_TYPE, Field, dataclass, fields, is_dataclass
from typing import Literal, Protocol, get_args, cast, Any, Type
import numpy as np

@dataclass(frozen=True)
class CACTFieldType:
    """ Holds info about a CACTypeField's type"""

    true_type: np.generic
    dummy_type: Type[Any]

    @classmethod
    def from_dclass_field(cls: Type[CACTFieldType], field: Field) -> CACTFieldType:
        """ Create CACTFieldType based on dataclass Field object

        Args:
            field (Field): The dataclass Field object

        Raises:
            TypeError: Field type isn't a `cact_field[T_true,T_dummy]`
            TypeError: Type annotation is missing arguments
            TypeError: Invalid type for `cact_field` `T_true`

        Returns:
            CACTFieldType:
        """
        # Validation
        if field.type.__name__ != 'cact_field':
            raise TypeError(f'Invalid type for CACType field `{field.name}`. CACType fields must use the `cact_field` dummy type')

        t_args = get_args(field.type)
        if len(t_args) != 2:
            raise TypeError(f'Missing arguments for `cact_field` annotation of CACType field `{field.name}`. `cact_field` requires 2 arguments')

        true_type, dummy_type = t_args
        print(true_type)
        if not any(issubclass(true_type,T) for T in [np.generic,]):
            raise TypeError(f'Invalid type for `cact_field` `T_true` argument for CACType field `{field.name}`. `T_true` must be an np.generic type')

        return cls(true_type,dummy_type)

@dataclass(frozen=True)
class CACTypeField:
    name: str
    field_type: CACTFieldType
    default_value: Any | None
    parent: CACType

    @classmethod
    def from_dclass_field(cls: Type[CACTypeField], field: Field, cact: CACType, default: Any | None = None) -> CACTypeField:
        fld_type = CACTFieldType.from_dclass_field(field)

        if default is not None:
            # Validate default value's type
            if not any(issubclass(default,T) for T in [fld_type.true_type,fld_type.dummy_type]):
                raise TypeError(f"Invalid default value for field `{field.name}`. Default value must match a field's T_true or T_dummy type")

        return cls(
            name=field.name, 
            field_type=fld_type,
            default_value=default,
            parent=cact
        )

class CACType[T]:
    """
    Holds CAC Type definitions.  
    Also includes dummy magic methods to appease the type checker when writing kernel function code.  
    """

    dclass: T
    fields: dict[str, CACTypeField]

    def __init__(self, dclass: T, cls: object) -> None:
        if not is_dataclass(dclass):
            raise TypeError("cac_type must decorate a dataclass")

        self.dclass = dclass

        dclass_fields = fields(dclass)
        defaults: dict[str, Any] = {
            fld.name[1:]: fld.default
            for fld in dclass_fields
            if fld.name.startswith('_') and not isinstance(fld.default, _MISSING_TYPE)
        }

        try:
            self.fields = {
                fld.name: CACTypeField.from_dclass_field(
                    field=fld,
                    cact=self,
                    default=defaults.get(f'_{fld.name}')
                )
                for fld in dclass_fields
                if not fld.name.startswith('_')
            }
        except DefinitionException as e:
            display_error(err=DefinitionException(
                f'\nException while creating CACType `{dclass.__name__}`\n{e.message}' # type: ignore
            ))
            exit(1)

    def __getitem__(self, idx: int | tuple[int | None, ...]) -> Any:
        return self  # simulate access to a specific cell

    def __getattr__(self, name: str) -> CACTypeField:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(f"{name} not found in CACType")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"dclass", "fields"}:
            super().__setattr__(name, value)
        pass

    def __repr__(self) -> str:
        return f'<CACType: {self.dclass.__qualname__}>'
    
class _FieldProto[G,T](Protocol):
    """Minimal API that kernel code expects from every cact_field."""
    def __getitem__(self,
                    idx: int | tuple[int, int] | tuple[int,int,int]) -> T: ...
    def __setitem__(self,
                    idx: int | tuple[int, int] | tuple[int,int,int],
                    value: Any) -> None: ...
    def __bool__(self) -> bool: ...

    # comparisons such as `leaders.steps_left <= 0`
    def __le__(self, other: T) -> bool: ...
    def __lt__(self, other: T) -> bool: ...
    def __ge__(self, other: T) -> bool: ...
    def __gt__(self, other: T) -> bool: ...

    def __sub__(self,other: T) -> T: ...
    def __add__(self,other: T) -> T: ...
    def __mul__(self,other: T) -> T: ...
    def __div__(self,other: T) -> T: ...
    def __mod__(self,other: T) -> T: ...

type cact_field[T_true: np.generic, T_dummy: Any] = _FieldProto[T_true,T_dummy]