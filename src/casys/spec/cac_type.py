from __future__ import annotations

from casys.dsl._core.schema.base_components import FieldSchema, GroupSchema
from casys.logging import DefinitionException, display_error

from dataclasses import _MISSING_TYPE, Field, dataclass, fields, is_dataclass
from typing import Protocol, get_args, Any
from casys.spec.virtual_types import VirtualType, is_virtual_type_annotation, unwrap_virtual_type

from casys.spec.ca_layer_spec import CaLayerRef

import numpy as np

@dataclass(frozen=True)
class CactFieldType:
    """Holds info about a CAC-type field's type"""

    true_type: type[np.generic] | VirtualType
    dummy_type: type[Any]
    schema_type: type[FieldSchema]

    @classmethod
    def from_dclass_field(cls: type[CactFieldType], field: Field) -> CactFieldType:
        """Create CactFieldType based on dataclass Field object

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

        field_schema = FieldSchema

        is_virtual = is_virtual_type_annotation(true_type)
        if is_virtual:
            t_true, t_schema = unwrap_virtual_type(true_type)

            true_type: type[np.generic] = t_true
            field_schema = t_schema
        else:
            # must be a NumPy scalar class (e.g., np.uint8)
            if not isinstance(true_type, type) or not issubclass(true_type, np.generic):
                raise TypeError(f'Invalid type for `cact_field` `T_true` argument for CACType field `{field.name}`. `T_true` must be an np.generic type')

        return cls(true_type,dummy_type, field_schema)
    
@dataclass(frozen=True)
class CactField:
    name: str
    field_type: CactFieldType
    default_value: Any | None
    parent: CaCellTypeSpec
    schema: FieldSchema

    @classmethod
    def from_dclass_field(cls: Type[CactField], field: Field, cact: CaCellTypeSpec, default: Any | None = None) -> CactField:
        fld_type = CactFieldType.from_dclass_field(field)

        if default is not None:
            # Validate default value's type
            if not any(issubclass(default,T) for T in [fld_type.true_type,fld_type.dummy_type]):
                raise TypeError(f"Invalid default value for field `{field.name}`. Default value must match a field's T_true or T_dummy type")
            
        field_schema = fld_type.schema_type(field.name, fld_type.true_type, default)

        return cls(
            name=field.name, 
            field_type=fld_type,
            default_value=default,
            parent=cact,
            schema=field_schema,
        )
    
class CaCellTypeSpec[T]:
    """
    Holds CA Cell (CAC) Type's spec. 
    Also includes dummy magic methods to appease the type checker when writing kernel function code.  
    """

    dclass: T
    fields: dict[str, CactField]
    schema: GroupSchema

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
                fld.name: CactField.from_dclass_field(
                    field=fld,
                    cact=self,
                    default=defaults.get(f'_{fld.name}')
                )
                for fld in dclass_fields
                if not fld.name.startswith('_')
            }

            self.schema = GroupSchema(dclass.__name__, {
                fld.name:fld.schema for fld in self.fields.values()
            })

        except DefinitionException as e:
            display_error(err=DefinitionException(
                f'\nException while creating CACType `{dclass.__name__}`\n{e.message}' # type: ignore
            ))
            exit(1)

    def __repr__(self) -> str:
        return f'<CaCellType: {self.dclass.__qualname__}>'

    def get_layer_ref(self, name: str) -> CaLayerRef:
        return CaLayerRef(name, self)
    
type t_int_like = int | np.int_ | np.int8 | np.int16 | np.uint | np.uint8 | np.uint16
    
class _FieldProto[G,T](Protocol):
    """Minimal API that kernel code expects from every cact_field."""
    def __getitem__(self,
                    idx: t_int_like | tuple[t_int_like, t_int_like] | tuple[t_int_like,t_int_like,t_int_like]) -> T: ...
    def __setitem__(self,
                    idx: t_int_like | tuple[t_int_like, t_int_like] | tuple[t_int_like,t_int_like,t_int_like],
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

type cact_field[T_true: np.generic | VirtualType, T_dummy: Any] = _FieldProto[T_true,T_dummy]