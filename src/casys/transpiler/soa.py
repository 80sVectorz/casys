from __future__ import annotations
from dataclasses import dataclass, field
from pprint import pformat
import numpy as np
from .wrappers import CACType, fields

@dataclass(frozen=True)
class SOASchema:
    input_schema: tuple[tuple[str, CACType], ...]
    output_schema: tuple[tuple[str, np.generic], ...] | None = field(default=None)
    names: set[str] | None = field(default = None)

    def __post_init__(self) -> None:
        names = set([b[0] for b in self.input_schema])
        names |= set(self.cvt(buf, fld) for buf, cac in self.input_schema for fld in cac.fields)
        object.__setattr__(self, "names", names)

        if self.output_schema is None:
            # flatten each field into its own buffer
            os = tuple((
                (f"{buf}_{fld}", f._dtype)
                for buf, cac in self.input_schema
                for fld, f in cac.fields.items()
            ))
            object.__setattr__(self, "output_schema", os)

    def get_subset(self,query: dict[str, list[str]]) -> SOASchema:
        return SOASchema(
            input_schema=self.input_schema, 
            output_schema=tuple((
                (f"{buf}_{fld}", f._dtype)
                for buf, cac in self.input_schema
                if buf in query
                for fld, f in cac.fields.items()
                if fld in query[buf]
            ))
        )
    
    def get_fields_map(self) -> dict[str,tuple[tuple[str,np.generic],...]]:
        return {
            buffer_name:tuple((self.cvt(buffer_name,fld_name),fld._dtype) for fld_name,fld in cact.fields.items())
            for buffer_name,cact in self.input_schema
        }

    def cvt(self, buffer_name: str, field_name: str) -> str:
        return f'{buffer_name}_{field_name}'
    
    def __repr__(self) -> str:
        return pformat({
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
        })