from __future__ import annotations
import ast

import numba
    
from casys.dsl._core import casys_ast
from casys.dsl._core.metadata_store import MetadataKey
from casys.dsl._core.soa_field_usage_info_helper import SoaFieldUsageInfo
from casys.dsl._core.schema.schema_base import Schema

MDK_SIGNATURE = MetadataKey[dict[str, numba.types.Type]]('', 'signature', factory=dict, doc='The final JITed kernel function signature')

MDK_TYPE_BINNED_SCHEMA_REFS = MetadataKey[dict[type[Schema], list[casys_ast.Cs_SchemaRef]]]('', 'type_binned_schema_refs', factory=dict, doc='All schema ref nodes sorted per schema type.')

MDK_READONLY = MetadataKey[set[str]]('', 'readonly', doc='A set of variables ids that have been marked readonly.')
MDK_POS_VARS = MetadataKey[dict[str,int]]('', 'pos_vars', factory=dict, doc='A dict that maps pos_var marked variables ids to their axes.')
MDK_ALIASES = MetadataKey[dict[str,ast.AST]]('', 'aliases', factory=dict, doc='A dict that maps variable ids to casys_ast nodes.')
MDK_NEEDS_DEDICATED_IDX = MetadataKey[list[str]]('', 'needs_dedicated_idx', factory=list, doc='A list of the buffers that require a dedicated double buffer index.')

MDK_SOA_FIELD_USAGE_INFO = MetadataKey[SoaFieldUsageInfo]('', 'soa_field_usage_info', doc="The kernel's SoA field usage info.")