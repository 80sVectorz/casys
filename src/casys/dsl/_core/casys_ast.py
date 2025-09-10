from __future__  import annotations

from typing import TYPE_CHECKING, Callable

from casys.dsl._core.schema.soa_layout import SoaField

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_Base
    from casys.dsl._core.ir import Ir_CaSys
    from casys.dsl._core.schema.schema_base import Schema

from dataclasses import dataclass, field, is_dataclass, fields
import ast

from .descriptors import KernelCallDescriptor

CASYS_META = 'casys_meta_data'

@dataclass(kw_only=True, slots=True)
class AstNodeMeta:
    node_origin: int | None = None

    source_ir: Ir_Base | None = None
    verified_bounds: bool = False
    local_access: bool  = False # Wether a subscript buffer access is using the kernel's position

    evaluates_to_uint_bool: bool = False # Does a expression always evaluate to 1 or 0

def copy_meta(new_node: ast.AST, node: ast.AST) -> AstNodeMeta:
    if hasattr(node, CASYS_META):
        new_node.__setattr__(CASYS_META, node.__getattribute__(CASYS_META))
        return new_node.__getattribute__(CASYS_META)
    return set_meta(new_node, get_meta(node))
        

def set_meta(node: ast.AST, meta: AstNodeMeta) -> AstNodeMeta:
    node.__setattr__(CASYS_META, meta)
    return meta

def get_meta(node: ast.AST) -> AstNodeMeta:
    if hasattr(node, CASYS_META):
        return node.__getattribute__(CASYS_META)
    return set_meta(node, AstNodeMeta(node_origin=id(node)))


class AutoFieldsAst(ast.AST):
    def __init_subclass__(cls):
        super().__init_subclass__()

        base_fields = set()
        for base in cls.__bases__:
            if hasattr(base, '_fields'):
                base_fields.update(base._fields)

        # Dataclass-based fields
        if is_dataclass(cls):
            own_fields = {f.name for f in fields(cls)}
        else:
            own_fields = set(getattr(cls, '__annotations__', {}).keys())

        # Merge and preserve order: base first, then own
        cls._fields = tuple(base_fields | own_fields)  # set union

        # Optional: add _attributes if needed
        if not hasattr(cls, '_attributes'):
            cls._attributes = ()

# ————— Step function nodes —————

@dataclass()
class Cs_KernelCall(AutoFieldsAst):
    desc: KernelCallDescriptor

@dataclass()
class Cs_ParallelGroup(AutoFieldsAst):
    # SoA field level granularity
    swaps: list[str]
    calls: list[KernelCallDescriptor]
    sync_r2w: list[str]
    sync_w2r: list[str]
    
@dataclass()
class Cs_DoubleBufferSwaps(AutoFieldsAst):
    layers: list[str] # Group schema (aka layers) level granularity

# ———— Kernel function nodes —————

class Cs_Macro(ast.Call):
    handler: Callable[[ast.Call, None | Ir_CaSys], list[ast.AST]]
    is_default_handler: bool

    def __init__(
        self, 
        is_default_handler: bool,
        handler: Callable[[ast.Call, None | Ir_CaSys], list[ast.AST]], 
        func: ast.expr, 
        args: list[ast.expr] = [], 
        keywords: list[ast.keyword] = [], 
        **kwargs
    ) -> None:
        super().__init__(func, args, keywords, **kwargs)

        self.handler = handler
        self.is_default_handler = is_default_handler

@dataclass
class Cs_Constant(AutoFieldsAst, ast.expr):
    constant_id: str

@dataclass
class Cs_KVal(AutoFieldsAst, ast.expr):
    kval: str

@dataclass
class Cs_AxisSize(AutoFieldsAst, ast.expr):
    """ Any value representing the grid size across a given axis:  
    - Width -> 0
    - Height -> 1
    - Depth -> 2  
    ...
    """
    ax: int

@dataclass
class Cs_KPos(AutoFieldsAst, ast.expr):
    """ Any value representing the current kernel position on a specific axis:  
    - x pos -> 0
    - y pos -> 1
    - z pos -> 2  
    ...
    """
    ax: int

# @dataclass
# class Cs_LayerFieldRef(AutoFieldsAst, ast.expr):
#     l: str
#     f: str
#     ctx: ast.Load | ast.Store

@dataclass
class Cs_SchemaRef[T_schema: Schema](AutoFieldsAst, ast.expr):
    s: T_schema
    ctx: ast.expr_context

@dataclass
class Cs_SoaFieldRef(AutoFieldsAst, ast.expr):
    field: SoaField
    ctx: ast.expr_context

@dataclass
class Cs_WrIdx(AutoFieldsAst, ast.expr):
    ...

@dataclass
class Cs_RdIdx(AutoFieldsAst, ast.expr):
    ...

@dataclass
class Cs_KFuncCall(AutoFieldsAst):
    kfunc: Callable
    kwargs: dict[str, ast.expr]