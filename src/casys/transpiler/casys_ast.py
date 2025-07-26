from dataclasses import dataclass, field
import ast
from typing import Callable

from casys.wrappers import CACType

CASYS_META = 'casys_meta_data'

@dataclass(kw_only=True)
class KernelASTNodeMeta:
    verified_bounds: bool = field(default=False)

def copy_meta(new_node: ast.AST, node: ast.AST) -> None:
    if hasattr(node, CASYS_META):
        new_node.__setattr__(CASYS_META, node.__getattribute__(CASYS_META))

def set_meta(node: ast.AST, meta: KernelASTNodeMeta) -> None:
    node.__setattr__(CASYS_META, meta)

def get_meta(node: ast.AST) -> KernelASTNodeMeta | None:
    if hasattr(node, CASYS_META):
        return node.__getattribute__(CASYS_META)
    return None

class CASYS_KVal(ast.Name):
    ...

class CASYS_KFuncCall(ast.Call):
    kfunc: Callable
    ...

class CASYS_BufferRead(ast.Expression):
    cact: CACType
    cact: CACType
    ...

class CASYS_BufferWrite(ast.Expression):
    buffer:
    ...

class CASYS_KernelCall(ast.Call):
    ...