from dataclasses import dataclass, field
import ast
from typing import Callable

K_META = 'kernel_meta_data'

@dataclass(kw_only=True)
class KernelASTNodeMeta:
    verified_bounds: bool = field(default=False)

def copy_meta(new_node: ast.AST, node: ast.AST) -> None:
    if hasattr(node, K_META):
        new_node.__setattr__(K_META, node.__getattribute__(K_META))

def set_meta(node: ast.AST, meta: KernelASTNodeMeta) -> None:
    node.__setattr__(K_META, meta)

def get_meta(node: ast.AST) -> KernelASTNodeMeta | None:
    if hasattr(node, K_META):
        return node.__getattribute__(K_META)
    return None

class CASYS_KVal(ast.Name):
    ...

class CASYS_KFuncCall(ast.Call):
    kfunc: Callable
    ...

class CASYS_BufferRead(ast.Expression):
    buffer:
    ...
    