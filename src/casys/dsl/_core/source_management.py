from __future__ import annotations
import hashlib
import inspect
import importlib.util
import pathlib
import sys
import ast
from types import ModuleType
from typing import Any, Literal

from casys.config import CASYS_CONFIG

try:
    import numba  # ok to import here
    from numba.core.registry import CPUDispatcher  # type: ignore[attr-defined]
except Exception:
    CPUDispatcher = ()  # type: ignore[assignment]

def _gen_root() -> pathlib.Path:
    return CASYS_CONFIG.cache_files_dir / 'gen_modules'

def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

def _sha1_str(s: str) -> str:
    return _sha1_bytes(s.encode('utf-8'))


def _collect_referenced_globals(source: str) -> set[str]:
    """Return names that are read from globals within the given source.

    This excludes names bound in the module itself.
    """
    tree = ast.parse(source)
    assigned: set[str] = set()
    globals_read: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            assigned.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            assigned.add(node.name)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                globals_read.add(node.id)

    V().visit(tree)
    globals_read.difference_update(assigned)
    # Always ignore builtins marker
    globals_read.discard('__builtins__')
    return globals_read


def _is_hashable_literal(obj: Any) -> bool:
    """Cheap filter for small literals worth hashing into the salt."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return True
    if isinstance(obj, tuple) and all(isinstance(x, (int, float, str, bool, type(None))) for x in obj):
        return True
    return False


def _unwrap_callable(obj: Any) -> Any:
    if CPUDispatcher and isinstance(obj, CPUDispatcher):
        return obj.py_func
    return obj

def _callable_source(obj: Any) -> str | None:
    try:
        src = inspect.getsource(obj)
        return src
    except Exception:
        try:
            co = getattr(obj, '__code__', None)
            if co is not None:
                return f'<code:{_sha1_bytes(co.co_code)}>'
        except Exception:
            return None
    return None


def _safe_obj_fingerprint(name: str, obj: Any) -> str:
    obj = _unwrap_callable(obj)
    if callable(obj):
        src = _callable_source(obj)
        if src is not None:
            return f'f:{name}:{_sha1_str(src)}'
        return f'fnorepr:{name}:{type(obj).__name__}'
    if _is_hashable_literal(obj):
        return f'lit:{name}:{repr(obj)}'
    return f'type:{name}:{type(obj).__name__}'


def _deps_hash_for_source(
    source: str,
    nspace: dict[str, Any] | None,
    *,
    dep_mode: str,
    depends: list[str] | None,
    extra_salt: str,
) -> str:
    names: list[str] = []
    if nspace is None:
        return _sha1_str(extra_salt)

    if dep_mode == 'explicit':
        names = depends or []
    elif dep_mode == 'all':
        names = list(nspace.keys())
    else:
        ref = _collect_referenced_globals(source)
        names = [n for n in ref if n in nspace]

    parts: list[str] = [extra_salt]
    for k in sorted(names):
        parts.append(_safe_obj_fingerprint(k, nspace[k]))
    return _sha1_str('|'.join(parts))




def get_assigned_names(source: str) -> set[str]:
    """Return names defined in the module source so we do not overwrite them."""
    tree = ast.parse(source)
    out: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            out.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            out.add(node.name)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
            self.generic_visit(node)

    V().visit(tree)
    return out


def import_from_source(
    source: str,
    *,
    virtual_filename: str,
    mirror_kind: str,
    cache_salt: str = '',
    nspace: dict[str, Any] | None = None,
    dep_mode: Literal['scan', 'all', 'explicit'] = 'scan',
    depends: list[str] | None = None,
    inject_into_module: bool = False,
) -> ModuleType:
    """Write source to a content-addressed module on disk and import it."""
    deps_h = _deps_hash_for_source(source, nspace, dep_mode=dep_mode, depends=depends, extra_salt=cache_salt)
    src_h = _sha1_str(source + '|' + deps_h)

    root = _gen_root() / mirror_kind
    pkg_dir = root / (virtual_filename.split('.')[0]+src_h)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    init_py = pkg_dir / '__init__.py'
    if not init_py.exists():
        init_py.write_text('', encoding='utf-8')


    mod_path = pkg_dir / virtual_filename

    write = True
    if mod_path.exists():
        try:
            existing = mod_path.read_text(encoding='utf-8')
            if existing == source:
                write = False
        except Exception:
            write = True

    if write:
        mod_path.write_text(source, encoding='utf-8')

    fqname = f'casys_gen.{mirror_kind}.{src_h}.{virtual_filename[:-3]}'
    spec = importlib.util.spec_from_file_location(fqname, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fqname] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)

    if inject_into_module and nspace:
        defined = get_assigned_names(source)
        for k, v in nspace.items():
            if k not in defined:
                module.__dict__.setdefault(k, v)

    return module