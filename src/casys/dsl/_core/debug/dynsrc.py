from __future__ import annotations
import linecache
import pathlib
import sys
from casys.config import CASYS_CONFIG

def _debug_dir() -> pathlib.Path:
    """Return the directory for mirrored generated files."""
    root = CASYS_CONFIG.debug_files_dir
    base = root / 'generated'
    return pathlib.Path(base)

def _register_source(virtual_filename: str, source: str) -> None:
    """Make Python's traceback/inspect/pdb able to find our dynamic source.

    Args:
        virtual_filename: A stable path-like name that will show in tracebacks.
        source: The exact source string.
    """
    linecache.cache[virtual_filename] = (
        len(source),
        None,
        [ln for ln in source.splitlines(True)],
        virtual_filename,
    )

def compile_and_exec(
    source: str,
    nspace: dict,
    *,
    virtual_filename: str,
    mirror_kind: str = 'misc',
) -> None:
    """Compile+exec generated source with a stable filename and optional mirroring.

    Args:
        source: Python source to execute.
        nspace: Execution namespace.
        virtual_filename: File name to appear in tracebacks (e.g. 'step.py').
        mirror_kind: Subfolder under the debug dir ('step', 'kernel', etc.).
    """
    # Choose a virtual path that is unique and descriptive.
    # This does not need to exist on disk to help tracebacks/pdb.
    vpath = f'/{mirror_kind}/__casys_gen__/{virtual_filename}'

    mode = getattr(CASYS_CONFIG, 'debug_dynsrc_mode', 'virtual')  # 'off'|'virtual'|'mirror'

    if mode in ('virtual', 'mirror'):
        _register_source(vpath, source)

    if mode == 'mirror':
        out = _debug_dir() / mirror_kind / virtual_filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source, encoding='utf-8')
        # Let debuggers prefer the real file if it exists
        vpath = str(out.resolve())

    code = compile(source, vpath, 'exec', dont_inherit=True)
    nspace.setdefault('__file__', vpath)
    exec(code, nspace, nspace)