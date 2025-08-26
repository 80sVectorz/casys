"""
Light-weight source-location bookkeeping.
Implements the `OriginMap` class that maps AST nodes to their
original source location, allowing for better error messages and debugging.
The `get_origin_map` function provides access to a global origin map,

All tooling is practically zero-cost when disabled
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, cast

from casys.config import CASYS_CONFIG
from casys.dsl._core import casys_ast

@dataclass(slots=True)
class SourcePos:
    """ Raw position info for one concrete span in the source file.

    :param file: Absolute or project-relative input path.
    :type file: str
    :param line: 1-based start line.
    :type line: int
    :param col: 0-based start column.
    :type col: int
    :param end_line: 1-based end line (inclusive).
    :type end_line: int
    :param end_col: 0-based end column (exclusive).
    :type end_col: int
    """
    file: str
    line: int
    col: int
    end_line: int
    end_col: int

    # Lazily loaded code lines for pretty printing / Rich Syntax
    _snippet: str | None = field(default=None, repr=False, init=False)

    def snippet(self, context: int = 2) -> str:
        """Return a small code excerpt around the span."""

        if self._snippet is not None:
            return self._snippet

        path = Path(self.file)
        if not path.exists():
            return ''

        src = path.read_text(encoding='utf-8').splitlines()
        lo = max(self.line - context - 1, 0)
        hi = min(self.end_line + context, len(src))
        excerpt = '\n'.join(src[lo:hi])
        self._snippet = excerpt
        return excerpt
    
_SOURCE_CACHE: dict[str, list[str]] = {}

def _get_source_lines(file: str) -> list[str] | None:
    """Return cached lines or load from disk once."""
    if file in _SOURCE_CACHE:
        return _SOURCE_CACHE[file]
    try:
        text = Path(file).read_text(encoding='utf-8')
    except (FileNotFoundError, OSError):
        return None
    _SOURCE_CACHE[file] = text.splitlines()
    return _SOURCE_CACHE[file]

class OriginMap:
    """Mutable mapping *node id* -> :class:`SourcePos`."""

    __slots__ = ('_map',)

    def __init__(self) -> None:
        self._map: dict[int, SourcePos] = {}

    # -- Basic API -- #

    def add(self, node: ast.AST, pos: SourcePos) -> None:
        self._map[id(node)] = pos
        casys_ast.get_meta(node).node_origin = id(node)
        setattr(node, 'origin', pos)  # single attribute on the node

    def update(self, items: Iterable[tuple[int,SourcePos]]) -> None: 
        self._map.update(items)

    def get(self, node: ast.AST | int | None) -> SourcePos | None:
        if node is None:
            return None
        node_id = node if isinstance(node,int) else id(node)
        return self._map.get(node_id)

    # -- Iteration helpers -- #

    def __contains__(self, node: ast.AST) -> bool:
        return id(node) in self._map

    def items(self):
        return self._map.items()

    def __len__(self) -> int: 
        return len(self._map)


class NullOriginMap(OriginMap):
    """Stub that silently swallows every call"""

    def __init__(self) -> None:
        pass

    def add(self, node: ast.AST, pos: SourcePos) -> None: 
        return

    def update(self, items: Iterable[tuple[int,SourcePos]]) -> None: 
        return

    def get(self, node: ast.AST | int | None) -> None:          
        return None

    def items(self): # type: ignore
        return iter(())

    def __len__(self) -> int:                             
        return 0

# --------------- #
# Public helpers  #
# --------------- #

def build_origin_map(
        tree: ast.AST, *,
        filename: str,
        source: str | None = None,
        line_offset: int = 0,
) -> OriginMap:
    """ Attach SourcePos to every node.

    If *source* is given, it is cached and used for future
    ``SourcePos.snippet`` look-ups instead of reading the file again.
    """



    if source is not None:
        _SOURCE_CACHE[filename] = source.splitlines()

    origin_map = OriginMap()
    for node in ast.walk(tree):
        if not hasattr(node, 'lineno') or not hasattr(node, 'col_offset'): # synthetic node
            continue
        node = cast(ast.expr, node)

        abs_line = line_offset + node.lineno
        abs_end_line = line_offset + getattr(node, 'end_lineno', node.lineno)

        origin_map.add(
            node,
            pos = SourcePos(
                file=filename,
                line=abs_line,
                col=node.col_offset,
                end_line=abs_end_line,
                end_col=getattr(node, 'end_col_offset', node.col_offset + 1),
            ),
        )
    return origin_map

# ---------- #
# Singletons #
# ---------- #

_INITIALIZED = False

_ORIGIN_MAP: OriginMap = NullOriginMap()

def init_from_config():
    global _INITIALIZED
    if _INITIALIZED: return

    if CASYS_CONFIG.debug_ast_origin_tracking:
        global _ORIGIN_MAP
        _ORIGIN_MAP = OriginMap()

    _INITIALIZED = True
    

def get_origin_map() -> OriginMap:
    """Get the global origin map, initializing it if needed."""
    init_from_config()
    return _ORIGIN_MAP