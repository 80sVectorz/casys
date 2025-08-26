from typing import get_origin

from casys.dsl._core import casys_ast
from .debug.ast_origin_tracking import get_origin_map
import ast

class CasysError(Exception): ...

class TranspileError(CasysError):
    """ Raised when the transpiler encounters an invalid kernel or AST state. """

    def __init__(self, msg: str, node: ast.AST | None = None,
                 *, phase: str | None = None) -> None:
        super().__init__(msg)
        self.msg   = msg
        self.node  = node
        self.phase = phase

        if node:
            node_origin = casys_ast.get_meta(node).node_origin
            self.pos = get_origin_map().get(node_origin) if node is not None else None

    def __str__(self) -> str:
        """ Return a CPython-style multi-line error string. """
        header: list[str] = []
        body:   list[str] = []

        # Header
        cls_name = self.__class__.__name__
        if self.phase:
            header.append(f'{cls_name} [{self.phase}]')
        else:
            header.append(cls_name)

        # Source location & snippet
        if self.pos is not None:
            header.append(f'at {self.pos.file}:{self.pos.line}:{self.pos.col}')
            snippet = self.pos.snippet(context=0)
            if snippet:
                # We asked for context=0, so snippet == offending lines only.
                lines = snippet.splitlines()
                start_line = self.pos.line
                for idx, src_line in enumerate(lines, start=start_line):
                    prefix = '>>> ' if idx == start_line else '    '
                    body.append(f'{prefix}{src_line.rstrip()}')

                # Caret underline (first line only for brevity)
                span = max(1, self.pos.end_col - self.pos.col)
                caret = ' ' * self.pos.col + '^' * span
                body.append(f'    {caret}')

        body.append(f'    {self.msg}')

        return '\n'.join(header + body)
    

class KernelError(CasysError): ...
class RuntimeCellError(CasysError): ...