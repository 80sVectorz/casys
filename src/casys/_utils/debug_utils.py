import ast
from dataclasses import is_dataclass, fields
import keyword
from re import I
from blib2to3.pgen2.grammar import line
import pygments.styles
from rich.syntax import Syntax
import pygments
from pygments.token import Comment
import logging

from rich.text import Text
from ..logging import log_debug
import pprint

import ast
import pprint
from dataclasses import is_dataclass, fields

def ast_recursive_dump(node, indent: int = 0,
                       list_multiline_threshold: int = 3,
                       line_width_threshold: int = 100) -> str:

    INDENT = 4
    pad = ' ' * indent
    next_indent = indent + INDENT
    next_pad = ' ' * next_indent

    def is_complex(value):
        return isinstance(value, (ast.AST, list)) or is_dataclass(value)

    def format_field(name, value, use_pad: bool = False):
        # Inline trivial values; start a fresh line for anything “struct‑like”.
        if is_complex(value):
            formatted_value = ast_recursive_dump(value, indent if use_pad else next_indent + INDENT,line_width_threshold=line_width_threshold,list_multiline_threshold=list_multiline_threshold)
            lines = formatted_value.splitlines()
            if len(lines) > 1:
                return f"{pad if use_pad else next_pad}{name} = {lines[0].lstrip()}\n{'\n'.join(lines[1:])},"
            else:
                return f"{pad if use_pad else next_pad}{name} = {' '.join(lines).lstrip()},"

        # primitives (str/int/None/...) stay on the same line
        return f"{next_pad}{name} = {pprint.pformat(value)},"

    # -- AST nodes --
    if isinstance(node, ast.AST):
        n_fields = len(node._fields)
        if n_fields:
            field_strings = [format_field(fname, fval, n_fields == 1) for fname, fval in ast.iter_fields(node)]
            fields_str = "\n".join(field_strings)
            if n_fields > 1:
                return f"{pad}{node.__class__.__name__}(\n{fields_str}\n{pad})"
            field_split = field_strings[0].split('=', 1)
            return f"{pad}{node.__class__.__name__}({field_split[0].lstrip()}= {field_split[1].lstrip()}{pad})"
        return f"{pad}{node.__class__.__name__}()"

    # -- Dataclasses --
    if is_dataclass(node):
        if hasattr(node, '__repr__'): return str(node)
        n_fields = len(fields(node))
        if n_fields:
            field_strings = [format_field(f.name, getattr(node, f.name), n_fields == 1) for f in fields(node)]
            fields_str = "\n".join(field_strings)
            if n_fields > 1:
                return f"{pad}{node.__class__.__name__}(\n{fields_str}\n{pad})"
            field_split = field_strings[0].split('=')
            return f"{pad}{node.__class__.__name__}({field_split[0].lstrip()}= {next_pad}{field_split[1].lstrip()}{pad})"
        return f"{pad}{node.__class__.__name__}()"

    # -- Lists --
    if isinstance(node, list):
        if not node:
            return f"{pad}[]"

        multiline = (
            len(node) >= list_multiline_threshold
            or any(is_complex(el) for el in node)
            or sum(len(pprint.pformat(el)) for el in node) > line_width_threshold
        )

        if multiline:
            items = [ast_recursive_dump(el, next_indent) + ',' for el in node]
            return f"{pad}[\n" + "\n".join(items) + f"\n{pad}]"
        inline_items = ', '.join(pprint.pformat(el) for el in node)
        return f"{pad}[{inline_items}]"

    # -- Primitives --
    return pad + pprint.pformat(node)

PARENS_PALETTE = ["#82aaff", "#ffe066", "#86b300", "#c792ea", "#ff5874", "#addb67"]

def rainbow_parens(text: Text) -> Text:
    depth = 0
    for i, char in enumerate(text.plain):
        if char in "([{":
            style = f"{PARENS_PALETTE[depth % len(PARENS_PALETTE)]}"
            text.stylize(style, i, i+1)
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
            style = f"{PARENS_PALETTE[depth % len(PARENS_PALETTE)]}"
            text.stylize(style, i, i+1)
    return text

class PrettyReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        attrs = pprint.pformat(self.__dict__, indent=4, width=80, compact=False)
        return f"{class_name}(\n{attrs}\n)"

def header(*txt):
    log_debug(f"[bold blue]{'\n'.join(txt)}")

def preview(node: ast.AST | None = None, txt: None | str = None):
    if logging.getLogger('casys').level != logging.DEBUG: return

    if node is not None:
        ast.fix_missing_locations(node)
    code = txt if txt else ast.unparse(node)
    lines = code.splitlines()
    longest_line = max(*[len(l) for l in lines])
    lines = [l + (longest_line-len(l))*' ' for l in lines]
    code = '\n'.join(lines)
    syntax = Syntax(code,lexer= 'python', theme='stata-dark', word_wrap=True, line_numbers=True)
    text = rainbow_parens(syntax.highlight(code))
    text = text.with_indent_guides(None,style='grey23')

    log_debug(text.markup)