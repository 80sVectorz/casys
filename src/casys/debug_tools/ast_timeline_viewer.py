"""
Interactive CLI AST Timeline Viewer.

Features:
- Loads JSON exported from TimelineTracker.to_json()
- Interactive navigation like a debugger:
  n / next  -> step forward
  p / prev  -> step backward
  in        -> step into phase
  out       -> step out of phase
  filter    -> set a Python lambda for tag filtering
  q         -> quit
- Diff view of AST source between snapshots
- Syntax highlighting using rich
"""

from __future__ import annotations

import ast
import json
import difflib
from pathlib import Path
from re import I
import shutil
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.text import Text

console = Console()

try:
    import black
except ImportError:
    black = None


class TimelineViewer:
    def __init__(self, json_path: str):
        self.raw_timeline: list[dict[str, Any]] = json.loads(Path(json_path).read_text())
        self.filter_fn: Callable[[tuple[str, ...]], bool] | None = None
        self.path: list[int] = []  # stack of child indices representing current phase path
        self.index: int = 0
        self.prev_src: str = ''
        self._prev_by_track: dict[str, str] = {}  # per-track previous source for diffs
        self._prev_meta_by_track: dict[str, str] = {}  # per-track previous metadata json for diffs
        self.filtered_timeline = self._apply_filter(self.raw_timeline)

    # -----------------
    # Filtering
    # -----------------
    def _apply_filter(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Recursively apply tag filter, keeping phase hierarchy intact."""
        if self.filter_fn is None:
            return nodes
        result: list[dict[str, Any]] = []
        for node in nodes:
            if 'phase' in node:
                filtered_children = self._apply_filter(node['children'])
                if filtered_children:
                    result.append({'phase': node['phase'], 'children': filtered_children})
            elif 'tags' in node:
                tags_tuple = tuple(node['tags'])
                if self.filter_fn(tags_tuple):
                    result.append(node)
        return result

    def set_filter(self, expr: str) -> None:
        """Set the tag filter from a lambda expression string."""
        try:
            fn = eval(expr)
            if not callable(fn):
                raise ValueError('Filter must be callable')
            self.filter_fn = fn
            self.filtered_timeline = self._apply_filter(self.raw_timeline)
            self.path.clear()
            self.index = 0
            self.prev_src = ''
            self._prev_by_track.clear()
            self._prev_meta_by_track.clear()
            console.print(f'[green]Filter set:[/green] {expr}')
        except Exception as e:
            console.print(f'[red]Invalid filter:[/red] {e}')

    # -----------------
    # Navigation helpers
    # -----------------
    def _get_current_nodes(self) -> list[dict[str, Any]]:
        nodes = self.filtered_timeline
        for idx in self.path:
            nodes = nodes[idx]['children']
        return nodes

    def _current_node(self) -> dict[str, Any]:
        nodes = self._get_current_nodes()
        return nodes[self.index]

    # -----------------
    # Display
    # -----------------
    def _display_diff(self, prev_src: str, curr_src: str) -> None:
        diff_lines = list(
            difflib.unified_diff(
                prev_src.splitlines(),
                curr_src.splitlines(),
                lineterm='',
                fromfile='prev',
                tofile='curr'
            )
        )
        if not diff_lines:
            console.print('[green]No changes[/green]')
            return
        syntax = Syntax('\n'.join(diff_lines), 'diff', theme='monokai', line_numbers=False, word_wrap=True)
        console.print(syntax)

    def _shorten_phase_path(self, max_len: int = 50) -> str:
        """Return the current phase path as a string, shortened if too long."""
        nodes = self.filtered_timeline
        parts = []
        for idx in self.path:
            phase = nodes[idx]['phase']
            parts.append(phase)
            nodes = nodes[idx]['children']
        if 'phase' in self._current_node():
            parts.append(self._current_node()['phase'])
        path_str = ' / '.join(parts)
        if len(path_str) > max_len:
            return '...' + path_str[-(max_len - 3):]
        parts = path_str.split(' / ')
        if len(parts) > 1:
            parts[-1] = f'[blue]{parts[-1]}[/blue]'
        path_str = ' / '.join(parts)
        return path_str

    def _format_python(self, code: str) -> str:
        """Format Python code with black if available."""
        if black:
            try:
                return black.format_str(code, mode=black.Mode(line_length=200, magic_trailing_comma=False))
            except Exception:
                return code
        return code

    def _prepare_display_code(self, src: str) -> str:
        """
        Convert a raw snapshot string into the exact string we render.

        :param src: Raw snapshot string as stored on the timeline.
        :type src: str
        :return: Fully prepared source to render.
        :rtype: str
        """
        code = src.replace('"""<<', '').replace('>>"""', '')
        code = self._format_python(code)
        return code

    def _track_key_from_tags(self, tags: tuple[str, ...]) -> str:
        """
        Build a stable track key so we only diff snapshots within the same logical stream.

        Priority:
        - Use the <KERNEL:...> tag if present.
        - Else use <STEP_FUNC> if present.
        - Else use all tags except <TRANSPILER_MODULE:...>, sorted and joined.

        :param tags: Snapshot tags tuple.
        :type tags: tuple[str, ...]
        :return: Track identifier string.
        :rtype: str
        """
        kernel = next((t for t in tags if t.startswith('<KERNEL:')), None)
        if kernel is not None:
            return kernel
        if '<STEP_FUNC>' in tags:
            return '<STEP_FUNC>'
        kept = tuple(sorted(t for t in tags if not t.startswith('<TRANSPILER_MODULE:')))
        return ' | '.join(kept) if kept else 'UNSPECIFIED'

    def _display_json_with_changes(self, prev_json: str, curr_json: str) -> None:
        """Render JSON with syntax highlighting and highlight changed lines.

        The diff is computed on the pretty-printed JSON strings.
        """
        sm = difflib.SequenceMatcher(a=prev_json.splitlines(), b=curr_json.splitlines())
        changed_lines: set[int] = set()
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'insert'):
                changed_lines.update(range(j1 + 1, j2 + 1))

        syntax = Syntax(
            curr_json,
            'python',
            theme='monokai',
            line_numbers=True,
            word_wrap=True,
            indent_guides=True,
            highlight_lines=changed_lines,
        )
        console.print(syntax)

    def _display_python_with_changes(self, prev_src: str, curr_src: str) -> None:
        """
        Render Python with syntax highlighting and inline change emphasis.

        The diff is computed on the *formatted* display strings so highlighting
        stays in sync with what is shown.

        :param prev_src: Previous snapshot source (raw).
        :type prev_src: str
        :param curr_src: Current snapshot source (raw).
        :type curr_src: str
        :return: None
        :rtype: None
        """
        prev_code = self._prepare_display_code(prev_src)
        curr_code = self._prepare_display_code(curr_src)

        sm = difflib.SequenceMatcher(a=prev_code.splitlines(), b=curr_code.splitlines())
        changed_lines: set[int] = set()
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'insert'):
                changed_lines.update(range(j1 + 1, j2 + 1))  # 1-based for Syntax

        syntax = Syntax(
            curr_code,
            'python',
            theme='monokai',
            line_numbers=True,
            word_wrap=True,
            indent_guides=True,
            highlight_lines=changed_lines,
        )
        console.print(syntax)

    def _display_ast_dump(self, ast_dump: str) -> None:
        """Render a Python-highlighted AST dump."""
        # Optional: pretty format with black if available
        code = self._format_python(ast_dump)

        syntax = Syntax(
            code,
            'python',
            theme='monokai',
            line_numbers=True,
            word_wrap=False,
            indent_guides=True,
        )
        console.print(syntax)

    def _display_node(self, node: dict[str, object]) -> None:
        """
        Render the current node with phase path and code (if snapshot).

        :param node: Current node from the filtered tree.
        :type node: dict[str, object]
        :return: None
        :rtype: None
        """
        width = shutil.get_terminal_size().columns
        path = self._shorten_phase_path(max_len=max(30, width - 10))
        console.print(Panel.fit(f'[bold blue]Phase path:[/bold blue] {path}', border_style='blue'))

        if 'phase' in node:
            console.print(f"[blue]Phase:[/blue] {node['phase']}")
        else:
            tags_tuple = tuple(node.get('tags', ()))  # type: ignore
            tags_str = ', '.join(tags_tuple)
            console.print(f'[magenta]Tags:[/magenta] {tags_str}')
            curr_src = str(node['unparsed_ast'])
            curr_src = curr_src.replace('\'<<','"<<').replace('>>\'','>>"')
            curr_src = curr_src.replace('"<<','"""<<').replace('>>"','>>"""').replace('\\n','\n')

            # Track-aware diff: compare only within same logical stream
            track_key = self._track_key_from_tags(tags_tuple)
            prev_for_track = self._prev_by_track.get(track_key, '')
            self._display_python_with_changes(prev_for_track, curr_src)
            self._prev_by_track[track_key] = curr_src

    # -----------------
    # Main loop
    # -----------------

    def handle_navigate(self, cmd: str, nodes: list[dict[str,Any]]):
        node = nodes[self.index]
        if cmd in ('n', 'next'):
            if self.index < len(nodes) - 1:
                self.index += 1
            else:
                console.print('[yellow]End of phase[/yellow]')
                self.handle_navigate('o',nodes)
                self.handle_navigate('n',self._get_current_nodes())
        elif cmd in ('p', 'prev'):
            if self.index > 0:
                self.index -= 1
                self.prev_src = ''
            else:
                console.print('[yellow]Start of phase[/yellow]')
        elif cmd in ('i', 'in'):
            if 'phase' in node and node['children']:
                self.path.append(self.index)
                self.index = 0
                self.prev_src = ''
            else:
                console.print('[yellow]Not a phase or has no children[/yellow]')
        elif cmd in ('o', 'out'):
            if self.path:
                self.path.pop()
                self.index = 0
                self.prev_src = ''
            else:
                console.print('[yellow]Already at top level[/yellow]')

    def run(self) -> None:
        while True:
            nodes = self._get_current_nodes()
            if not nodes:
                console.print('[red]No nodes to display[/red]')
                return

            node = nodes[self.index]
            self._display_node(node)

            cmd = Prompt.ask(
                '[bold cyan]Command[/bold cyan]',
                choices=[
                    'n', 'next',' ',
                    'p', 'prev',' ',
                    'i', 'in',' ',
                    'o', 'out',' ',
                    'a', 'ast', ' ',
                    'm', 'meta',' ',
                    'f', 'filter',' ',
                    'q', 'quit',' ',
                ],
                default='n'
            )
            if cmd in ('q', 'quit'):
                break

            match cmd:
                case (
                    'n' | 'next' |
                    'p' | 'prev' |
                    'i' | 'in'   |
                    'o' | 'out'
                ):
                    self.handle_navigate(cmd,nodes)
                case 'f' | 'filter':
                    expr = Prompt.ask("Enter filter lambda (e.g., 'lambda tags: \"<STEP_FUNC>\" in tags')")
                    self.set_filter(expr)
                case 'a' | 'ast':
                    self._display_ast_dump(node['str_ast_dump'])
                case 'm' | 'meta':
                    meta = node.get('metadata') if isinstance(node, dict) else None
                    if meta is None:
                        console.print('[yellow]No metadata for this snapshot[/yellow]')
                    else:
                        meta_str = meta
                        tags_tuple = tuple(node.get('tags', ()))  # type: ignore
                        track_key = self._track_key_from_tags(tags_tuple)
                        prev_meta = self._prev_meta_by_track.get(track_key, '')
                        self._display_json_with_changes(prev_meta, meta_str)
                        self._prev_meta_by_track[track_key] = meta_str


if __name__ == '__main__':
    import argparse

    def main() -> None:
        """
        CLI entry point for the interactive timeline viewer.  
        Parses command line arguments and starts the TUI session.
        """
        parser = argparse.ArgumentParser(
            prog='timeline-viewer',
            description='Interactive AST timeline viewer with rich diffs and tag-filter lambdas.'
        )
        parser.add_argument('json_path', help='Path to TimelineTracker JSON file.')
        parser.add_argument(
            '--filter',
            dest='filter_expr',
            default=None,
            help='Python lambda to prefilter snapshots, e.g. "lambda tags: \'<STEP_FUNC>\' in tags"'
        )
        args = parser.parse_args()

        viewer = TimelineViewer(args.json_path)
        if args.filter_expr:
            viewer.set_filter(args.filter_expr)

        try:
            viewer.run()
        except KeyboardInterrupt:
            console.print('[yellow]Interrupted[/yellow]')
    
    main()