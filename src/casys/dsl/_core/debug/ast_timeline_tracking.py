"""
Light-weight pass based AST change bookkeeping.

Implements the `ChangeTracker` class that can take AST snapshots.
And track per phase and sub-phase changes to the AST.
The snapshots are linked to phases and sub-phases.
It's up to the transpiler modules themselves to notify the tracker when phases start and end.
And to log snapshots at the right time.

All tooling is practically zero-cost when disabled
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import pprint
from typing import TYPE_CHECKING, Any

from casys._utils.debug_utils import ast_recursive_dump
from casys._utils.ast_unparse import ast_to_source

from casys.config import CASYS_CONFIG

from casys.dsl._core.metadata_store import MetadataStore

type t_rec_ast = dict[str, Any] | list[Any] | str | int | float | bool | None

@dataclass(slots=True)
class Snapshot:
    tags: tuple[str,...]
    unparsed_ast: str | None
    str_ast_dump: str | None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ASTTimelineNode:
    phase: str
    children: list[ASTTimelineNode | Snapshot]


class TimelineTracker:
    _timeline: list[ASTTimelineNode]
    _node_path: list[ASTTimelineNode]
    _last_snapshot: str | None = None

    def __init__(self) -> None:
        self._timeline = []
        self._node_path = []

    def enter_phase(self, phase: str) -> None:
        new_node = ASTTimelineNode(
            phase=phase,
            children=[],
        )
        if self._node_path:
            self._node_path[-1].children.append(new_node)
        else:
            self._timeline.append(new_node)

        self._node_path.append(new_node)
        self._last_snapshot = None

    def exit_phase(self) -> None:
        assert self._node_path, "Cannot exit phase without entering one"
        self._last_snapshot = None
        self._node_path.pop()

    def add_snapshot(self, tags: tuple[str,...], ast_node: ast.AST | None = None, metadata: dict[str,Any] | MetadataStore | None = None) -> None:
        src = None
        str_ast = None
        if ast_node is not None:
            src = ast_to_source(ast_node)
            str_ast = ast_recursive_dump(ast_node)
            if metadata is None and src == self._last_snapshot: return
            self._last_snapshot = src
        
        if isinstance(metadata, MetadataStore):
            metadata = metadata.to_dict()

        self._node_path[-1].children.append(Snapshot(
            tags=tags,
            unparsed_ast=src,
            str_ast_dump=str_ast,
            metadata=copy.deepcopy(metadata),
        ))

    def to_json(self):
        """Convert the timeline to a JSON-serializable structure."""
        def serialize_node(node: ASTTimelineNode | Snapshot) -> dict[str, Any]:
            if isinstance(node, ASTTimelineNode):
                return {
                    'phase': node.phase,
                    'children': [serialize_node(child) for child in node.children]
                }
            elif isinstance(node, Snapshot):
                return {
                    'tags': node.tags,
                    'unparsed_ast': node.unparsed_ast,
                    'str_ast_dump': node.str_ast_dump,
                    'metadata': pprint.pformat(node.metadata),
                }
            else:
                raise TypeError(f"Unexpected node type: {type(node)}")

        return [serialize_node(node) for node in self._timeline]


class NullTimelineTracker(TimelineTracker):
    """A no-op tracker that does nothing, used when tracking is disabled."""
    
    def enter_phase(self, phase: str) -> None:
        pass

    def exit_phase(self) -> None:
        pass

    def add_snapshot(self, tags: tuple[str,...], ast_node: ast.AST | None = None, metadata: Any = None) -> None:
        pass

# -------------------------------------- #
# Pre defined tags to ensure consistency #
# -------------------------------------- #

def f_tag_kernel(kernel_name: str) -> str:
    return f'<KERNEL:{kernel_name}>'

def f_tag_transpiler_module(transpiler_module: object) -> str:
    name_attr = getattr(transpiler_module, '__name__', None)
    name = name_attr if isinstance(name_attr, str) else transpiler_module.__class__.__name__
    return f'<TRANSPILER_MODULE:{name}>'

TAG_STEP_FUNC = "<STEP_FUNC>"

# ---------- #
# Singletons #
# ---------- #

_INITIALIZED = False

_TRACKER: TimelineTracker = NullTimelineTracker()

def init_from_config():
    global _INITIALIZED
    if _INITIALIZED: return

    if CASYS_CONFIG.debug_ast_timeline:
        global _TRACKER
        _TRACKER = TimelineTracker()

    _INITIALIZED = True

def get_tracker() -> TimelineTracker:
    """Get the global timeline tracker, initializing it if needed."""
    init_from_config()
    return _TRACKER