from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from numpy.tests.test__all__ import test_no_duplicates_in_np__all__

class Message:
    ...

type t_expected_bindings = list[str | t_expected_bindings]

@dataclass
class AnnounceBinding(Message):
    expected_bindings: str | t_expected_bindings
    node: Pattern

# ---------------------------------------------------------------------------
# Core pattern base class
# ---------------------------------------------------------------------------
class Pattern(ast.AST):
    """Base class for AST matching patterns."""

    _buffered_messages: list[Message] | None = None
    _forces_list: bool = False  # overridden by repetition wrappers
    _parent: Pattern | None = None

    # ---------------------------------- public API ----------------------------------
    def match(self, node: object, bindings: dict[str, object] | None = None):
        """Match *node* and return updated *bindings* or *None*."""
        bindings = bindings or {}
        raise NotImplementedError

    def handle_message(self, message: Message) -> None:
        self.bubble_message(message)

    def bubble_message(self, message: Message) -> None:
        if self._parent is not None:
            self._parent.handle_message(message)
        else:
            if self._buffered_messages is None:
                self._buffered_messages = []
            self._buffered_messages.append(message)

    # ------------------------------- parent navigation -------------------------------
    def _on_parent_init(self) -> None:
        if self._buffered_messages:
            for m in self._buffered_messages:
                self.bubble_message(m)

    def _set_parent(self, parent: Pattern) -> None:
        self._parent = parent
        self._on_parent_init()

    def _ancestor_forces_list(self) -> bool:
        cur = self._parent
        while cur is not None:
            if getattr(cur, 'expected_bindings', False):
                return getattr(cur, '_forces_list', False)
            cur = cur._parent  # type: ignore[attr-defined]
        return False

    # ----------------------------------- helpers ------------------------------------
    @staticmethod
    def _to_list(val: Any) -> list[Any]:
        return val if isinstance(val, list) else [val]


# ---------------------------------------------------------------------------
# Simple binder
# ---------------------------------------------------------------------------
class Bind(Pattern):
    """Bind the current node to *name*."""

    key: str

    def __init__(self, name: str):
        self.key = name

    def _on_parent_init(self) -> None:
        super()._on_parent_init()
        self.bubble_message(AnnounceBinding(self.key, self))

    def match(self, node: Any, bindings: dict[str, Any] | None = None):
        bindings = bindings or {}
        force = self._ancestor_forces_list()
        if self.key in bindings:
            if not force:
                return None
            bindings[self.key] = self._to_list(bindings[self.key]) + [node]
        else:
            bindings[self.key] = [node] if force else node
        return bindings

class WildCard(Pattern):
    """Match any node"""

    def match(self, node: Any, bindings: dict[str, Any] | None = None):
        bindings = bindings or {}
        return bindings

# ---------------------------------------------------------------------------
# Node pattern with field constraints
# ---------------------------------------------------------------------------
class NodePattern(Pattern):
    """Match an AST node of *node_type* with constraints on its fields."""

    def __init__(self, node_type: type[ast.AST], **field_patterns: "Pattern | Any"):
        self.node_type = node_type
        self.field_patterns = field_patterns
        for pat in field_patterns.values():
            if isinstance(pat, Pattern):
                pat._set_parent(self)

    def match(self, node: Any, bindings: dict[str, Any] | None = None):
        bindings = bindings or {}
        if not isinstance(node, self.node_type):
            return None
        merged = dict(bindings)
        for field, pat in self.field_patterns.items():
            val = getattr(node, field, None)
            if isinstance(pat, Pattern):
                if val is None:
                    return None
                # Match list-valued field
                if isinstance(val, list) and not isinstance(pat, Bind):
                    res = _match_patterns([pat], val, 0, {})
                    if not res or res[0][1] != len(val):
                        return None
                    sub_bind = res[0][0]
                else:
                    sub_bind = pat.match(val, {})
                    if sub_bind is None:
                        return None
                # merge sub bindings
                for k, v in sub_bind.items():
                    if k in merged:
                        if not self._ancestor_forces_list():
                            return None
                        merged[k] = self._to_list(merged[k]) + self._to_list(v)
                    else:
                        merged[k] = self._to_list(v) if self._ancestor_forces_list() else v
            else:
                if val != pat:
                    return None
        return merged

class Collect(Pattern):
    """Collect *node* under *key* and merge sub-bindings into current scope."""

    expected_bindings: t_expected_bindings

    def __init__(self, pattern: Pattern, key: str):
        self.pattern = pattern
        self.key = key
        self.expected_bindings = []
        pattern._set_parent(self)
        
    def _on_parent_init(self) -> None:
        super()._on_parent_init()
        if self._ancestor_forces_list():
            if self.expected_bindings:
                self.bubble_message(AnnounceBinding([self.key, self.expected_bindings], self))
            else:
                self.bubble_message(AnnounceBinding(self.key, self))
        else:
            for binding in [*self.expected_bindings,self.key]:
                self.bubble_message(message=AnnounceBinding(binding, self))

    def match(self, node: Any, bindings: dict[str, Any] | None = None) -> None | dict[str, Any]:
        bindings = bindings or {}
        inner = self.pattern.match(node, {})
        if inner is None:
            return None
        force = self._ancestor_forces_list()
        merged = dict(bindings)

        if force:
            if self.expected_bindings:
                # Inside a repetition wrapper – *do not* merge inner bindings;
                # instead, append the inner-dict itself to the list under *key*.
                if self.key in merged:
                    merged[self.key].append(inner)
                else:
                    merged[self.key] = [inner]
                return merged
            else:
                if self.key in merged:
                    merged[self.key].append(node)
                else:
                    merged[self.key] = [node]
                return merged

        # Outside repetition – store node and merge inner bindings
        if self.key in merged:
            return None  # scalar expected, duplicate found
        merged[self.key] = node
        for k, v in inner.items():
            if k in merged:
                return None  # scalar expected, duplicate found
            merged[k] = v
        return merged

    def handle_message(self, message: Message) -> None:
        match message:
            case AnnounceBinding():
                self.expected_bindings.append(message.expected_bindings)
                # print(self.key, self.expected_bindings)

class Filter(Pattern):
    """Accept nodes where *predicate(node)* is *True* and optionally bind."""

    def __init__(self, predicate: Callable[[Any], bool], key: str | None = None):
        self.predicate = predicate
        self.key = key

    def _on_parent_init(self) -> None:
        if self.key:
            self.bubble_message(AnnounceBinding(self.key, self))

    def match(self, node: Any, bindings: dict[str, Any] | None = None):
        bindings = bindings or {}
        if not self.predicate(node):
            return None
        if self.key is None:
            return bindings
        force = self._ancestor_forces_list()
        if self.key in bindings:
            if not force:
                return None
            bindings[self.key] = self._to_list(bindings[self.key]) + [node]
        else:
            bindings[self.key] = [node] if force else node
        return bindings


# ---------------------------------------------------------------------------
# Repetition wrappers
# ---------------------------------------------------------------------------

class Repetition(Pattern):
    """Base class for repetition patterns."""

    _forces_list = True
    expected_bindings: list[str | t_expected_bindings]

    def __init__(self, pattern: Pattern):
        self.expected_bindings = []
        self.pattern = pattern
        pattern._set_parent(self)

    def handle_message(self, message: Message) -> None:
        match message:
            case AnnounceBinding():
                self.expected_bindings.append(message.expected_bindings)

        super().handle_message(message)


class OneOrMore(Repetition):
    """One-or-more repetition."""

class ZeroOrMore(Repetition):
    """Zero-or-more repetition."""


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

def _match_patterns(patterns: Sequence[Pattern], nodes: list[Any], pos: int, bindings: dict[str, Any]):
    if not patterns:
        return [(bindings, pos)]
    first, *rest = patterns
    out: list[tuple[dict[str, Any], int]] = []

    match first:
        case OneOrMore():
            max_rep = 0
            while pos + max_rep < len(nodes) and first.pattern.match(nodes[pos + max_rep], {}):
                max_rep += 1
            for rep in range(max_rep, 0, -1):
                cur_bind, cur_pos = dict(bindings), pos
                for _ in range(rep):
                    res = first.pattern.match(nodes[cur_pos], cur_bind)
                    if res is None:
                        break
                    cur_bind, cur_pos = res, cur_pos + 1
                else:
                    out.extend(_match_patterns(rest, nodes, cur_pos, cur_bind))
            return out

        case ZeroOrMore():
            # zero-repeat branch (only if remainder advances or at end)
            for b, p in _match_patterns(rest, nodes, pos, dict(bindings)):
                if p != pos or p == len(nodes):
                    out.append((b, p))
            # one-or-more branch
            max_rep = 0
            while pos + max_rep < len(nodes) and first.pattern.match(nodes[pos + max_rep], {}):
                max_rep += 1
            for rep in range(max_rep, 0, -1):
                cur_bind, cur_pos = dict(bindings), pos
                for _ in range(rep):
                    res = first.pattern.match(nodes[cur_pos], cur_bind)
                    if res is None:
                        break
                    cur_bind, cur_pos = res, cur_pos + 1
                else:
                    out.extend(_match_patterns(rest, nodes, cur_pos, cur_bind))
            return out

    # ---------- single ----------
    if pos < len(nodes):
        res = first.match(nodes[pos], dict(bindings))
        if res is not None:
            out.extend(_match_patterns(rest, nodes, pos + 1, res))
    return out


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def match_sequence(patterns: list[Pattern], nodes: list[Any]):
    """Return list of binding dicts for non-overlapping matches in *nodes*."""
    results: list[dict[str, Any]] = []
    i = 0
    while i < len(nodes):
        m = _match_patterns(patterns, nodes, i, {})
        if not m:
            i += 1
            continue
        b, new_pos = m[0]
        results.append(b)
        i = new_pos
    return results


# -----------------------------------------------------------------------------
# transformer & finder utilities
# -----------------------------------------------------------------------------

type ReplaceResult = ast.AST | list[ast.AST] | None

class PatternTransformer(ast.NodeTransformer):
    """
    Walk the tree, find non-overlapping matches of *pattern* and run *actions*.
    In addition, every successful set of bindings is appended to
    `self.matches` (like PatternFinder).

    actions = {collect_key: handler | None}

      • handler(bindings) → list[ast.AST]
          - `bindings` is the FULL match dict
          - the returned list replaces the *anchor* node (see below)

      • None
          - delete every node collected under *collect_key*
    """

    def __init__(
        self,
        pattern: Sequence[Pattern],
        actions: dict[str, Callable[[dict[str, Any]], list[ast.AST]] | None],
    ):
        super().__init__()
        self.pattern = pattern
        self.actions = actions
        self.matches: list[dict[str, Any]] = []

    def _record(self, bindings: dict[str, Any]) -> None:
        """Internal helper so we have just one place that appends."""
        self.matches.append(bindings)

    def _normalize_replace_for_nonlist(self, rep: ReplaceResult, field: str) -> ast.AST:
        """
        Normalize a replacement for a non-list field.

        - ast.AST -> ast.AST
        - [ast.AST] with len==1 -> element
        - None or list with len!=1 -> error
        """
        if isinstance(rep, ast.AST):
            return rep
        if isinstance(rep, list):
            if len(rep) == 1 and isinstance(rep[0], ast.AST):
                return rep[0]
            raise ValueError(f'Cannot replace non-list field {field} with {len(rep)} nodes')
        raise ValueError(f'Cannot delete non-list field {field}')


    # ------------- helpers to interpret collected values ------------------
    # @staticmethod
    # def _as_nodes(value: Any) -> list[ast.AST]:
    #     if isinstance(value, ast.AST):
    #         return [value]
    #     if isinstance(value, list):
    #         return [n for n in value if isinstance(n, ast.AST)]
    #     if isinstance(value, dict):
    #         n = value.get("_node")
    #         return [n] if isinstance(n, ast.AST) else []
    #     return []
    
    @staticmethod
    def _as_nodes(value: Any) -> list[ast.AST]:
        """
        Extract AST nodes from arbitrarily nested values.

        The value may be:
        - ast.AST: returned as a single-element list
        - list: flattened recursively
        - dict: values scanned recursively (no special key required)
        - other: ignored

        Returns:
            list[ast.AST]: All AST nodes found, in encounter order.
        """
        out: list[ast.AST] = []
        def rec(v: Any) -> None:
            if isinstance(v, ast.AST):
                out.append(v)
            elif isinstance(v, list):
                for it in v:
                    rec(it)
            elif isinstance(v, dict):
                for it in v.values():
                    rec(it)
        rec(value)
        return out
    
    def _replace_within(self, parent: ast.AST, target: ast.AST, repl: list[ast.AST]) -> bool:
        """
        Replace `target` somewhere inside `parent` with `repl`.
        Returns True if a replacement was performed.
        """
        def walk(node: ast.AST) -> bool:
            for field, val in ast.iter_fields(node):
                if val is target:
                    if len(repl) != 1:
                        raise ValueError(f'Cannot replace non-list field {field} with {len(repl)} nodes')
                    setattr(node, field, repl[0])
                    return True
                if isinstance(val, list):
                    for i, elem in enumerate(val):
                        if elem is target:
                            val[i:i+1] = repl
                            return True
                if isinstance(val, ast.AST) and walk(val):
                    return True
                if isinstance(val, list):
                    for elem in val:
                        if isinstance(elem, ast.AST) and walk(elem):
                            return True
            return False
        return walk(parent)

    def _delete_within(self, parent: ast.AST, target: ast.AST) -> bool:
        """
        Delete `target` if it appears in a list field somewhere inside `parent`.
        Returns True if a deletion was performed.
        """
        def walk(node: ast.AST) -> bool:
            for _, val in ast.iter_fields(node):
                if isinstance(val, list):
                    i = 0
                    while i < len(val):
                        elem = val[i]
                        if elem is target:
                            del val[i]
                            return True
                        if isinstance(elem, ast.AST) and walk(elem):
                            return True
                        i += 1
                elif isinstance(val, ast.AST) and walk(val):
                    return True
            return False
        return walk(parent)

    def _contains(self, root: ast.AST, target: ast.AST) -> bool:
        """Return True if target occurs anywhere in root's subtree."""
        if root is target:
            return True
        for _, v in ast.iter_fields(root):
            if isinstance(v, ast.AST):
                if self._contains(v, target):
                    return True
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, ast.AST) and self._contains(it, target):
                        return True
        return False

    def _find_owner_in_span(self, span: list[ast.AST], anchor: ast.AST) -> ast.AST | None:
        """Return the first node in span whose subtree contains anchor."""
        for n in span:
            if self._contains(n, anchor):
                return n
        return None


    # ---------------- plan replacements / removals for a list -------------


    # def _plan(self, seq: list[ast.AST]) -> tuple[dict[int, list[ast.AST]], set[int]]:
    #     """Plan replacements and removals for a sibling list.

    #     Scans left-to-right for non-overlapping matches of `self.pattern` in `seq`,
    #     records matches, and computes:
    #     - a mapping from anchor node id -> replacement list
    #     - a set of node ids to remove

    #     Args:
    #         seq: The list of sibling AST nodes to scan.

    #     Returns:
    #         tuple[dict[int, list[ast.AST]], set[int]]: (replacements, removals).
    #     """
    #     repl: dict[int, list[ast.AST]] = {}
    #     remove: set[int] = set()
    #     i = 0
    #     # while i < len(seq):
    #     #     mtch = _match_patterns(self.pattern, seq, i, {})
    #     #     if not mtch:
    #     #         i += 1
    #     #         continue

    #     #     bindings, new_pos = mtch[0]
    #     #     self._record(bindings)
    #     #     span_nodes = seq[i:new_pos]

    #     #     for key, action in self.actions.items():
    #     #         if key not in bindings:
    #     #             continue

    #     #         collected = self._as_nodes(bindings[key])
    #     #         if not collected:
    #     #             continue

    #     #         anchor = collected[0]
    #     #         container = anchor if anchor in span_nodes else span_nodes[0]

    #     #         if action is None:
    #     #             remove.update(id(n) for n in collected)
    #     #         else:
    #     #             r = action(bindings) or []
    #     #             if not isinstance(r, list):
    #     #                 raise TypeError('Handler must return list[ast.AST].')
    #     #             repl[id(container)] = r
    #     #             remove.update(id(n) for n in collected[1:])

    #     #     i = new_pos

    #     # return repl, remove
    #     while i < len(seq):
    #         mtch = _match_patterns(self.pattern, seq, i, {})
    #         if not mtch:
    #             i += 1
    #             continue

    #         bindings, new_pos = mtch[0]
    #         self._record(bindings)
    #         span_nodes = seq[i:new_pos]

    #         for key, action in self.actions.items():
    #             if key not in bindings:
    #                 continue

    #             collected = self._as_nodes(bindings[key])
    #             if not collected:
    #                 continue

    #             anchor = collected[0]

    #             if anchor in span_nodes:
    #                 # anchor is the list element itself: replace/remove in the list
    #                 if action is None:
    #                     remove.update(id(n) for n in collected)
    #                 else:
    #                     r = action(bindings) or []
    #                     if not isinstance(r, list):
    #                         raise TypeError('Handler must return list[ast.AST].')
    #                     repl[id(anchor)] = r
    #                     remove.update(id(n) for n in collected[1:])
    #             else:
    #                 # anchor is nested inside the matched subtree; mutate the subtree in place
    #                 container = span_nodes[0]
    #                 if action is None:
    #                     # best-effort: only possible if anchor is in a list field
    #                     ok_any = False
    #                     for n in collected:
    #                         ok_any |= self._delete_within(container, n)
    #                     if not ok_any:
    #                         raise ValueError('Cannot delete non-list child node in-place.')
    #                 else:
    #                     r = action(bindings) or []
    #                     if not isinstance(r, list):
    #                         raise TypeError('Handler must return list[ast.AST].')
    #                     if not self._replace_within(container, anchor, r):
    #                         raise ValueError('Could not locate collected child in matched subtree for in-place replacement.')

    #         i = new_pos

    #     return repl, remove

    def _plan(self, seq: list[ast.AST]) -> tuple[dict[int, list[ast.AST]], set[int]]:
        repl: dict[int, list[ast.AST]] = {}
        remove: set[int] = set()
        i = 0
        while i < len(seq):
            mtch = _match_patterns(self.pattern, seq, i, {})
            if not mtch:
                i += 1
                continue

            bindings, new_pos = mtch[0]
            self._record(bindings)
            span_nodes = seq[i:new_pos]

            for key, action in self.actions.items():
                if key not in bindings:
                    continue

                collected = self._as_nodes(bindings[key])
                if not collected:
                    continue

                list_anchors = [n for n in collected if n in span_nodes]

                if action is None:
                    # Remove only list elements; ignore nested nodes
                    if list_anchors:
                        remove.update(id(n) for n in list_anchors)
                    else:
                        # Best-effort nested delete, but soft-fail if not in a list field
                        owner = self._find_owner_in_span(span_nodes, collected[0]) or span_nodes[0]
                        for n in collected:
                            self._delete_within(owner, n)
                    continue

                # Replacement
                r = action(bindings) or []
                if not isinstance(r, list):
                    raise TypeError('Handler must return list[ast.AST].')

                if list_anchors:
                    # Replace the first list element anchor and remove the rest
                    repl[id(list_anchors[0])] = r
                    remove.update(id(n) for n in list_anchors[1:])
                else:
                    # Replace nested child within its owner
                    owner = self._find_owner_in_span(span_nodes, collected[0]) or span_nodes[0]
                    # Try each collected node until one is found in owner
                    replaced = False
                    for n in collected:
                        if self._replace_within(owner, n, r):
                            replaced = True
                            break
                    if not replaced:
                        raise ValueError('Could not locate collected child in matched subtree for in-place replacement.')

            i = new_pos

        return repl, remove

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Transform children, then apply sequential matching in list fields."""
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                # 1) visit children
                children: list[ast.AST] = []
                for v in old_value:
                    if isinstance(v, ast.AST):
                        nv = self.visit(v)
                        if nv is None:
                            continue
                        if not isinstance(nv, ast.AST):
                            raise TypeError('List fields must contain AST nodes.')
                        children.append(nv)
                    else:
                        children.append(v)

                # 2) plan and splice sequential replacements/removals
                if children and all(isinstance(c, ast.AST) for c in children):
                    replace, remove = self._plan(children)
                    new_children: list[ast.AST] = []
                    for ch in children:
                        rid = id(ch)
                        if rid in replace:
                            new_children.extend(replace[rid])
                        elif rid not in remove:
                            new_children.append(ch)
                    setattr(node, field, new_children)
                else:
                    setattr(node, field, children)

            elif isinstance(old_value, ast.AST):
                visited = self.visit(old_value)
                if isinstance(visited, ast.AST):
                    rep = self._maybe_replace(visited)
                    if rep is not None:
                        new_node = self._normalize_replace_for_nonlist(rep, field)
                        setattr(node, field, new_node)
                else:
                    # Deleting or expanding is not valid for non-list fields.
                    setattr(node, field, old_value)

        return node

    # def generic_visit(self, node: ast.AST) -> ast.AST:
    #     for field, old_value in ast.iter_fields(node):
    #         if isinstance(old_value, list):
    #             new_values: list[ast.AST] = []
    #             changed = False
    #             for value in old_value:
    #                 if isinstance(value, ast.AST):
    #                     visited = self.visit(value)
    #                     rep: ReplaceResult = self._maybe_replace(visited)
    #                     if rep is None:
    #                         changed = True
    #                         continue
    #                     if isinstance(rep, list):
    #                         new_values.extend(rep)
    #                         changed = True
    #                     elif isinstance(rep, ast.AST):
    #                         new_values.append(rep)
    #                         changed = True
    #                 else:
    #                     new_values.append(value)
    #             if changed:
    #                 setattr(node, field, new_values)
    #         elif isinstance(old_value, ast.AST):
    #             visited = self.visit(old_value)
    #             rep: ReplaceResult = self._maybe_replace(visited)
    #             if rep is not None:
    #                 new_node = self._normalize_replace_for_nonlist(rep, field)
    #                 setattr(node, field, new_node)
    #     return node

    def _maybe_replace(self, node: ast.AST) -> ast.AST | list[ast.AST] | None:
        """
        Try to match `self.pattern` against just `node` and apply actions.

        Returns
        -------
            ast.AST | list[ast.AST] | None:
                - ast.AST: replace this node with a single node
                - list[ast.AST]: splice multiple nodes (only valid in list fields)
                - None: delete this node
        """
        res = _match_patterns(self.pattern, [node], 0, {})
        if not res:
            return None

        bindings, _ = res[0]
        self._record(bindings)

        did_inplace = False

        for key, action in self.actions.items():
            if key not in bindings:
                continue

            collected = self._as_nodes(bindings[key])

            # Case 1: the action targets THIS node -> replace/delete the node itself
            if node in collected:
                if action is None:
                    return None
                repl = action(bindings) or []
                if not isinstance(repl, list):
                    raise TypeError('Handler must return list[ast.AST].')
                if len(repl) == 0:
                    return None
                if len(repl) == 1:
                    return repl[0]
                return repl  # multi-element expansion only valid in list fields

            # Case 2: the action targets nested anchors -> mutate in place
            if action is None:
                for n in collected:
                    did_inplace |= self._delete_within(node, n)
            else:
                repl = action(bindings) or []
                if not isinstance(repl, list):
                    raise TypeError('Handler must return list[ast.AST].')
                for n in collected:
                    did_inplace |= self._replace_within(node, n, repl)

        # If we only did in-place nested edits, keep this node (signal "no top-level replace")
        return None

    
class BottomUpPatternTransformer(ast.NodeTransformer):
    """
    Like PatternTransformer, but applies pattern matching *after* visiting all children.
    This enables bottom-up rewriting (i.e., transforming from the leaves upward).
    """
    def __init__(
        self,
        pattern: Sequence[Pattern],
        actions: dict[str, Callable[[dict[str, Any]], list[ast.AST]] | None],
    ):
        super().__init__()
        self.pattern = pattern
        self.actions = actions
        self.matches: list[dict[str, Any]] = []

    def _record(self, bindings: dict[str, Any]) -> None:
        self.matches.append(bindings)

    @staticmethod
    def _as_nodes(value: Any) -> list[ast.AST]:
        """
        Extract AST nodes from arbitrarily nested values.

        Returns:
            list[ast.AST]: All AST nodes found within value.
        """
        out: list[ast.AST] = []
        def rec(v: Any) -> None:
            if isinstance(v, ast.AST):
                out.append(v)
            elif isinstance(v, list):
                for it in v:
                    rec(it)
            elif isinstance(v, dict):
                for it in v.values():
                    rec(it)
        rec(value)
        return out

    # def _as_nodes(self, value: Any) -> list[ast.AST]:
    #     if isinstance(value, ast.AST):
    #         return [value]
    #     if isinstance(value, list):
    #         return [n for n in value if isinstance(n, ast.AST)]
    #     if isinstance(value, dict):
    #         n = value.get("_node")
    #         return [n] if isinstance(n, ast.AST) else []
    #     return []

    def visit(self, node: ast.AST) -> ast.AST | None:
        # First, transform children recursively (bottom-up)
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, ast.AST):
                        new_item = self.visit(item)
                        if new_item is None:
                            continue
                        elif isinstance(new_item, list):
                            new_list.extend(new_item)
                        else:
                            new_list.append(new_item)
                    else:
                        new_list.append(item)
                setattr(node, field, new_list)
            elif isinstance(value, ast.AST):
                new_value = self.visit(value)
                setattr(node, field, new_value)

        # Then apply pattern matching to this node
        res = _match_patterns(self.pattern, [node], 0, {})
        if res:
            bindings, _ = res[0]
            self._record(bindings)
            for key, action in self.actions.items():
                if key in bindings:
                    if action is not None:
                        replacements = action(bindings)
                        if replacements:
                            return replacements[0] if len(replacements) == 1 else replacements
                    else:
                        return None  # delete the matched node

        return node


class PatternFinder(ast.NodeVisitor):
    """Collect bindings for every occurrence of *pattern* in an AST."""

    def __init__(self, pattern: Sequence[Pattern]):
        super().__init__()
        self.visited: set[int] = set()
        self.pattern = pattern
        self.matches: list[dict[str, Any]] = []

    def generic_visit(self, node: ast.AST):
        if id(node) in self.visited: return

        self.visited.add(id(node))
        res = _match_patterns(self.pattern, [node], 0, {})
        if res:
            self.matches.append(res[0][0])
        for _, val in ast.iter_fields(node):
            if isinstance(val, list):
                if len(self.pattern) > 1:
                    self._scan_list(val)
                for elem in val:
                    if isinstance(elem, ast.AST):
                        self.visit(elem)
            elif isinstance(val, ast.AST):
                self.visit(val)

    def _scan_list(self, seq: list[ast.AST]):
        i = 0
        while i < len(seq):
            res = _match_patterns(self.pattern, seq, i, {})
            if res:
                binds, new_pos = res[0]
                self.matches.append(binds)
                i = new_pos
            else:
                i += 1

class SingleOccurrenceFinder(ast.NodeVisitor):
    """
    Quickly checks whether a single match of the given pattern sequence exists in an AST.
    Returns True on the first match, short-circuiting the traversal.
    """

    match_node: ast.AST | None

    def __init__(self, pattern: Sequence[Pattern]):
        super().__init__()
        self.match_node = None
        self.pattern = pattern
        self.found = False

    def visit(self, node: ast.AST):
        if self.found:
            return  # short-circuit: we've already found a match

        res = _match_patterns(self.pattern, [node], 0, {})
        if res:
            self.found = True
            self.match_node = node
            return

        # Continue traversal
        for _, val in ast.iter_fields(node):
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, ast.AST):
                        self.visit(item)
                        if self.found:
                            return
            elif isinstance(val, ast.AST):
                self.visit(val)
                if self.found:
                    return

    def found_match(self) -> bool:
        return self.found