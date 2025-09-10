from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Type

from casys.dsl._core.schema.soa_layout import SoaField

class Schema(ABC):
    """Base class for logical structure defining components that compile to SoA fields."""

    name: str

    @property
    def parent(self) -> None | Schema:
        return getattr(self,'_parent',None)

    @parent.setter 
    def parent(self, value: Schema | None):
        setattr(self,'_parent',value)

    def set_has_dirty_offspring(self) -> None:
        """Mark as having dirty offspring and bubble up once."""
        if not getattr(self, '_has_dirty_offspring', False):
            setattr(self, '_has_dirty_offspring', True)
            p = self.parent
            if p is not None:
                p.set_has_dirty_offspring()

    def clear_has_dirty_offspring(self) -> None:
        """Unset has dirty offspring flag."""
        if getattr(self, '_has_dirty_offspring', False):
            setattr(self, '_has_dirty_offspring', False)

    @property
    def has_dirty_offspring(self) -> bool:
        """Whether this node's children require processing."""
        return getattr(self, '_has_dirty_offspring', False)

    def get_children(self) -> list[Schema]:
        """
        Returns the children of any schema nodes that implement a child-hierarchy feature.
        The base Schema node does not have any children so a empty array is returned
        """
        return []
    
    def canonical_name(self, sep: str = '_') -> str:
        """Hierarchical name built from the ancestor chain."""
        parts: list[str] = []
        node: Schema | None = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        parts.reverse()
        return sep.join(parts)

    @abstractmethod
    def resolve_fields(self) -> dict[str,SoaField]:
        """Return the canonical collection of SoA layout fields that this schema component defines."""
        raise NotImplementedError

    def insert_resolved_fields(self, target_dict: dict[str,SoaField]) -> dict[str,SoaField]:
        """Extend a collection of fields with the fields from this Schema object."""
        target_dict.update(self.resolve_fields())
        return target_dict
    
    def get_flattened_tree(self, filter_predicate: Callable[[Schema], bool] | None = None) -> list[Schema]:
        """Get a flat list of all schema nodes in a schema tree. Optionally filtered by `filter_predicate`"""
        all_nodes: list[Schema] = []

        def visit_node(node: Schema):
            for child in node.get_children():
                if filter_predicate is None or filter_predicate(child):
                    all_nodes.append(child)
                visit_node(child)

        if filter_predicate is None or filter_predicate(self):
            all_nodes.append(self)

        visit_node(self)

        return all_nodes
    
class DirtySchema(Schema):

    @abstractmethod
    def get_post_processor(cls) -> Type[SchemaPostProcessor]:
        """Returns the required post processor for this DirtySchema type."""
        raise NotImplementedError

    @property
    def is_dirty(self) -> bool:
        """Whether this node requires processing."""
        return getattr(self, '_dirty', False)

    def set_dirty(self) -> None:
        """Set the dirty flag and notify potential parent."""
        if not getattr(self, '_dirty', False):
            setattr(self, '_dirty', True)
            p = self.parent
            if p is not None:
                p.set_has_dirty_offspring()

    def unset_dirty(self) -> None:
        """Unset the dirty flag on this node."""
        setattr(self, '_dirty', False)

    
class SchemaPostProcessor[T_s: Schema](ABC):
    """Base class for schema node post processors"""

    @abstractmethod
    def process(self, target: T_s):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Allows for pre processors to use a two phase approach"""
        pass