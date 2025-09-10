from __future__ import annotations
from typing import TYPE_CHECKING, Type

from casys.dsl._core.schema.schema_base import DirtySchema, Schema, SchemaPostProcessor

if TYPE_CHECKING:
    from casys.dsl._core.schema.soa_layout import SoaField
    from casys.dsl._core.schema.base_components import GroupSchema
    from casys.dsl._core.schema.schema_base import SchemaPostProcessor

class WorldSchema(Schema):
    groups: dict[str,GroupSchema]
    post_processors: dict[Type[Schema], SchemaPostProcessor]

    def __init__(self, groups: dict[str,GroupSchema]) -> None:
        self.groups = groups
        self.post_processors = {}

    def resolve_fields(self) -> dict[str,SoaField]:

        def handle_dirty(node: Schema):
            node_class = node.__class__
            if isinstance(node, DirtySchema):
                if node_class not in self.post_processors:
                    post_processor = node.get_post_processor()()
                    self.post_processors[node.__class__] = post_processor
                else:
                    post_processor = self.post_processors[node.__class__]

                post_processor.process(node)
                node.unset_dirty()

            if node.has_dirty_offspring:
                for child in node.get_children():
                    handle_dirty(child)

                node.clear_has_dirty_offspring()

        for group in self.groups.values():
            if group.has_dirty_offspring:
                handle_dirty(group)

        for processor in self.post_processors.values():
            processor.finalize()

        soa_fields = {}
        for group in self.groups.values():
            group.insert_resolved_fields(soa_fields)

        return soa_fields
    
    def get_children(self) -> list[Schema]:
        return list(self.groups.values())