from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from casys.spec.cac_type import FieldSchema, GroupSchema

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Filter
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_TYPE_BINNED_SCHEMA_REFS


class InsertSchemaRefs(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Marking schema refs')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        call_permutations: dict[str, list[GroupSchema]]
        
        def snippet_subscript(schema_ref: casys_ast.Cs_SchemaRef) -> ast.AST:
            slice_node = ast.Tuple(elts=[
                casys_ast.Cs_KPos(i) for i in range(len(dims))
            ])

            new_node = ast.Subscript(schema_ref,
                slice = slice_node,
                ctx = schema_ref.ctx
            )
            casys_ast.copy_meta(new_node, schema_ref).verified_bounds = True
            
            return new_node
        
        def get_attr_path(attr_node: ast.Attribute, path: list[str], node_path: list[ast.AST]) -> tuple[list[str], list[ast.AST]]:
            path.insert(0, attr_node.attr)
            if isinstance(attr_node.value, ast.Attribute):
                node_path.insert(0, attr_node)
                path, node_path = get_attr_path(attr_node.value, path, node_path)
            elif isinstance(attr_node.value, ast.Name):
                path.insert(0, attr_node.value.id)
                node_path.insert(0, attr_node.value)
            return path, node_path
        
    
        for name, kernel in ir.kernels.items():
            layer_args = kernel.base.layer_args
            call_permutations = ir.step_func.base.kcall_permutations[name]

            type_binned_schema_refs = defaultdict(list)

            replace_map: dict[int, ast.AST] = {}

            def handle_attr(node: ast.Attribute, subscript_node: ast.Subscript | None = None):
                attr_path, attr_path_nodes = get_attr_path(node, [], [])

                for n in attr_path_nodes:
                    skip.add(id(n))

                if attr_path[0] in layer_args:
                    check_slice = attr_path[1:]
                    check_slice_nodes = attr_path_nodes[1:]
                    schema_node = call_permutations[attr_path[0]][0]
                    while (
                        check_slice
                        and isinstance(schema_node, GroupSchema)
                        and check_slice[0] in schema_node.fields
                    ):
                        schema_node = schema_node.fields[check_slice[0]]
                        check_slice = check_slice[1:]
                        check_slice_nodes = check_slice_nodes[1:]
                    
                    if check_slice:
                        raise TranspileError(f"'{schema_node.name}' does not have any field '{check_slice[0]}'", check_slice_nodes[0])
                    
                    if isinstance(node.ctx, ast.Del):
                        raise TranspileError(f" Cannot delete '{schema_node.name}'", node)
                    
                    if not isinstance(schema_node, FieldSchema):
                        raise TranspileError(f"Cannot read non-field '{schema_node.name}'", node)
                    
                    new_node_ctx = node.ctx if subscript_node is None else subscript_node.ctx

                    new_node = casys_ast.Cs_SchemaRef(schema_node, new_node_ctx.__class__())
                    casys_ast.copy_meta(new_node,node)

                    type_binned_schema_refs[schema_node.__class__].append(new_node)

                    if subscript_node is None:
                        new_node = snippet_subscript(new_node)

                    replace_map[id(node)] = new_node

            skip: set[int] = set()
            for node in ast.walk(kernel.ir_ast):
                if id(node) in skip:
                    continue
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Attribute):
                        handle_attr(node.value, node)
                        skip.add(id(node))
                        skip.add(id(node.value))
                elif isinstance(node, ast.Attribute):
                    handle_attr(node)
                    skip.add(id(node))

            (tf:=PatternTransformer([Filter(lambda n: isinstance(n,ast.Attribute) and id(n) in replace_map, 'node')], {
                'node': lambda m: [replace_map[id(m['node'])]]
            })).visit(kernel.ir_ast)

            kernel.metadata.set(MDK_TYPE_BINNED_SCHEMA_REFS, type_binned_schema_refs)

            if tf.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name),f_tag_transpiler_module(self)),
                    ast_node=kernel.ir_ast,
                    metadata=kernel.metadata
                )

        trkr.exit_phase()