from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from casys.spec.ca_layer_spec import CaLayerSpec

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, PatternFinder, Collect, Bind, NodePattern, Filter
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS

class MarkLayerFieldRefs(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Marking layer field accesses')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)
        call_permutations: dict[str, list[CaLayerSpec]]

        def mark_buffer_ref(m: dict[str, Any]) -> list[ast.AST]:
            attribute_node: ast.Attribute = m['attribute_node']
            local_layer_name: str = m['layer_ref']
            field_name: str = m['field']

            layer_name = call_permutations[local_layer_name][0].name

            layer = kernel.base.layer_args[layer_name]

            if field_name not in layer.cact.fields:
                raise TranspileError(f'layer {layer_name} does not have a field called "{field_name}"', m['attribute_node'])

            new_node = casys_ast.Cs_LayerFieldRef(m['layer_ref'], m['field'], ctx=attribute_node.ctx) # type: ignore
            casys_ast.copy_meta(new_node,m['attribute_node'])
            
            return [new_node]
        
        def handle_no_subscript(m: dict[str, Any]) -> list[ast.AST]:
            layer_field_ref: casys_ast.Cs_LayerFieldRef = m['layer_ref']

            slice_node = ast.Tuple(elts=[
                casys_ast.Cs_KPos(i) for i in range(len(dims))
            ])

            new_node = ast.Subscript(layer_field_ref,
                slice = slice_node,
                ctx = layer_field_ref.ctx
            )
            casys_ast.copy_meta(new_node, layer_field_ref).verified_bounds = True
            
            return [new_node]

        for name, kernel in ir.kernels.items():
            layer_args = kernel.base.layer_args
            call_permutations = ir.step_func.base.kcall_permutations[name]

            ptrn_layer_refs = [
                Collect( NodePattern(
                    ast.Attribute,
                    value=NodePattern(ast.Name,id=Filter(lambda n : n in layer_args, 'layer_ref')),
                    attr=Bind('field')
                ), 'attribute_node')
            ]

            (tf:=PatternTransformer(ptrn_layer_refs, {
                'attribute_node': mark_buffer_ref
            })).visit(kernel.ir_ast)


            ptrn_subscript = [
                NodePattern(
                node_type=ast.Subscript,
                value=Collect( NodePattern(
                    casys_ast.Cs_LayerFieldRef,
                ), 'layer_ref')
                )
            ]

            (finder:=PatternFinder(ptrn_subscript)).visit(kernel.ir_ast)
            subscripted_refs = set(id(m['layer_ref']) for m in finder.matches)

            ptrn_no_subscript = [
                Filter(lambda n: isinstance(n, casys_ast.Cs_LayerFieldRef) and id(n) not in subscripted_refs, 'layer_ref')
            ]

            PatternTransformer(ptrn_no_subscript, {
                'layer_ref': handle_no_subscript
            }).visit(kernel.ir_ast)

            if tf.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name),f_tag_transpiler_module(self)),
                    ast_node=kernel.ir_ast
                )

        trkr.exit_phase()