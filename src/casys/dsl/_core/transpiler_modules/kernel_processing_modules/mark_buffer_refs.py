from typing import Any, Sequence

from vispy.visuals.transforms import arg_to_array
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, PatternFinder, Collect, Bind, NodePattern, Filter, OneOrMore
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS

class MarkBufferRefs(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('MarkBufferRefs')

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)

        def mark_buffer_ref(m: dict[str, Any]) -> list[ast.AST]:
            attribute_node: ast.Attribute = m['attribute_node']
            buffer_name: str = m['buffer_name']
            field_name: str = m['field']

            buffer = kernel.base.buffers[buffer_name]

            if field_name not in buffer.cact._fields:
                raise TranspileError(f'Buffer {buffer_name} does not have a field called "{field_name}"', m['attribute_node'])

            new_node = casys_ast.Cs_BufferRef(m['buffer_name'], m['field'], ctx=attribute_node.ctx) # type: ignore
            casys_ast.copy_meta(new_node,m['attribute_node'])
            
            return [new_node]
        
        def handle_no_subscript(m: dict[str, Any]) -> list[ast.AST]:
            buffer_ref: casys_ast.Cs_BufferRef = m['buffer_ref']

            slice_node = ast.Tuple(elts=[
                casys_ast.Cs_KPos(i) for i in range(len(dims))
            ])

            new_node = ast.Subscript(buffer_ref,
                slice = slice_node,
                ctx = buffer_ref.ctx
            )
            casys_ast.copy_meta(new_node, buffer_ref).verified_bounds = True
            
            return [new_node]

        for name, kernel in ir.kernels.items():
            buffers = kernel.base.buffers
            ptrn_buffer_refs = [
                Collect( NodePattern(
                    ast.Attribute,
                    value=NodePattern(ast.Name,id=Filter(lambda n : n in buffers, 'buffer_name')),
                    attr=Bind('field')
                ), 'attribute_node')
            ]
            (tf:=PatternTransformer(ptrn_buffer_refs, {
                'attribute_node': mark_buffer_ref
            })).visit(kernel.ir_ast)


            ptrn_subscript = [
                NodePattern(
                node_type=ast.Subscript,
                value=Collect( NodePattern(
                    casys_ast.Cs_BufferRef,
                ), 'buffer_ref')
                )
            ]
            (finder:=PatternFinder(ptrn_subscript)).visit(kernel.ir_ast)
            subscripted_refs = set(id(m['buffer_ref']) for m in finder.matches)

            ptrn_no_subscript = [
                Filter(lambda n: isinstance(n, casys_ast.Cs_BufferRef) and id(n) not in subscripted_refs, 'buffer_ref')
            ]

            PatternTransformer(ptrn_no_subscript, {
                'buffer_ref': handle_no_subscript
            }).visit(kernel.ir_ast)

            if tf.matches:
                trkr.add_snapshot(
                    tags=(f_tag_kernel(name),f_tag_transpiler_module(self)),
                    ast_node=kernel.ir_ast
                )

        trkr.exit_phase()