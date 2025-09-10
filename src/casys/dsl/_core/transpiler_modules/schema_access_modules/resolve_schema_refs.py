from __future__ import annotations

from typing import Sequence
import ast

from casys.dsl._core.schema.soa_layout import SoaLayout
from casys.dsl._core.transpiler_modules.schema_access_modules.transform_vbool_fields import handle_vbool_field_schema_refs
from casys.spec.cac_type import FieldSchema

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Filter
from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS, MDK_SOA_LAYOUT
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_TYPE_BINNED_SCHEMA_REFS
from casys.spec.virtual_types import VirtualBoolField

class ResolveSchemaRefs(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Resolving and transforming schema refs')

        soa_layout = SoaLayout(ir.world_schema.resolve_fields())
        ir.metadata.set(MDK_SOA_LAYOUT, soa_layout)

        dims: Sequence[int] = ir.metadata.get(MDK_DIMS)

        for kname, kernel in ir.kernels.items():
            type_binned_schema_refs = kernel.metadata.get(MDK_TYPE_BINNED_SCHEMA_REFS)

            for schema_type, schema_refs in type_binned_schema_refs.items():
                match schema_type.__name__:
                    case FieldSchema.__name__: handle_field_schema_refs(schema_refs, kernel.ir_ast)
                    case VirtualBoolField.__name__: handle_vbool_field_schema_refs(schema_refs, kernel.ir_ast)

            trkr.add_snapshot(
                ast_node=kernel.ir_ast,
                tags=(f_tag_kernel(kname), f_tag_transpiler_module(self)),
            )

        trkr.exit_phase()


def handle_field_schema_refs(
    schema_refs: list[casys_ast.Cs_SchemaRef[FieldSchema]],
    ir_ast: ast.AST
):
    ptrn = [Filter(lambda n: isinstance(n, casys_ast.Cs_SchemaRef) and n in schema_refs, 'node')] 
    PatternTransformer(
        ptrn,
        {'node': lambda m: [snippet_schema_field(m['node'])]}
    ).visit(ir_ast)


def snippet_schema_field(node: casys_ast.Cs_SchemaRef[FieldSchema]) -> casys_ast.Cs_SoaFieldRef:
    return casys_ast.Cs_SoaFieldRef(node.s.resolve_field(), node.ctx)