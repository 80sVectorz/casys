from typing import Any
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternTransformer, Collect, Bind, NodePattern, Filter, OneOrMore

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_DIMS
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_ALIASES, MDK_POS_VARS, MDK_READONLY
from casys.dsl.kernel_utils import (
    k_get_pos, k_get_dims
)

class HandleKGets(TranspilerModule):
    current_kernel: str = 'NOT STARTED'

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Handling KGets')

        ptrn_k_get_assign = [
            Collect(
                pattern=NodePattern(
                    ast.Assign,
                    targets=NodePattern(
                        ast.Tuple,
                        elts=OneOrMore(
                            Collect(
                                NodePattern(ast.Name),
                                'targets'
                            )
                        )
                    ),
                    value=Filter(
                        predicate=lambda n: (
                            isinstance(n, casys_ast.Cs_Macro)
                            and (func_name:=(NodePattern(ast.Name,id=Bind('name')).match(n.func, {}))) is not None
                            and func_name['name'] in (
                                k_get_pos.__name__,
                                k_get_dims.__name__,
                            )
                        ),
                        key='macro'
                    )
                ), key='assign',
            )
        ]

        dims: tuple[int,...] = ir.metadata.get(MDK_DIMS)

        pos_vars: dict[str,int]
        readonly: set[str]
        aliases: dict[str, ast.AST]

        def replace_assign(m: dict[str, Any]) -> list[ast.AST]:
            targets: list[ast.Name] = m['targets']
            assign_node: ast.Assign = m['assign']
            macro: casys_ast.Cs_Macro = m['macro']
            assert isinstance(macro.func, ast.Name)

            macro_name = macro.func.id

            readonly.update(t.id for t in targets)

            match macro_name:
                case k_get_pos.__name__:
                    pos_vars.update((t.id,i) for i,t in enumerate(targets))
                    new_node = ast.Assign(
                        targets=[ast.Tuple([t for t in targets[:len(dims)]], ast.Store())],
                        value=ast.Tuple(elts=(elts:=[
                            casys_ast.Cs_KPos(i) for i in range(len(targets))
                    ]), ctx=ast.Load()))

                    meta = casys_ast.copy_meta(new_node,assign_node)
                    meta.verified_bounds = True

                    aliases.update({t.id:e for t,e in zip(targets[:len(dims)],elts)})

                    return [new_node]

                case k_get_dims.__name__:
                    new_node = ast.Assign(
                        targets=[ast.Tuple([t for t in targets[:len(dims)]], ast.Store())],
                        value=ast.Tuple(elts=(elts:=[
                        casys_ast.Cs_AxisSize(i) for i in range(len(targets))
                    ]), ctx=ast.Load()))

                    aliases.update({t.id:e for t,e in zip(targets[:len(dims)],elts)})

                    return [new_node]
                
            return [assign_node]
        
        for name, kernel in ir.kernels.items():
            readonly = kernel.metadata.get(MDK_READONLY)
            pos_vars = kernel.metadata.get(MDK_POS_VARS)
            aliases = kernel.metadata.get(MDK_ALIASES)
            
            self.current_kernel = name
            (tf:=PatternTransformer(ptrn_k_get_assign, {'assign':replace_assign})).visit(kernel.ir_ast)
            if tf.matches:
                trkr.add_snapshot(
                    ast_node=kernel.ir_ast,
                    tags=(
                        f_tag_kernel(name),
                        f_tag_transpiler_module(self)
                    ),
                    metadata=kernel.metadata
                )

        trkr.exit_phase()

    def get_phase_name(self) -> str:
        return f'Handling k_get functions ({self.current_kernel})'