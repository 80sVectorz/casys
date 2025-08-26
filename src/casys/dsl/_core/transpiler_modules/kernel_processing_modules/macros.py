from typing import Any
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaKernel, Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, PatternTransformer, BottomUpPatternTransformer, Collect, Bind, NodePattern, Filter

from casys.dsl._core import core_macros

from casys.dsl._core import macro_handlers # Needs to be last to ensure macro handlers are registered

class HandleMacrosRecursive(TranspilerModule):

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Handling Macros')
        for name,kernel in ir.kernels.items():
            trkr.enter_phase(f'Handling Macros ({name})')

            marking_module = MarkMacros(name,kernel)
            handling_module = HandleMacros(name,kernel)

            handled_nodes: set[int] = set()

            finder = None
            marking_module.process(ir)
            while finder is None or finder.matches:
                ptrn_un_handled_macros = [
                    Collect(
                        pattern=Filter(lambda n : isinstance(n, casys_ast.Cs_Macro) and id(n) not in handled_nodes),
                        key='macro'
                    )
                ]

                finder = PatternFinder(ptrn_un_handled_macros)
                finder.visit(kernel.ir_ast)
                if finder.matches:
                    handled_nodes.update(id(m['macro']) for m in finder.matches)
                    handling_module.process(ir)
                    marking_module.process(ir)
                else:
                    break

            trkr.exit_phase()
        trkr.exit_phase()


class MarkMacros(TranspilerModule):
    target_kernel: str
    target_kernel_ir: Ir_CaKernel

    def __init__(self, target_kernel: str, target_kernel_ir: Ir_CaKernel): 
        self.target_kernel = target_kernel
        self.target_kernel_ir = target_kernel_ir

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()

        ptrn_macro_calls = [
            Collect(pattern=NodePattern(
                    node_type=ast.Call,
                    func=NodePattern(
                        node_type=ast.Name,
                        id=Filter(predicate=lambda n: n in core_macros._MACROS, key='name')
                    )
                ),
                key='call'
            )
        ]

        def mark_macro(m: dict[str,Any]) -> list[ast.AST]:
            call_node: ast.Call = m['call']
            new_node = casys_ast.Cs_Macro(
                func=(c:=m['call']).func,
                handler=(h:=core_macros._MACRO_HANDLERS)[m['name']],
                is_default_handler=hasattr(h, '_is_default_handler'),
                args=c.args,
                keywords=c.keywords,
            ) if not isinstance(m['call'], casys_ast.Cs_Macro) else call_node

            casys_ast.copy_meta(new_node,call_node)

            return [new_node]

        (tf:=PatternTransformer(
            pattern=ptrn_macro_calls,
            actions={ 'call': mark_macro }
        )).visit(self.target_kernel_ir.ir_ast)

        trkr.add_snapshot(
            ast_node=self.target_kernel_ir.ir_ast,
            tags=(
                f_tag_kernel(self.target_kernel),
                f_tag_transpiler_module(self),
            )
        )

    def get_phase_name(self) -> str:
        return f'Macro marking ({self.target_kernel})'
    
class HandleMacros(TranspilerModule):
    target_kernel: str
    target_kernel_ir: Ir_CaKernel

    def __init__(self, target_kernel: str, target_kernel_ir: Ir_CaKernel): 
        self.target_kernel = target_kernel
        self.target_kernel_ir = target_kernel_ir

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()

        ptrn_macro_calls = [
            Collect(pattern=NodePattern(
                    node_type=casys_ast.Cs_Macro,
                    func=NodePattern(
                        node_type=ast.Name,
                        id=Bind('name')
                    )
                ),
                key='call'
            )
        ]

        (tf:=BottomUpPatternTransformer(
            pattern=ptrn_macro_calls,
            actions={
                'call': lambda m: core_macros._MACRO_HANDLERS[m['name']](m['call'], ir)
            }
        )).visit(self.target_kernel_ir.ir_ast)

        trkr.add_snapshot(
            ast_node=self.target_kernel_ir.ir_ast,
            tags=(
                f_tag_kernel(self.target_kernel),
                f_tag_transpiler_module('HandleMacros'),
            )
        )

    def get_phase_name(self) -> str:
        return f'Macro handling ({self.target_kernel})'