from typing import Any
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaKernel, Ir_CaSys, Ir_SimStepFunc
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_kernel, f_tag_transpiler_module

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import (
    PatternFinder, 
    PatternTransformer, 
    BottomUpPatternTransformer, 
    Collect, 
    Bind, 
    NodePattern, 
    Filter,
)
from casys._ast_pattern_utils.ast_pattern_templates import match_in_expr

from casys.dsl._core import core_macros

from casys.dsl._core import macro_handlers # Needs to be last to ensure macro handlers are registered

def handle_macros_recursive(name, target_ir: Ir_CaKernel | Ir_SimStepFunc, sys_ir: Ir_CaSys):
    marking_module = MarkMacros(name,target_ir)
    handling_module = HandleMacros(name,target_ir)

    handled_nodes: set[int] = set()

    finder = None
    marking_module.process(sys_ir)
    while finder is None or finder.matches:
        ptrn_un_handled_macros = [
            Collect(
                pattern=Filter(lambda n : isinstance(n, casys_ast.Cs_Macro) and id(n) not in handled_nodes),
                key='macro'
            )
        ]

        finder = PatternFinder(ptrn_un_handled_macros)
        finder.visit(target_ir.ir_ast)
        if finder.matches:
            handled_nodes.update(id(m['macro']) for m in finder.matches)
            handling_module.process(sys_ir)
            marking_module.process(sys_ir)
        else:
            break


class MarkMacros[T_ir: Ir_CaKernel | Ir_SimStepFunc](TranspilerModule):
    target_name: str
    target_ir: T_ir

    def __init__(self, target_name: str, target_ir: T_ir): 
        self.target_name = target_name
        self.target_ir = target_ir

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()

        ptrn_macro_calls = [
            Collect(pattern=NodePattern(
                node_type=ast.Call,
                func=NodePattern(
                    node_type=ast.Name,
                    id=Filter(predicate=lambda n: n in core_macros._MACROS, key='name')
                )
            ), key='call')
        ]

        ptrn_expr_calls = [
            Collect(match_in_expr(
            Collect(pattern=NodePattern(
                    node_type=casys_ast.Cs_Macro,
                    func=NodePattern(
                        node_type=ast.Name,
                        id=Bind('name')
                    )
                ),
                key='call')
            ),'expr')
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

        PatternTransformer(
            pattern=ptrn_macro_calls,
            actions={ 'call': mark_macro }
        ).visit(self.target_ir.ir_ast)

        # Remove enclosing Expr node
        PatternTransformer(
            pattern=ptrn_expr_calls,
            actions={ 'expr': lambda m:[m['call']]}
        ).visit(self.target_ir.ir_ast)

        trkr.add_snapshot(
            ast_node=self.target_ir.ir_ast,
            tags=(
                f_tag_kernel(self.target_name) if isinstance(self.target_ir, Ir_CaKernel) else TAG_STEP_FUNC,
                f_tag_transpiler_module(self),
            )
        )


    def get_phase_name(self) -> str:
        return f'Macro marking ({self.target_name})'

    
class HandleMacros[T_ir: Ir_CaKernel | Ir_SimStepFunc](TranspilerModule):
    target_name: str
    target_ir: T_ir

    def __init__(self, target_name: str, target_ir: T_ir): 
        self.target_name = target_name
        self.target_ir = target_ir

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

        ptrn_static_if_blocks = [
            Collect(NodePattern(
                node_type=ast.If,
                test = NodePattern(
                        node_type=ast.Constant,
                        value=Filter(lambda v: isinstance(v, bool), 'test')
                ),
                body=Bind('body'),
                orelse=Bind('orelse')
            ), 'node')
        ]

        ptrn_static_if_expressions = [
            Collect(NodePattern(
                node_type=ast.IfExp,
                test=NodePattern(
                        node_type=ast.Constant,
                        value=Filter(lambda v: isinstance(v, bool), 'test')
                ),
                body=Bind('body'),
                orelse=Bind('orelse'),
            ), 'node')
        ]

        ptrn_if_blocks = [
            Collect(NodePattern(
                node_type=ast.If,
                body=Bind('body'),
                orelse=Bind('orelse'),
            ), 'node')
        ]

        transformers = (
            BottomUpPatternTransformer(
                pattern=ptrn_macro_calls,
                actions={
                    'call': lambda m: core_macros._MACRO_HANDLERS[m['name']](m['call'], ir)
                }
            ),

            # Dissolve static conditionals
            # This allows easy #if #endif like behavior
            BottomUpPatternTransformer(
                pattern=ptrn_static_if_expressions,
                actions={
                    'node': lambda m: [m['body']] if m['test'] else [m['orelse']]
                }
            ),
            BottomUpPatternTransformer(
                pattern=ptrn_static_if_blocks,
                actions={
                    'node': lambda m: m['body'] if m['test'] else m['orelse']
                }
            ),
            # Dissolve empty if blocks
            BottomUpPatternTransformer(
                pattern=ptrn_if_blocks,
                actions={
                    'node': lambda m: [m['node']] if m['body'] else m['orelse']
                }
            )
        )

        for tf in transformers: tf.visit(self.target_ir.ir_ast)

        trkr.add_snapshot(
            ast_node=self.target_ir.ir_ast,
            tags=(
                f_tag_kernel(self.target_name) if isinstance(self.target_ir, Ir_CaKernel) else TAG_STEP_FUNC,
                f_tag_transpiler_module(self),
            )
        )

    def get_phase_name(self) -> str:
        return f'Macro handling ({self.target_name})'