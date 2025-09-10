from typing import Sequence
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import SingleOccurrenceFinder, Collect, NodePattern, Filter, OneOrMore

class ValidateStrictKernels(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:

        ptrn_subscript = [
            Collect(
                pattern=NodePattern(
                node_type=ast.Subscript,
                    slice=Filter(lambda n: isinstance(n,ast.Tuple) and not casys_ast.get_meta(n).local_access == True),
                    ctx=NodePattern(ast.Store)
                ),
                key='subscript'
            ),
        ]

        for name, kernel in ir.kernels.items():
            self.current_kernel = name
            (finder:=SingleOccurrenceFinder(pattern=ptrn_subscript)).visit(kernel.ir_ast)
            if finder.found:
                raise TranspileError(f"Illegal buffer assign in kernel '{name}'", finder.match_node)