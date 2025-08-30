from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

from collections import Counter

import ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, Collect, Bind, NodePattern, Filter, OneOrMore
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_READONLY

class ValidateReadonly(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        readonly: set[str]

        for name, kernel in ir.kernels.items():
            readonly = kernel.metadata.get(MDK_READONLY)

            tally = Counter()

            ptrn_frozen_assigns = [
                Collect( NodePattern(
                        node_type=ast.Name,
                        id=Filter(lambda n: n in readonly, 'id'),
                        ctx=Filter(lambda x: isinstance(x,ast.Store))
                ), key='name')
            ]

            self.current_kernel = name
            (finder:=PatternFinder(ptrn_frozen_assigns)).visit(kernel.ir_ast)
            for m in finder.matches:
                t: str = m['id']
                n: ast.Name = m['name']
                tally[t] += 1
                if tally[t] > 1:
                    raise TranspileError(f"Tried to modify readonly variable '{t}'", n)