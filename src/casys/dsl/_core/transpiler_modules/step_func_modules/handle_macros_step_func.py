from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaKernel, Ir_CaSys, Ir_SimStepFunc
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker, f_tag_kernel, f_tag_transpiler_module

from casys.dsl._core.transpiler_modules.macro_handling import handle_macros_recursive # Needs to be last to ensure macro handlers are registered

class HandleMacrosStepFunc(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Handling Macros')
        handle_macros_recursive(ir.step_func.base.func.__name__, ir.step_func, ir)
        trkr.exit_phase()