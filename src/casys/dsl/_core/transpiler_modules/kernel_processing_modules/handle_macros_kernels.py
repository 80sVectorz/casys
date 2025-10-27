from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker

from casys.dsl._core.transpiler_modules.macro_handling import handle_macros_recursive # Needs to be last to ensure macro handlers are registered

class HandleMacrosKernels(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Handling Macros')
        for name,kernel in ir.kernels.items():
            trkr.enter_phase(f'Handling Macros ({name})')
            handle_macros_recursive(name, kernel, ir)
            trkr.exit_phase()
        trkr.exit_phase()
