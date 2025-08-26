from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casys.dsl._core.ir import Ir_CaSys

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.debug.ast_timeline_tracking import get_tracker

class PipelineSection(TranspilerModule):
    section_phase: str | None = None
    ir: Ir_CaSys
    pipeline_overwrite: list[TranspilerModule | None] | None

    @property
    def pipeline(self) -> list[TranspilerModule | None]:
        if self.pipeline_overwrite: return self.pipeline_overwrite
        return []

    def __init__(self, ir, section_phase: str | None = None, pipeline: list[TranspilerModule | None] | None = None) -> None:
        self.ir = ir
        if section_phase:
            self.section_phase = section_phase
        self.pipeline_overwrite = pipeline

    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        if self.section_phase:
            trkr.enter_phase(self.section_phase)

        pipeline = self.pipeline
        for module in pipeline:
            if module is None: continue
            module.process(ir)

        if self.section_phase:
            trkr.exit_phase()
        
