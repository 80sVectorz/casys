from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from casys.dsl._core.ir_metadata_specs.md_core_transpiler import MDK_SOA_LAYOUT
from casys.dsl._core.soa_field_usage_info_helper import UnfinishedSoaFieldUsageInfo
from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_kernels_base import MDK_SOA_FIELD_USAGE_INFO, SoaFieldUsageInfo

from collections import defaultdict

from casys.dsl._core import casys_ast

@dataclass(slots=True)
class ParallelGroup(casys_ast.Cs_ParallelGroup):
    soa_field_usage: SoaFieldUsageInfo | None = None
    original: casys_ast.Cs_ParallelGroup | None = None

class OptimizeParallelGrouping(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Optimize parallel grouping and double buffer swaps')

        soa_layout = ir.metadata.get(MDK_SOA_LAYOUT)

        pgroups: list[ParallelGroup] = []

        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup): continue
            assert not (node.swaps and node.calls)

            if node.calls:
                soa_field_usage = SoaFieldUsageInfo.merge([
                    ir.kernels[kcall.kernel_name].metadata.get(MDK_SOA_FIELD_USAGE_INFO)
                    for kcall in node.calls
                ])
            else:
                soa_field_usage = UnfinishedSoaFieldUsageInfo(ir).finalized()

            pgroup = ParallelGroup(
                swaps=node.swaps, 
                calls=node.calls, 
                sync_r2w=[], 
                sync_w2r=[], 
                original=node,
                soa_field_usage=soa_field_usage,
            )

            pgroups.append(pgroup)

        # Remove redundant swaps and merge non-conflicting groups

        swap_history: defaultdict[str, list[ParallelGroup]] = defaultdict(list)
        clean_swaps: set[str] = set() # Track which SoA fields haven't been accessed since being swapped
        dirty: set[str] = set()

        for i, pgroup in enumerate(pgroups):
            pgroup = pgroups[i]

            assert pgroup.soa_field_usage is not None
            
            buffer_usage: SoaFieldUsageInfo = pgroup.soa_field_usage

            if clean_swaps.intersection(pgroup.swaps):
                swaps_removed: list[str] = []
                for swap in pgroup.swaps:
                    if swap in clean_swaps:
                        swaps_removed.append(swap)

                for swap in swaps_removed:
                    pgroups[pgroups.index(swap_history[swap].pop())].swaps.remove(swap)
                    pgroup.swaps.remove(swap)
                    clean_swaps.discard(swap)

            for fld in soa_layout.fields:
                if buffer_usage.check_writes(fld):
                    dirty.add(fld)
            
            for swap in pgroup.swaps:
                swap_history[swap].append(pgroup)
                clean_swaps.add(swap)
                dirty.discard(swap)

            clean_swaps.difference_update([fld for fld in soa_layout.fields if buffer_usage.check_accesses(fld)])
            clean_swaps.update(pgroup.swaps)

            if i > 0:
                swaps_moved: list[str] = []
                for swap in pgroups[i-1].swaps:
                    if not buffer_usage.check_accesses(swap):
                        swaps_moved.append(swap)

                for swap in swaps_moved:
                    swap_history[swap][-1] = pgroup
                    pgroups[i-1].swaps.remove(swap)
                    pgroup.swaps.append(swap)
                        
        for fld in swap_history:
            if fld in clean_swaps:
                pgroups[pgroups.index(swap_history[fld].pop())].swaps.remove(fld)

        if dirty:
            pgroups.append(ParallelGroup(
                swaps=list(dirty), calls=[],
                sync_r2w = [],
                sync_w2r = []
            ))
            for fld in dirty:
                swap_history[fld].append(pgroups[-1])

        # Insert syncs for user swaps

        handled_syncs: set[str] = set()

        new_pgroups: list[ParallelGroup] = pgroups.copy()

        for fld in soa_layout.fields:

            predicate: Callable[[ParallelGroup],bool] = lambda g: (
                (
                    g.soa_field_usage is not None
                    and fld in g.soa_field_usage.index_lut
                    and g.soa_field_usage.check_accesses(fld)
                ) or (
                    fld in g.swaps
                    or fld in g.sync_r2w
                    or fld in g.sync_w2r
                )
            )
            first_use_group = next(filter(predicate, new_pgroups), None)
            if first_use_group:
                first_use_idx = new_pgroups.index(first_use_group)
                head_slice = new_pgroups[:first_use_idx]

                if head_slice:
                    head_slice[0].sync_r2w.append(fld)
                else:
                    new_pgroups.insert(0,ParallelGroup(
                        swaps=[], calls=[],
                        sync_r2w = [fld],
                        sync_w2r = []
                    ))

        for fld in swap_history.copy():
            for swap_number, swap_next in enumerate(swap_history[fld]):
                swap_idx_prev = 0
                if swap_number != 0:
                    swap_idx_prev = new_pgroups.index(swap_history[fld][swap_number-1])
                    
                swap_idx_next = new_pgroups.index(swap_next)

                handled_syncs.clear()

                for i in range(swap_idx_next-1,swap_idx_prev,-1):
                    pgroup = new_pgroups[i]

                    buffer_usage = pgroup.soa_field_usage # type: ignore

                    if buffer_usage is not None and buffer_usage.check_accesses(fld):

                        if buffer_usage.check_read_local_only(fld):
                            # If buffer is only read at kernel position a sync can safely happen after any update logic
                            new_pgroups[i].sync_r2w.append(fld)
                        else:
                            if i == swap_idx_next - 1: break
                            new_pgroups[i+1].sync_r2w.append(fld)

                        handled_syncs.add(fld)
                        break
                    
                    if i == swap_idx_prev+1:
                        pgroup.sync_r2w.append(fld)
                        handled_syncs.add(fld)

                if fld not in handled_syncs:
                    new_pgroups.insert(swap_idx_next, ParallelGroup(
                        swaps=[], calls=[],
                        sync_r2w = [fld],
                        sync_w2r = []
                    ))

        # Update the AST
                
        new_body: list[casys_ast.Cs_ParallelGroup] = []

        for pg in new_pgroups:
            if pg.calls or pg.swaps or pg.sync_r2w or pg.sync_w2r:
                if pg.original:
                    pg.original.calls = pg.calls
                    pg.original.swaps = pg.swaps
                    pg.original.sync_r2w = pg.sync_r2w
                    pg.original.sync_w2r = pg.sync_w2r

                    new_body.append(pg.original)
                else:
                    new_body.append(casys_ast.Cs_ParallelGroup(
                        pg.swaps,
                        pg.calls,
                        pg.sync_r2w,
                        pg.sync_w2r,
                    ))

        ir.step_func.ir_ast.body = new_body # type: ignore

        trkr.add_snapshot(
            tags=(TAG_STEP_FUNC, f_tag_transpiler_module(self)),
            ast_node=ir.step_func.ir_ast
        )
        
        trkr.exit_phase()

            