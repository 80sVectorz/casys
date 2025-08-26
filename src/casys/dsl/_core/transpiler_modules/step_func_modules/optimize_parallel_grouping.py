from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from casys.dsl._core.core_transpiler import TranspilerModule
from casys.dsl._core.errors import TranspileError
from casys.dsl._core.ir import Ir_CaSys
from casys.dsl._core.debug.ast_timeline_tracking import TAG_STEP_FUNC, get_tracker, f_tag_transpiler_module
from casys.dsl._core.ir_metadata_specs.md_kernels_base import BufferUsageInfo
from casys.dsl._core.ir_metadata_specs.md_stepfunc_base import MDK_KCALL_BUFFER_USAGE_INFO

from collections import Counter, defaultdict

import ast
from casys.dsl._core import casys_ast
from casys._ast_pattern_utils.ast_pattern_engine import PatternFinder, PatternTransformer, SingleOccurrenceFinder, Collect, Bind, NodePattern, Filter, OneOrMore

@dataclass(slots=True)
class ParallelGroup(casys_ast.Cs_ParallelGroup):
    buffer_usage: BufferUsageInfo | None = None
    original: casys_ast.Cs_ParallelGroup | None = None

class OptimizeParallelGrouping(TranspilerModule):
    def process(self, ir: Ir_CaSys) -> None:
        trkr = get_tracker()
        trkr.enter_phase('Optimize parallel grouping and double buffer swaps')

        buffers = ir.step_func.base.buffers
        kcall_buffer_usage_map = ir.step_func.metadata.get(MDK_KCALL_BUFFER_USAGE_INFO)

        pgroups: list[ParallelGroup] = []

        for node in ir.step_func.ir_ast.body:
            if not isinstance(node, casys_ast.Cs_ParallelGroup): continue
            assert not (node.swaps and node.calls)

            pgroup = ParallelGroup(
                swaps=node.swaps, 
                calls=node.calls, 
                sync_r2w=[], 
                sync_w2r=[], 
                original=node,
                buffer_usage=BufferUsageInfo.merge([
                    kcall_buffer_usage_map[kcall]
                    for kcall in node.calls
                ])
            )

            pgroups.append(pgroup)

        # Remove redundant swaps and merge non-conflicting groups
        
        swap_history: defaultdict[str, list[ParallelGroup]] = defaultdict(list)
        clean_swaps: set[str] = set() # Track which buffers haven't been accessed since being swapped

        for i, pgroup in enumerate(pgroups):
            pgroup = pgroups[i]
            if pgroup.buffer_usage is None: continue

            buffer_usage: BufferUsageInfo = pgroup.buffer_usage

            buffer_accesses = [buffer for buffer in buffer_usage.buffers if buffer_usage.check_accesses(buffer)]

            if clean_swaps.intersection(pgroup.swaps):
                swaps_removed: list[str] = []
                for swap in pgroup.swaps:
                    if swap in clean_swaps:
                        swaps_removed.append(swap)

                for swap in swaps_removed:
                    pgroups[pgroups.index(swap_history[swap].pop())].swaps.remove(swap)
                    pgroup.swaps.remove(swap)
                    clean_swaps.discard(swap)
            
            for swap in pgroup.swaps:
                swap_history[swap].append(pgroup)
                clean_swaps.add(swap)

            clean_swaps.difference_update(buffer_accesses)
            clean_swaps.update(pgroup.swaps)

            if i > 0:
                swaps_moved: list[str] = []
                for swap in pgroups[i-1].swaps:
                    if swap not in buffer_accesses:
                        swaps_moved.append(swap)

                for swap in swaps_moved:
                    swap_history[swap][-1] = pgroup
                    pgroups[i-1].swaps.remove(swap)
                    pgroup.swaps.append(swap)
                        
        for buffer in swap_history:
            if buffer in clean_swaps:
                pgroups[pgroups.index(swap_history[buffer].pop())].swaps.remove(buffer)

        # Insert syncs for user swaps

        handled_syncs: set[tuple[str,str]] = set()
        buffers = ir.step_func.base.buffers
        all_pairs: dict[str, set[tuple[str,str]]] = {
            buffer.name: set(buffer.get_soa_pairs_multi([buffer]))
            for buffer in buffers.values()
        }

        new_pgroups: list[ParallelGroup] = pgroups.copy()
        new_pgroups.insert(0,ParallelGroup(
            swaps=[], calls=[],
            sync_r2w = [],
            sync_w2r = []
        ))

        for buffer in buffers:
            for pair in all_pairs[buffer]:
                predicate: Callable[[ParallelGroup],bool] = lambda g: (
                    ( g.buffer_usage is not None
                    and pair in g.buffer_usage.index_lut
                    and g.buffer_usage.check_accesses(*pair)
                    ) or (
                        pair[0] in g.swaps
                        or pair in g.sync_r2w
                        or pair in g.sync_w2r
                    )
                )
                first_use_group = next(filter(predicate, new_pgroups), None)
                if first_use_group:
                    first_use_idx = new_pgroups.index(first_use_group)
                    last_use_idx = new_pgroups.index(next(filter(predicate, new_pgroups[::-1])))
                    head_slice = new_pgroups[:first_use_idx]
                    tail_slice = new_pgroups[last_use_idx+1:]

                    if tail_slice:
                        tail_slice[0].sync_r2w.append(pair)
                    elif head_slice:
                        head_slice[-1].sync_w2r.append(pair)

            if buffer in swap_history:
                for swap_number, swap_next in enumerate(swap_history[buffer]):
                    swap_idx_prev = 0
                    if swap_number != 0:
                        swap_idx_prev = new_pgroups.index(swap_history[buffer][swap_number-1])
                        
                    swap_idx_next = new_pgroups.index(swap_next)

                    handled_syncs.clear()
                    for pair in all_pairs[buffer]:
                        for i in range(swap_idx_next-2,swap_idx_prev-1,-1):
                            pgroup = new_pgroups[i]

                            buffer_usage = pgroup.buffer_usage # type: ignore

                            if buffer_usage is not None and buffer_usage.check_accesses(*pair):
                                if i == len(new_pgroups) - 1: break

                                if buffer_usage.check_read_local_only(*pair):
                                    # If buffer is only read at kernel position a sync can safely happen after any update logic
                                    new_pgroups[i].sync_r2w.append(pair)
                                else:
                                    new_pgroups[i+1].sync_r2w.append(pair)
                                handled_syncs.add(pair)
                                break

                            if (i == swap_idx_next-1 and buffer_usage is None) or i == swap_idx_prev-1:
                                new_pgroups[i].sync_r2w.append(pair)
                                handled_syncs.add(pair)

                    new_pgroups.insert(swap_idx_next, ParallelGroup(
                        swaps=[], calls=[],
                        sync_r2w = list(all_pairs[buffer].difference(handled_syncs)),
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

            