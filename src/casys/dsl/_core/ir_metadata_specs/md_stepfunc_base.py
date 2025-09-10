from __future__ import annotations

from dataclasses import dataclass

from casys.dsl._core.descriptors import KernelCallDescriptor
from casys.dsl._core.ir_metadata_specs.md_kernels_base import SoaFieldUsageInfo

from casys.dsl._core.metadata_store import MetadataKey

import numba

MDK_SIGNATURE = MetadataKey[dict[str, numba.types.Type]]('', 'signature', factory=dict, doc='The final Numba typed step function signature')
MDK_SIGNATURE_BUFFERS = MetadataKey[list[str]]('', 'signature_buffers', factory=list, doc='The buffers in the final Numba typed step function signature')

MDK_SWAP_TARGETS = MetadataKey[set[str]]('', 'swap_targets', factory=set, doc='A set containing the buffers that are swapped by step_func_swap')
MDK_NEEDS_DEDICATED_IDX = MetadataKey[list[str]]('', 'needs_dedicated_idx', factory=list, doc='A list of the buffers that require dedicated double buffer index ids.')
    
MDK_KCALL_BUFFER_USAGE_INFO = MetadataKey[dict[KernelCallDescriptor, SoaFieldUsageInfo]]('', 'kcall_buffer_usage_info', factory=dict, doc='A dict of Cs_KernelCall node IDs and BufferUsageInfo objects.')