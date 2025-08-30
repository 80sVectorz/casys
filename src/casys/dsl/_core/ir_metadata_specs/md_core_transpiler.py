from __future__ import annotations
from typing import Any, TypedDict

from casys.dsl._core.metadata_store import MetadataKey
import numba

class CoreConfig(TypedDict):
    strict_kernels: bool

MDK_DIMS = MetadataKey[tuple[int,...]]('', 'dims', doc='The simulation grid dimensions.')
MDK_DIMS_SIGNED_NB_TYPES = MetadataKey[tuple[numba.types.Type,...]]('', 'dims_signed_nb_types', doc='The smallest signed numba data type that fully spans each axis of the sim dimensions. For example the kernel position values.')

MDK_CORE_CONF = MetadataKey[CoreConfig]('', 'core_conf', doc='The core configs that all pipelines have.')
MDK_CONSTANTS = MetadataKey[dict[str,Any]]('', 'constants', doc='The constants that are used by components of the CA-System.')