from __future__ import annotations
from typing import Any, TypedDict

from casys.dsl._core.metadata_store import MetadataKey

class CoreConfig(TypedDict):
    strict_kernels: bool

MDK_DIMS = MetadataKey[tuple[int,...]]('', 'dims', doc='The simulation grid dimensions')
MDK_CORE_CONF = MetadataKey[CoreConfig]('', 'core_conf', doc='The core configs that all pipelines have')
MDK_CONSTANTS = MetadataKey[dict[str,Any]]('', 'constants', doc='The constants that are used by components of the CA-System')