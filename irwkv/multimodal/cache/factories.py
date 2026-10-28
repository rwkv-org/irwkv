# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing.synchronize import Lock as LockType
from typing import Literal

from irwkv.config import IrwkvConfig

from .base import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalProcessorOnlyCache,
)
from .lru import (
    LruKeyReplicatedReceiverCache,
    LruKeyReplicatedSenderCache,
)
from .shm import (
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
)


def _get_cache_type(
    irwkv_config: IrwkvConfig,
) -> Literal[None, "processor_only", "lru", "shm"]:
    return None

    # Check if the cache is disabled.


def processor_cache_from_config(
    irwkv_config: IrwkvConfig,
) -> BaseMultiModalProcessorCache | None:
    """Return a `BaseMultiModalProcessorCache`, if enabled."""
    cache_type = _get_cache_type(irwkv_config)
    if cache_type is None:
        return None
    elif cache_type == "processor_only":
        return MultiModalProcessorOnlyCache(irwkv_config.model_config)
    elif cache_type == "lru":
        return LruKeyReplicatedSenderCache(irwkv_config.model_config)
    elif cache_type == "shm":
        return ShmObjectStoreSenderCache(irwkv_config)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")


def processor_only_cache_from_config(
    irwkv_config: IrwkvConfig,
) -> MultiModalProcessorOnlyCache | None:
    """Return a `MultiModalProcessorOnlyCache`, if enabled."""
    cache_type = _get_cache_type(irwkv_config)
    if cache_type is None:
        return None

    return MultiModalProcessorOnlyCache(irwkv_config.model_config)


def engine_receiver_cache_from_config(
    irwkv_config: IrwkvConfig,
) -> BaseMultiModalReceiverCache | None:
    """Return a `BaseMultiModalReceiverCache` for the engine process."""
    cache_type = _get_cache_type(irwkv_config)
    if cache_type in (None, "processor_only", "shm"):
        return None
    elif cache_type == "lru":
        return LruKeyReplicatedReceiverCache(irwkv_config.model_config)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")


def worker_receiver_cache_from_config(
    irwkv_config: IrwkvConfig,
    shared_worker_lock: LockType | None,
) -> BaseMultiModalReceiverCache | None:
    """Return a `BaseMultiModalReceiverCache` for the worker process."""
    cache_type = _get_cache_type(irwkv_config)
    if cache_type in (None, "processor_only", "lru"):
        return None
    elif cache_type == "shm":
        if shared_worker_lock is None:
            raise ValueError(
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

        return ShmObjectStoreReceiverCache(irwkv_config, shared_worker_lock)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")
