# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from irwkv.config import CacheConfig
from irwkv.utils.mem_constants import GiB_bytes
from irwkv.utils.mem_utils import MemorySnapshot
from irwkv.v1.core.kv_cache_utils import KVCacheBlockCopy
from irwkv.v1.worker.utils import (
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
    request_memory,
)


def test_copy_cpu_kv_cache_logical_blocks_ignores_storage_padding():
    backing = torch.full((10, 2, 3), -1, dtype=torch.float32)
    cache = backing[1:9]
    cache[2:4] = 7
    cache[6:8] = 11

    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks=4,
        kv_cache_block_copies=[
            KVCacheBlockCopy(1, 0),
            KVCacheBlockCopy(3, 2),
        ],
    )

    torch.testing.assert_close(cache[0:2], torch.full_like(cache[0:2], 7))
    torch.testing.assert_close(cache[4:6], torch.full_like(cache[4:6], 11))
    assert (backing[0] == -1).all()
    assert (backing[9] == -1).all()


def test_bind_kv_cache_keeps_scheduler_block_views_for_runner():
    """Block copies index scheduler blocks, so the runner keeps the unmapped
    views while a packed layer is bound with kernel blocks."""

    class _Layer:
        def bind_kv_cache(self, kv_cache):
            self.kv_cache = kv_cache

    num_blocks, row = 4, 64
    storage = torch.zeros(num_blocks * row)
    # Two layers packed into one block row: an 8x4 page, then a 4x2 page
    # split into two 2-row kernel blocks.
    mla = storage.as_strided((num_blocks, 8, 4), (row, 4, 1))
    indexer = storage.as_strided((num_blocks, 1, 4, 2), (row, 8, 2, 1), 40)
    # The indexer in 2-row kernel blocks: block b's kernel block j is b * 16 + j.
    mapped = indexer.as_strided((50, 1, 2, 2), (4, 8, 2, 1))
    kv_caches = {"layers.0.attn": mla, "layers.1.indexer": indexer}
    ctx = {name: _Layer() for name in kv_caches}
    runner_kv_caches: list[torch.Tensor] = []

    bind_kv_cache(
        kv_caches,
        ctx,
        runner_kv_caches,
        layer_kv_caches={**kv_caches, "layers.1.indexer": mapped},
    )

    assert ctx["layers.1.indexer"].kv_cache is mapped
    assert runner_kv_caches == [mla, indexer]
    copies = [KVCacheBlockCopy(1, 3)]
    with pytest.raises(AssertionError, match="not divisible"):
        copy_kv_cache_blocks_inplace([mapped], num_blocks, copies)
    storage[row : 2 * row] = 7
    copy_kv_cache_blocks_inplace(runner_kv_caches, num_blocks, copies)
    assert (mla[3] == 7).all() and (indexer[3] == 7).all()


def test_bind_kv_cache():
    class Layer:
        def bind_kv_cache(self, kv_cache):
            self.kv_cache = kv_cache

    ctx = {
        "layers.0.self_attn": Layer(),
        "layers.1.self_attn": Layer(),
        "layers.2.self_attn": Layer(),
        "layers.3.self_attn": Layer(),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def _memory_snapshot(total_gib: int, free_gib: int) -> MemorySnapshot:
    return MemorySnapshot(
        free_memory=free_gib * GiB_bytes,
        total_memory=total_gib * GiB_bytes,
        device="cpu",
        auto_measure=False,
    )


def test_request_memory_charges_external_weights():
    """Externally held weights are charged against the utilization budget;
    the engine is granted only the remainder."""
    cache_config = CacheConfig(gpu_memory_utilization=0.9)
    # 70 GiB of a 100 GiB device is held externally before the worker starts.
    snapshot = _memory_snapshot(total_gib=100, free_gib=30)

    with pytest.raises(ValueError, match="less than desired"):
        request_memory(snapshot, cache_config)

    # 90 GiB budget - 70 GiB external = 20 GiB for the engine.
    assert request_memory(snapshot, cache_config, 70 * GiB_bytes) == 20 * GiB_bytes

    # The external weights alone exceed the utilization budget.
    with pytest.raises(ValueError, match="exceed the desired"):
        request_memory(snapshot, cache_config, 95 * GiB_bytes)

    # Other tenants squeeze free memory below the engine's remainder.
    with pytest.raises(ValueError, match="less than the engine's budget"):
        request_memory(
            _memory_snapshot(total_gib=100, free_gib=10), cache_config, 70 * GiB_bytes
        )
