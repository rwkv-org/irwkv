# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for model-agnostic IPC weight cache behavior."""

import pytest


def _parallel(**kw):
    from types import SimpleNamespace

    base = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        nnodes=1,
        node_rank=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_daemon_places_tp_and_dp_ranks_on_local_gpus():
    """Each node serves a contiguous global-rank block (DP-major), so without
    DP local GPU i is TP rank r*local+i, and single-node DP launchers offset
    by --data-parallel-start-rank."""
    from irwkv.model_executor.model_loader.weight_cache.daemon import plan_local_ranks

    tp_second_node = _parallel(tensor_parallel_size=8, nnodes=2, node_rank=1)
    assert plan_local_ranks(tp_second_node) == [(i, 0, 0, 4 + i) for i in range(4)]

    dp_third_node = _parallel(
        data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=8
    )
    assert plan_local_ranks(dp_third_node) == [(i, 8 + i, 0, 0) for i in range(4)]

    dp_tp = _parallel(
        tensor_parallel_size=2,
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=2,
    )
    assert plan_local_ranks(dp_tp) == [
        (0, 2, 0, 0),
        (1, 2, 0, 1),
        (2, 3, 0, 0),
        (3, 3, 0, 1),
    ]

    # DP + nnodes: node-local TP replicas
    dp_tp_node0 = _parallel(tensor_parallel_size=8, data_parallel_size=2, nnodes=2)
    assert plan_local_ranks(dp_tp_node0) == [(i, 0, 0, i) for i in range(8)]
    dp_tp_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=2, node_rank=1
    )
    assert plan_local_ranks(dp_tp_node1) == [(i, 1, 0, i) for i in range(8)]

    # DP + nnodes: TP group spans nodes (TP8 x DP2 on 4 nodes)
    tp_span_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=1
    )
    assert plan_local_ranks(tp_span_node1) == [(i, 0, 0, 4 + i) for i in range(4)]
    tp_span_node2 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=2
    )
    assert plan_local_ranks(tp_span_node2) == [(i, 1, 0, i) for i in range(4)]

    # DP + nnodes: several DP replicas per node (TP4 x DP4 on 2 nodes)
    dp_multi_node0 = _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    assert plan_local_ranks(dp_multi_node0) == [(i, i // 4, 0, i % 4) for i in range(8)]

    # PP adds one daemon placement per stage while preserving TP rank order.
    pp_tp = _parallel(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert plan_local_ranks(pp_tp) == [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (2, 0, 1, 0),
        (3, 0, 1, 1),
    ]


def test_daemon_rejects_unmappable_parallelism():
    from irwkv.model_executor.model_loader.weight_cache.daemon import (
        _reject_unsupported_parallelism,
    )

    _reject_unsupported_parallelism(
        _parallel(
            data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=12
        )
    )
    # DP combines with --nnodes when the world size divides evenly.
    _reject_unsupported_parallelism(
        _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    )
    _reject_unsupported_parallelism(_parallel(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(_parallel(tensor_parallel_size=3, nnodes=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(
            _parallel(tensor_parallel_size=2, data_parallel_size=3, nnodes=4)
        )
    with pytest.raises(ValueError, match="exceeds"):
        _reject_unsupported_parallelism(
            _parallel(
                data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=13
            )
        )


def test_weight_cache_key_distinguishes_dp_ranks():
    from dataclasses import replace

    from irwkv.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    key = WeightCacheKey(
        checkpoint="ckpt",
        model_arch="Arch",
        tp_size=1,
        tp_rank=0,
        dtype="bf16",
        quantization=None,
        quant_config_hash="h",
        revision=None,
        irwkv_version="v",
        dp_size=16,
        dp_rank=3,
        pp_size=2,
        pp_rank=1,
    )
    assert key.mismatched_fields(replace(key, dp_rank=4)) == ["dp_rank"]
    assert key.mismatched_fields(replace(key, dp_size=8, dp_rank=3)) == ["dp_size"]
    assert key.mismatched_fields(replace(key, pp_rank=0)) == ["pp_rank"]


def test_ipc_loader_copy_mode_reports_no_external_weight_memory():
    """Copy mode clones the weights into the engine, so nothing is external
    (irwkv_config is never touched)."""
    from irwkv.config import LoadConfig
    from irwkv.model_executor.model_loader.weight_cache.ipc_loader import (
        IpcModelLoader,
    )

    copy_mode = LoadConfig(
        load_format="ipc_cache", model_loader_extra_config={"mode": "copy"}
    )
    loader = IpcModelLoader(copy_mode)
    assert loader.get_external_weight_memory(None) == 0
