# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import irwkv.v1.worker.gpu.model_runner as gpu_model_runner_module
from irwkv.platforms.interface import Platform
from irwkv.utils.gc_utils import freeze_gc_for_cudagraph_capture
from irwkv.v1.attention.backend import MultipleOf
from irwkv.v1.worker.block_table import (
    MultiGroupBlockTable,
    SlotMappingMode,
    get_block_table_width,
)
from irwkv.v1.worker.gpu.model_runner import GPUModelRunner
from irwkv.v1.worker.utils import select_common_block_size


@pytest.mark.parametrize("gc_initially_enabled", [True, False])
def test_freeze_gc_disables_and_restores_automatic_gc(
    monkeypatch: pytest.MonkeyPatch,
    gc_initially_enabled: bool,
):
    monkeypatch.setenv("IRWKV_ENABLE_CUDAGRAPH_GC", "0")
    original_gc_state = gc.isenabled()
    try:
        if gc_initially_enabled:
            gc.enable()
        else:
            gc.disable()

        with (
            pytest.raises(RuntimeError, match="capture failed"),
            freeze_gc_for_cudagraph_capture(),
        ):
            assert not gc.isenabled()
            raise RuntimeError("capture failed")

        assert gc.isenabled() is gc_initially_enabled
    finally:
        if original_gc_state:
            gc.enable()
        else:
            gc.disable()


def _make_mock_backend_for_kernel_block_size(
    supported_sizes: list[int | MultipleOf],
):
    class _MockBackend:
        @staticmethod
        def get_name():
            return "MOCK"

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return supported_sizes

    return _MockBackend()


def test_select_common_block_size_prefers_manager_block_size():
    backend_a = _make_mock_backend_for_kernel_block_size([MultipleOf(32)])
    backend_b = _make_mock_backend_for_kernel_block_size([64, MultipleOf(16)])

    selected_size = select_common_block_size(128, [backend_a, backend_b])
    assert selected_size == 128


def test_select_common_block_size_uses_largest_shared_int():
    backend_a = _make_mock_backend_for_kernel_block_size([128, 64])
    backend_b = _make_mock_backend_for_kernel_block_size([64, 32])

    selected_size = select_common_block_size(256, [backend_a, backend_b])
    assert selected_size == 64


def test_select_common_block_size_without_active_backends_uses_manager_size():
    assert select_common_block_size(256, []) == 256


def test_select_common_block_size_no_valid_option():
    backend_a = _make_mock_backend_for_kernel_block_size([64])
    backend_b = _make_mock_backend_for_kernel_block_size([MultipleOf(16)])

    with pytest.raises(ValueError):
        select_common_block_size(48, [backend_a, backend_b])


def _mock_backend(supported: list, *, exact: bool = False):
    """Backend accepting multiples of ``supported``, or only those exact sizes
    when ``exact`` (as CPU_MLA does)."""
    from irwkv.v1.attention.backend import AttentionBackend

    class _MockBackendCls(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "MOCK_EXACT" if exact else "MOCK"

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return list(supported)

        if exact:

            @classmethod
            def supports_block_size(cls, block_size: int | None) -> bool:
                return block_size is None or block_size in supported

    return _MockBackendCls


@pytest.mark.parametrize(
    "backends,expected",
    [
        # A lone backend keeps the default; MultipleOf(8) makes a
        # minimum-based regression return 8 instead of 16.
        ([[MultipleOf(8)]], 16),
        # Sparse-MLA main backend (32 or 64) beside an indexer supporting only
        # 64: the main backend alone picks 32, which the indexer rejects at
        # select_common_block_size time.
        ([[32, 64], [64]], 64),
        ([[32, 64]], 32),
        # Neither divides the other, so only the LCM satisfies both.
        ([[32], [48]], 96),
    ],
)
def test_preferred_block_size_satisfies_every_backend(backends, expected):
    classes = [_mock_backend(s) for s in backends]
    assert Platform._preferred_block_size_for_backends(classes, 16, None) == expected


def test_preferred_block_size_searches_past_an_exact_size_backend():
    # Extending greedily picks lcm(16, 32) = 32, which the exact backend
    # rejects; 96 is accepted by both.
    classes = [_mock_backend([16, 96], exact=True), _mock_backend([MultipleOf(32)])]
    assert Platform._preferred_block_size_for_backends(classes, 16, None) == 96


def test_preferred_block_size_rejects_backends_with_no_common_size():
    classes = [_mock_backend([16], exact=True), _mock_backend([MultipleOf(64)])]
    with pytest.raises(ValueError, match="share no supported KV cache block size"):
        Platform._preferred_block_size_for_backends(classes, 16, None)


@pytest.mark.parametrize("cpu", [False, True])
def test_alignment_rejected_by_a_sibling_backend_raises(monkeypatch, cpu):
    # Alignment that lands on a size an exact-size sibling rejects must fail
    # here, not later in select_common_block_size().
    backends = [_mock_backend([MultipleOf(16)]), _mock_backend([16], exact=True)]
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", classmethod(lambda cls, c: backends)
    )

    def bump(cls, irwkv_config, backend_cls):
        irwkv_config.cache_config.block_size = 32

    monkeypatch.setattr(Platform, "_align_hybrid_block_size", classmethod(bump))
    cache_config = SimpleNamespace(
        block_size=16, user_specified_block_size=True, kv_cache_dtype_skip_layers=None
    )
    irwkv_config = SimpleNamespace(
        cache_config=cache_config, model_config=SimpleNamespace(is_hybrid=True)
    )
    from irwkv.platforms.cpu import CpuPlatform  # overrides the method; must check too

    platform = CpuPlatform if cpu else Platform
    with pytest.raises(ValueError, match="MOCK_EXACT"):
        platform.update_block_size_for_backend(irwkv_config)


def test_reload_weights_path_replaces_object_storage_source(monkeypatch):
    # An engine started from an object-storage URI keeps that URI in
    # model_config.model_weights, which the runai_streamer loader prefers over
    # model_config.model. A new weights_path must replace it, or the loader
    # silently re-streams the original checkpoint.
    loader = Mock()
    loader.get_all_weights.return_value = iter(())
    monkeypatch.setattr(gpu_model_runner_module, "get_model_loader", lambda _: loader)
    monkeypatch.setattr(gpu_model_runner_module, "initialize_layerwise_reload", Mock())
    monkeypatch.setattr(gpu_model_runner_module, "finalize_layerwise_reload", Mock())

    runner = Mock()
    runner.model_config = SimpleNamespace(
        model="/tmp/pulled-config-files",
        model_weights="s3://bucket/original",
        revision="abc123",
        quantization=None,
    )
    runner.get_model.return_value.named_parameters.return_value = []
    runner.get_model.return_value.load_weights.return_value = None

    GPUModelRunner.reload_weights(runner, weights_path="org/new-model")

    loader.get_all_weights.assert_called_once_with(
        runner.model_config, runner.get_model.return_value
    )
    cfg = runner.model_config
    assert (cfg.model, cfg.model_weights, cfg.revision) == ("org/new-model", "", None)


def test_v2_runner_snapshots_late_interleave_adjustment(monkeypatch):
    from irwkv.v1.worker.gpu import model_runner as v2_model_runner_module

    runner = object.__new__(v2_model_runner_module.GPUModelRunner)
    runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=16)
    runner.cp_interleave = 1

    class StopInitialization(Exception):
        pass

    monkeypatch.setattr(
        v2_model_runner_module,
        "deepcopy",
        Mock(side_effect=StopInitialization),
    )

    with pytest.raises(StopInitialization):
        runner.initialize_kv_cache(SimpleNamespace())

    assert runner.cp_interleave == 16


def test_get_block_table_width_aligns_to_128_tokens():
    assert get_block_table_width(1875, 64) == 1876


def test_mamba_state_table_width_is_not_aligned():
    block_tables = MultiGroupBlockTable(
        max_num_reqs=1,
        max_num_batched_tokens=1,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[39664],
        max_num_blocks=[1],
        slot_mapping_modes=[SlotMappingMode.NONE],
    )

    assert block_tables[0].max_num_blocks_per_req == 1
