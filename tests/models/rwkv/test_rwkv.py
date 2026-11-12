# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from irwkv.config.compilation import CompilationMode, CUDAGraphMode
from irwkv.config.model import ModelConfig
from irwkv.engine.arg_utils import EngineArgs
from irwkv.exceptions import IRWKVValidationError
from irwkv.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
    weight_update_active,
)
from irwkv.model_executor.model_loader.reload.direct import record_checkpoint_layout
from irwkv.model_executor.model_loader.reload.lifecycle import abort_reload
from irwkv.model_executor.models.config import RwkvForCausalLMConfig
from irwkv.model_executor.models.rwkv import (
    RwkvFeedForward,
    RwkvForCausalLM,
    RwkvModel,
    RwkvStateLayer,
    _load_flashrwkv2,
)
from irwkv.sampling_params import SamplingParams
from irwkv.utils.hashing import sha256
from irwkv.v1.attention.backend import AttentionCGSupport
from irwkv.v1.attention.backends.rwkv_attn import (
    RwkvAttentionBackend,
    RwkvAttentionMetadata,
    RwkvAttentionMetadataBuilder,
    RwkvStateSpec,
)
from irwkv.v1.core.kv_cache_manager import KVCacheManager
from irwkv.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from irwkv.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from irwkv.v1.request import Request
from irwkv.v1.worker.gpu.model_states.rwkv import (
    RwkvModelState,
    RwkvSampler,
    _RwkvSamplingGraph,
)


class _FakeStateHandle:
    def __init__(self, state: torch.Tensor) -> None:
        self.state = state
        self.materialized: list[torch.Tensor] = []

    @property
    def memory_layout(self) -> dict[str, int]:
        per_slot = self.state[0].numel() * self.state.element_size()
        return {
            "bytes_per_slot": per_slot,
            "fixed_workspace_nbytes": 0,
            "total_nbytes": self.state.numel() * self.state.element_size(),
        }

    def reset_slots_(self, indices: torch.Tensor) -> None:
        self.state.index_fill_(0, indices.long(), 0)

    def copy_slots_(
        self,
        source: "_FakeStateHandle",
        source_indices: torch.Tensor,
        destination_indices: torch.Tensor,
    ) -> None:
        source_indices = source_indices.long()
        destination_indices = destination_indices.long()
        self.state.index_copy_(
            0, destination_indices, source.state.index_select(0, source_indices)
        )

    def materialize_slots_(self, indices: torch.Tensor) -> None:
        self.materialized.append(indices.clone())


def _fake_flashrwkv2() -> ModuleType:
    module: Any = ModuleType("flashrwkv2")
    module.__version__ = "0.1.0a19"
    module.get_tuning_cache_dir = lambda: None
    module.prepare_tuning = lambda calls, *, mode: [call() for call in calls]

    module.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor = _FakeStateHandle
    module.prepare_tmix_wkv7_recurrent_fp16_state_from_tensor = _FakeStateHandle
    module.prepare_tmix_wkv7_recurrent_metadata = lambda *args, **kwargs: (args, kwargs)
    for name in (
        "infer_cmix_forward_varlen",
        "infer_embedding_ln0_forward_varlen",
        "infer_head_linear_all_forward_varlen",
        "infer_post_norm_output_forward_varlen",
        "infer_sampling_six_parameter_forward_varlen",
        "infer_tmix_postnorm_tokenshift_forward_varlen",
        "infer_tmix_readout_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
        "infer_tmix_wkv_prepare_forward_varlen",
        "setup_sampling_states",
    ):
        setattr(module, name, lambda *args, **kwargs: None)
    return module


@pytest.fixture(autouse=True)
def _clear_flashrwkv2_cache():
    _load_flashrwkv2.cache_clear()
    yield
    _load_flashrwkv2.cache_clear()


def _state_irwkv_config(state_dtype: str = "float32") -> SimpleNamespace:
    hf_config = SimpleNamespace(
        hidden_size=128,
        head_size=64,
        num_attention_heads=2,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype=state_dtype,
            mamba_block_size=16,
            mamba_cache_mode="align",
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
@pytest.mark.parametrize("padded_page", [False, True])
def test_state_spec_and_caller_backed_slot_lifecycle(
    monkeypatch, state_dtype, padded_page
) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    irwkv_config = _state_irwkv_config(state_dtype)
    state_layer = RwkvStateLayer(irwkv_config, 2, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(irwkv_config)

    assert spec.mamba_type.get_class() is RwkvAttentionBackend
    assert (
        RwkvAttentionMetadataBuilder.get_cudagraph_support(irwkv_config, spec)
        is AttentionCGSupport.ALWAYS
    )
    assert spec.external_bytes_per_page == 0
    assert spec.external_fixed_memory_bytes == 0
    assert spec.num_prefill_checkpoint_blocks == 0

    num_slots = 3
    page_size = spec.page_size_bytes
    allocation_page_size = page_size * 3 // 2 if padded_page else page_size
    raw = torch.zeros((num_slots, 1, 1, allocation_page_size), dtype=torch.int8)
    state_layer.bind_kv_cache(raw[..., :page_size])
    tmix, handle, cmix = state_layer.get_layer_state(0)
    assert handle.state.shape == (num_slots, 2, 64, 64)
    assert handle.state.dtype == getattr(torch, state_dtype)
    assert spec.page_size_bytes == 2 * (
        2 * 128 * torch.empty((), dtype=torch.bfloat16).element_size()
        + 2 * 64 * 64 * handle.state.element_size()
    )
    assert tmix.dtype == cmix.dtype == torch.bfloat16
    assert handle.state.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    tmix[1].fill_(1)
    handle.state[1].fill_(2)
    cmix[1].fill_(4)

    source = torch.tensor([1], dtype=torch.int32)
    destination = torch.tensor([2], dtype=torch.int32)
    state_layer.copy_state_slots(source, destination)
    assert torch.all(tmix[2] == 1)
    assert torch.all(handle.state[2] == 2)
    assert torch.all(cmix[2] == 4)

    state_layer.reset_state_slots(destination)
    assert torch.count_nonzero(tmix[2]) == 0
    assert torch.count_nonzero(handle.state[2]) == 0
    assert torch.count_nonzero(cmix[2]) == 0

    tmix[0].fill_(5)
    tmix[1].fill_(1)
    handle.state[0].fill_(6)
    handle.state[1].fill_(2)
    cmix[0].fill_(8)
    cmix[1].fill_(4)
    state_layer.copy_state_slots(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    )
    assert torch.all(tmix[0] == 1) and torch.all(tmix[1] == 5)
    assert torch.all(handle.state[0] == 2) and torch.all(handle.state[1] == 6)
    assert torch.all(cmix[0] == 4) and torch.all(cmix[1] == 8)
    assert torch.count_nonzero(raw.view(-1)[num_slots * page_size :]) == 0

    state_layer.materialize_state_slots(source)
    handles = []
    for layer_idx in range(2):
        layer_handle = state_layer.get_layer_state(layer_idx)[1]
        assert isinstance(layer_handle, _FakeStateHandle)
        handles.append(layer_handle)
    assert all(
        len(layer_handle.materialized) == 1
        and torch.equal(layer_handle.materialized[0], source)
        for layer_handle in handles
    )


def test_live_metadata_reuses_tensors_per_graph_descriptor() -> None:
    builder = object.__new__(RwkvAttentionMetadataBuilder)
    builder._device = torch.device("cpu")
    builder._live_buffers = {}
    builder._live_query_layouts = {}
    builder._captured_descriptors = set()
    prepared_ticket = object()
    builder._state_layer = SimpleNamespace(
        warmup_live_metadata=lambda metadata: None,
        prepare_metadata_ticket=lambda metadata: prepared_ticket,
    )

    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )
    state_indices = torch.tensor([1, 2], dtype=torch.int32)

    first = builder._build_live(metadata, state_indices, for_capture=True)
    second = builder._build_live(metadata, state_indices, for_capture=True)
    runtime_query_start_loc = torch.tensor([0, 3, 3], dtype=torch.int32)
    runtime_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=runtime_query_start_loc,
        query_start_loc_cpu=runtime_query_start_loc,
    )
    runtime = builder._build_live(
        runtime_metadata,
        state_indices,
        for_capture=False,
    )
    versions = (
        runtime.cu_seqlens._version,
        runtime.num_active_tokens._version,
        runtime.num_active_sequences._version,
    )
    repeated_runtime = builder._build_live(
        runtime_metadata,
        state_indices,
        for_capture=False,
    )
    repeated_versions = (
        repeated_runtime.cu_seqlens._version,
        repeated_runtime.num_active_tokens._version,
        repeated_runtime.num_active_sequences._version,
    )
    changed_runtime = builder._build_live(
        metadata,
        state_indices,
        for_capture=False,
    )
    other_query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    other_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=other_query_start_loc,
        query_start_loc_cpu=other_query_start_loc,
    )
    other = builder._build_live(
        other_metadata,
        torch.tensor([3], dtype=torch.int32),
        for_capture=True,
    )

    assert first.cu_seqlens is second.cu_seqlens
    assert first.state_indices is second.state_indices
    assert first.cu_seqlens is runtime.cu_seqlens
    assert first.state_indices is runtime.state_indices
    assert repeated_runtime.cu_seqlens is runtime.cu_seqlens
    assert versions == repeated_versions
    assert repeated_versions == (
        changed_runtime.cu_seqlens._version - 1,
        changed_runtime.num_active_tokens._version - 1,
        changed_runtime.num_active_sequences._version - 1,
    )
    assert changed_runtime.num_active_sequences.item() == 2
    assert other.cu_seqlens is not first.cu_seqlens
    assert other.state_indices is not first.state_indices
    assert first.max_seqlen_capacity == metadata.max_query_len
    assert first.max_seqlen == metadata.max_query_len
    assert runtime.max_seqlen == runtime_metadata.max_query_len
    assert runtime.validated_metadata is None
    assert runtime.has_padding

    eager_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    eager_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=3,
        max_query_len=3,
        query_start_loc=eager_query_start_loc,
        query_start_loc_cpu=eager_query_start_loc,
    )
    eager = builder._build_live(
        eager_metadata,
        torch.tensor([4], dtype=torch.int32),
        for_capture=False,
    )
    assert eager.validated_metadata is prepared_ticket


def test_live_metadata_prepares_and_retains_each_capture_ticket(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    irwkv_config = _state_irwkv_config()
    state_layer = RwkvStateLayer(irwkv_config, 1, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(irwkv_config)
    raw = torch.zeros((1, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    metadata = RwkvAttentionMetadata(
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        state_indices=torch.tensor([0], dtype=torch.int32),
        batch_size=1,
        max_seqlen=1,
        token_capacity=1,
        sequence_capacity=1,
        max_seqlen_capacity=1,
        num_active_tokens=torch.tensor([0], dtype=torch.int32),
        num_active_sequences=torch.tensor([0], dtype=torch.int32),
        retain_ticket=True,
    )

    state_layer.warmup_live_metadata(metadata)
    warmup_ticket = state_layer._live_graph_tickets[-1]
    capture_ticket = state_layer.prepare_metadata_ticket(metadata)
    next_capture_ticket = state_layer.prepare_metadata_ticket(metadata)

    assert capture_ticket is not warmup_ticket
    assert next_capture_ticket is not capture_ticket
    assert len(state_layer._live_graph_tickets) == 3
    assert state_layer._live_graph_tickets[0] is warmup_ticket
    assert state_layer._live_graph_tickets[1] is capture_ticket
    assert state_layer._live_graph_tickets[2] is next_capture_ticket


@pytest.mark.usefixtures("_single_rank")
def test_feed_forward_loads_value_weight_for_provider() -> None:
    config = SimpleNamespace(hidden_size=2, intermediate_size=8, rwkv_lora=None)
    feed_forward = RwkvFeedForward(config, prefix="model.layers.0.mlp")
    checkpoint_weight = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    param = feed_forward.value.weight
    param.weight_loader(param, checkpoint_weight)

    assert torch.equal(param, checkpoint_weight.T.contiguous())


@pytest.mark.parametrize("fold_embedding_norm", [True, False])
def test_model_folds_embedding_norm_after_loading(monkeypatch, fold_embedding_norm):
    module: Any = _fake_flashrwkv2()
    module.infer_embedding_ln0_forward_varlen = lambda embedding, weight, bias, eps: (
        torch.nn.functional.layer_norm(
            embedding, (embedding.shape[-1],), weight, bias, eps
        ).to(torch.bfloat16)
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    model = object.__new__(RwkvModel)
    torch.nn.Module.__init__(model)
    embedding = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    model.embed_tokens = SimpleNamespace(weight=torch.nn.Parameter(embedding.clone()))
    model.embedding_norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
    model.config = SimpleNamespace(layer_norm_epsilon=1e-5)
    model.layers = torch.nn.ModuleList()
    model.start_layer = 0
    model.end_layer = 0
    model._fold_embedding_norm = fold_embedding_norm
    model._embedding_norm_folded = False
    model._embedding_norm_folded_buffer = torch.tensor(False)
    expected = torch.nn.functional.layer_norm(
        embedding,
        (4,),
        model.embedding_norm.weight,
        model.embedding_norm.bias,
        model.config.layer_norm_epsilon,
    ).to(torch.bfloat16)

    model.process_weights_after_loading()

    assert model._embedding_norm_folded == fold_embedding_norm
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert torch.equal(
        model.embed_tokens.weight, expected if fold_embedding_norm else embedding
    )


@pytest.mark.parametrize(
    ("saved_folded", "enable_prompt_embeds"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_tensorized_embedding_norm_state(
    monkeypatch, saved_folded, enable_prompt_embeds
):
    module: Any = _fake_flashrwkv2()
    module.infer_embedding_ln0_forward_varlen = lambda x, weight, bias, eps: (
        torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps).to(
            torch.bfloat16
        )
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    model = object.__new__(RwkvModel)
    torch.nn.Module.__init__(model)
    embedding = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    model.embedding_norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
    folded = torch.nn.functional.layer_norm(
        embedding,
        (4,),
        model.embedding_norm.weight,
        model.embedding_norm.bias,
        1e-5,
    ).to(torch.bfloat16)
    model.embed_tokens = SimpleNamespace(
        weight=torch.nn.Parameter((folded if saved_folded else embedding).clone())
    )
    model.config = SimpleNamespace(layer_norm_epsilon=1e-5)
    model._fold_embedding_norm = not enable_prompt_embeds
    model._embedding_norm_folded = False
    model._embedding_norm_folded_buffer = torch.tensor(saved_folded)
    outer = object.__new__(RwkvForCausalLM)
    torch.nn.Module.__init__(outer)
    outer.model = model

    if saved_folded and enable_prompt_embeds:
        with pytest.raises(ValueError, match="Reserialize"):
            outer.process_weights_after_tensorizer_loading()
        return
    outer.process_weights_after_tensorizer_loading()
    assert model._embedding_norm_folded == (not enable_prompt_embeds)
    assert torch.equal(
        model.embed_tokens.weight, embedding if enable_prompt_embeds else folded
    )


def test_align_state_advance_uses_execution_owned_token_counts() -> None:
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    resets: list[torch.Tensor] = []
    state_layer = SimpleNamespace(
        prefix="model.layers.0.rwkv_state",
        copy_state_slots=lambda source, destination: copies.append(
            (source.clone(), destination.clone())
        ),
        reset_state_slots=lambda slots: resets.append(slots.clone()),
    )
    model_state = object.__new__(RwkvModelState)
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([0, 0], dtype=np.int32)
    model_state._state_num_computed_tokens = np.array([15, 16], dtype=np.int64)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state.device = torch.device("cpu")

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        num_computed_tokens_np=np.array([31, 32], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
    )
    block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    model_state.preprocess_state(
        input_batch,
        (block_table,),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )

    assert len(copies) == 1
    assert not resets
    assert torch.equal(copies[0][0], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(copies[0][1], torch.tensor([4], dtype=torch.int32))
    assert np.array_equal(
        model_state._state_block_columns, np.array([0, 1], dtype=np.int32)
    )
    assert np.array_equal(model_state._state_num_computed_tokens, np.array([16, 17]))


def test_align_state_advance_resets_new_request_destination() -> None:
    resets: list[torch.Tensor] = []
    state_layer = SimpleNamespace(
        prefix="model.layers.0.rwkv_state",
        copy_state_slots=lambda source, destination: None,
        reset_state_slots=lambda slots: resets.append(slots.clone()),
    )
    model_state = object.__new__(RwkvModelState)
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([-1], dtype=np.int32)
    model_state._state_num_computed_tokens = np.array([0], dtype=np.int64)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state.device = torch.device("cpu")

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        num_scheduled_tokens=np.array([32], dtype=np.int32),
    )
    block_table = torch.tensor([[7, 9]], dtype=torch.int32)
    model_state.preprocess_state(
        input_batch,
        (block_table,),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )

    assert len(resets) == 1
    assert torch.equal(resets[0], torch.tensor([9], dtype=torch.int32))
    assert np.array_equal(model_state._state_block_columns, np.array([1]))
    assert np.array_equal(model_state._state_num_computed_tokens, np.array([32]))


def _block_hashed_request(request_id: str, token_ids: list[int]) -> Request:
    params = SamplingParams(max_tokens=64)
    params.update_from_generation_config({}, eos_token_id=0)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=params,
        block_hasher=get_request_block_hasher(16, sha256),
    )


@pytest.mark.parametrize(
    ("spec_cls", "cached_blocks", "continuation_hit"),
    [(MambaSpec, {0}, 16), (RwkvStateSpec, {0, 2}, 48)],
)
def test_align_continuation_resumes_from_latest_committed_state(
    spec_cls, cached_blocks, continuation_hit
) -> None:
    """A finished RWKV turn keeps only its latest committed state, so the next
    turn resumes there instead of at the previous prompt boundary."""
    init_none_hash(sha256)
    spec = spec_cls(
        block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["state"], spec)],
            prefix_cache_retention_interval=0,
        ),
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    turn = _block_hashed_request("turn", list(range(1, 21)))
    for chunk in (16, 4):
        assert manager.allocate_slots(turn, chunk) is not None
        turn.num_computed_tokens += chunk
        manager.new_step_starts()
    # Decode consumes each sampled token in the following step; the final
    # sampled token is never consumed, so 51 of 52 tokens reach the state.
    for token in range(100, 131):
        turn.append_output_token_ids(token)
        assert manager.allocate_slots(turn, 1) is not None
        turn.num_computed_tokens += 1
        manager.new_step_starts()
    turn.append_output_token_ids(131)
    # Only the prompt boundary goes to a connector; superseded latest
    # checkpoints are not offloaded.
    (state_manager,) = manager.coordinator.single_type_managers
    assert [
        boundary
        for *_, boundary in state_manager.take_pending_boundary_state_offloads()
    ] == [16]
    manager.free(turn)

    pool = manager.block_pool
    assert {
        i
        for i, block_hash in enumerate(turn.block_hashes)
        if pool.get_cached_block(block_hash, kv_cache_group_ids=[0]) is not None
    } == cached_blocks
    continuation = _block_hashed_request(
        "continuation", [*turn.all_token_ids, *range(200, 210)]
    )
    _, num_computed_tokens, _ = manager.get_computed_blocks(continuation)
    assert num_computed_tokens == continuation_hit


def _config() -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        dtype=torch.bfloat16,
        architecture="RwkvForCausalLM",
        supports_mamba_prefix_caching=True,
        max_model_len=1024,
        enforce_eager=False,
        logprobs_mode="raw_logprobs",
        return_sampling_mask=False,
        enable_trace_replay=False,
        quantization=None,
        quantization_config=None,
    )
    cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_block_size=None,
        block_size=16,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        kv_offloading_size=None,
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            async_scheduling=None,
            max_num_batched_tokens=8192,
            max_num_seqs=512,
        ),
        quant_config=None,
        kv_transfer_config=None,
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        compilation_config=SimpleNamespace(
            mode=None,
            cudagraph_mode=None,
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=None,
        ),
    )


def test_config_uses_bf16_shift_fp32_wkv_and_full_graph_defaults() -> None:
    irwkv_config = _config()
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)

    assert irwkv_config.cache_config.mamba_cache_dtype == "bfloat16"
    assert irwkv_config.cache_config.mamba_ssm_cache_dtype == "float32"
    assert not irwkv_config.cache_config.enable_prefix_caching
    assert irwkv_config.cache_config.mamba_cache_mode == "none"
    assert irwkv_config.cache_config.mamba_block_size == 1024
    assert irwkv_config.scheduler_config.async_scheduling is None
    assert irwkv_config.compilation_config.mode == CompilationMode.NONE
    assert irwkv_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert irwkv_config.model_config.hf_config.head_dtype == "float32"
    assert irwkv_config.compilation_config.max_cudagraph_capture_size == 8192
    assert 1024 in irwkv_config.compilation_config.cudagraph_capture_sizes
    assert 5120 in irwkv_config.compilation_config.cudagraph_capture_sizes
    assert 8192 in irwkv_config.compilation_config.cudagraph_capture_sizes


@pytest.mark.parametrize("architecture", ["RwkvForCausalLM", "LlamaForCausalLM"])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_rwkv_prefix_cache_default_preserves_explicit_choice(architecture, explicit):
    args = EngineArgs(model=".", enable_prefix_caching=explicit)
    config = SimpleNamespace(
        architecture=architecture,
        is_chunked_prefill_supported=True,
        is_prefix_caching_supported=True,
        runner_type="generate",
    )
    args._set_default_chunked_prefill_and_prefix_caching_args(config)
    expected = architecture != "RwkvForCausalLM" if explicit is None else explicit
    assert args.enable_prefix_caching is expected


@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
def test_config_accepts_explicit_rwkv_prefix_cache(state_dtype) -> None:
    irwkv_config = _config()
    irwkv_config.cache_config.enable_prefix_caching = True
    irwkv_config.cache_config.mamba_ssm_cache_dtype = state_dtype
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    assert irwkv_config.cache_config.mamba_ssm_cache_dtype == state_dtype
    assert irwkv_config.cache_config.enable_prefix_caching
    assert irwkv_config.cache_config.mamba_cache_mode == "align"
    assert irwkv_config.cache_config.mamba_block_size == 16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", torch.float16, "--dtype bfloat16"),
        ("mamba_cache_dtype", "float16", "shift state requires bfloat16"),
        ("mamba_ssm_cache_dtype", "bfloat16", "WKV state supports float16 or float32"),
    ],
)
def test_config_requires_flashrwkv2_numerical_contract(field, value, message) -> None:
    irwkv_config = _config()
    target = (
        irwkv_config.model_config if field == "dtype" else irwkv_config.cache_config
    )
    setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)


def test_head_preserves_provider_fp32_logits_and_trims_padding(monkeypatch) -> None:
    module: Any = _fake_flashrwkv2()
    logits = torch.tensor([[1.0001, 1.0002, -4.0]], dtype=torch.float32)
    module.infer_head_linear_all_forward_varlen = lambda hidden, weight: logits
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=2),
        lm_head=SimpleNamespace(weight=torch.empty(3, 4, dtype=torch.bfloat16)),
        logits_processor=lambda head, values: values,
    )
    result = RwkvForCausalLM.compute_logits(
        model, torch.zeros(1, 4, dtype=torch.bfloat16)
    )
    assert result.dtype == torch.float32
    assert torch.equal(result, logits[:, :2])
    assert result[0, 1] > result[0, 0]


@pytest.mark.parametrize("capturing", [False, True])
def test_graph_warmup_checks_coverage_without_repeating_state_updates(
    monkeypatch, capturing
) -> None:
    """Coverage executes the existing warmup once and stays outside capture."""
    module: Any = _fake_flashrwkv2()
    modes = []
    updates = []
    output = torch.ones(2, 4)

    def prepare(calls, *, mode):
        modes.append(mode)
        for call in calls:
            call()
        return {"complete": True}

    def forward(*args):
        updates.append(args)
        return output

    module.prepare_tuning = prepare
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setenv("IRWKV_RWKV_TUNING_MODE", "require")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_rwkv_metadata",
        lambda: SimpleNamespace(retain_ticket=True),
    )
    result = RwkvForCausalLM.forward(
        SimpleNamespace(model=forward), torch.tensor([1, 2]), torch.arange(2)
    )
    assert result is output
    assert len(updates) == 1
    assert modes == ([] if capturing else ["require"])


def test_graph_warmup_propagates_missing_profile_diagnostic(monkeypatch) -> None:
    module: Any = _fake_flashrwkv2()

    def prepare(calls, *, mode):
        assert mode == "require"
        raise RuntimeError("missing verified tuning profile: tmix/wkv7")

    module.prepare_tuning = prepare
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setenv("IRWKV_RWKV_TUNING_MODE", "require")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_rwkv_metadata",
        lambda: SimpleNamespace(retain_ticket=True),
    )
    with pytest.raises(
        RuntimeError, match="missing verified tuning profile: tmix/wkv7"
    ):
        RwkvForCausalLM.forward(
            SimpleNamespace(model=lambda *args: None),
            torch.tensor([1]),
            torch.tensor([0]),
        )


@pytest.mark.parametrize("fails", [False, True])
def test_tuning_restores_startup_slots_after_active_preparation(monkeypatch, fails):
    module: Any = _fake_flashrwkv2()
    metadata = RwkvAttentionMetadata(
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        state_indices=torch.zeros(2, dtype=torch.int32),
        batch_size=2,
        max_seqlen=4,
        token_capacity=4,
        sequence_capacity=2,
        max_seqlen_capacity=4,
        num_active_tokens=torch.zeros(1, dtype=torch.int32),
        num_active_sequences=torch.zeros(1, dtype=torch.int32),
        retain_ticket=True,
    )
    pool = torch.zeros(2, dtype=torch.float32)
    modes = []
    context = metadata

    @contextmanager
    def forward_context(active, config):
        nonlocal context
        context = active
        yield
        context = metadata

    def forward(*args):
        if context is not metadata:
            assert context.num_active_tokens.item() == 4
            assert context.num_active_sequences.item() == 2
            assert context.state_indices.tolist() == [0, 1]
            assert context.cu_seqlens.tolist() == [0, 2, 4]
            assert not context.retain_ticket
            pool.index_fill_(0, context.state_indices.long(), 1)
        return torch.ones(4, 2)

    def prepare(calls, *, mode):
        modes.append(mode)
        for call in calls:
            call()
        if fails:
            raise RuntimeError("verification failed")
        return {"complete": True}

    module.prepare_tuning = prepare
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setenv("IRWKV_RWKV_TUNING_MODE", "tune")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_rwkv_metadata", lambda: metadata
    )
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.set_forward_context", forward_context
    )
    state = SimpleNamespace(
        reset_state_slots=lambda slots: pool.index_fill_(0, slots.long(), 0)
    )
    model = SimpleNamespace(
        model=forward,
        get_rwkv_state_layer=lambda: state,
        irwkv_config=None,
        _prepared_tuning_shapes=set(),
    )
    if fails:
        with pytest.raises(RuntimeError, match="verification failed"):
            RwkvForCausalLM.forward(model, torch.arange(4), torch.arange(4))
    else:
        assert RwkvForCausalLM.forward(
            model, torch.arange(4), torch.arange(4)
        ).shape == (4, 2)
    assert modes == (["tune"] if fails else ["tune", "require"])
    assert torch.equal(pool, torch.zeros_like(pool))
    assert metadata.num_active_tokens.item() == 0
    assert metadata.state_indices.tolist() == [0, 0]


@pytest.mark.parametrize("live_metadata", [False, True])
def test_first_eager_shape_tunes_without_changing_request_state(
    monkeypatch, live_metadata
):
    module: Any = _fake_flashrwkv2()
    request_offsets = [0, 4, 11, 11] if live_metadata else [0, 4, 11]
    token_capacity = 12 if live_metadata else 11
    sequence_capacity = len(request_offsets) - 1
    metadata = RwkvAttentionMetadata(
        cu_seqlens=torch.tensor(request_offsets, dtype=torch.int32),
        state_indices=torch.arange(sequence_capacity, dtype=torch.int32),
        batch_size=2,
        max_seqlen=7,
        token_capacity=token_capacity,
        sequence_capacity=sequence_capacity,
        max_seqlen_capacity=7,
        num_active_tokens=(
            torch.tensor([11], dtype=torch.int32) if live_metadata else None
        ),
        num_active_sequences=(
            torch.tensor([2], dtype=torch.int32) if live_metadata else None
        ),
        has_padding=live_metadata,
    )
    pools = [torch.full((1, sequence_capacity, 1), 3.0) for _ in range(2)]
    pools.append(torch.full((1, sequence_capacity, 1, 1, 1), 3.0))
    calls = []
    modes = []
    context = metadata

    @contextmanager
    def forward_context(active, config):
        nonlocal context
        context = active
        yield
        context = metadata

    def forward(input_ids, *args):
        calls.append((input_ids.clone(), pools[0][0, 0, 0].item()))
        tuning = context is not metadata
        assert context.cu_seqlens.tolist() == (
            [0, 4, 8, 12] if tuning and live_metadata else request_offsets
        )
        assert context.max_seqlen == 7
        if live_metadata:
            assert context.num_active_tokens.item() == (12 if tuning else 11)
            assert context.num_active_sequences.item() == (3 if tuning else 2)
        value = 9 if tuning else 5
        for pool in pools:
            pool.fill_(value)
        return input_ids

    def prepare(calls, *, mode):
        modes.append(mode)
        for call in calls:
            call()
        return {"complete": True}

    module.prepare_tuning = prepare
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.delenv("IRWKV_RWKV_TUNING_MODE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.get_rwkv_metadata", lambda: metadata
    )
    monkeypatch.setattr(
        "irwkv.model_executor.models.rwkv.set_forward_context", forward_context
    )
    state = SimpleNamespace(
        _tmix_shift_pool=pools[0],
        _cmix_shift_pool=pools[1],
        _wkv_state_pool=pools[2],
        reset_state_slots=lambda slots: [pool.zero_() for pool in pools],
    )
    model = SimpleNamespace(
        model=forward,
        get_rwkv_state_layer=lambda: state,
        irwkv_config=None,
        _prepared_tuning_shapes=set(),
    )
    positions = torch.arange(token_capacity)
    input_ids = positions + 1
    RwkvForCausalLM.forward(model, input_ids, positions)
    RwkvForCausalLM.forward(model, input_ids, positions)

    assert modes == ["tune", "require"]
    assert [value for _, value in calls] == [0, 3, 5]
    assert torch.equal(calls[0][0], torch.zeros_like(input_ids))
    assert all(torch.equal(ids, input_ids) for ids, _ in calls[1:])
    assert all(torch.equal(pool, torch.full_like(pool, 5)) for pool in pools)


_LORA = {
    "r": 4,
    "lora_alpha": 8,
    "use_nora": True,
    "target_modules": ["r_proj", "o_proj", "mlp.value", "lm_head"],
}


def _tiny_model_irwkv_config(rwkv_lora: dict | None) -> SimpleNamespace:
    irwkv_config = _state_irwkv_config()
    irwkv_config.model_config.enable_prompt_embeds = False
    irwkv_config.model_config.hf_config = SimpleNamespace(
        architecture_version="rwkv7",
        head_size=64,
        hidden_size=128,
        num_attention_heads=2,
        intermediate_size=512,
        num_hidden_layers=2,
        decay_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        gate_low_rank_dim=8,
        group_norm_epsilon=64e-5,
        layer_norm_epsilon=1e-5,
        vocab_size=256,
        tie_word_embeddings=False,
        rwkv_lora=rwkv_lora,
    )
    return irwkv_config


@pytest.fixture
def _single_rank(monkeypatch, default_irwkv_config):
    group = SimpleNamespace(
        rank_in_group=0, world_size=1, is_first_rank=True, is_last_rank=True
    )
    monkeypatch.setattr("irwkv.distributed.parallel_state._TP", group)
    monkeypatch.setattr("irwkv.distributed.parallel_state._PP", group)


def test_config_accepts_rwkv_lora() -> None:
    irwkv_config = _config()
    irwkv_config.model_config.hf_config.rwkv_lora = dict(_LORA)
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)


@pytest.mark.parametrize(
    ("rwkv_lora", "message"),
    [
        ({**_LORA, "lora_dropout": 0.0}, "exactly the keys"),
        ({k: v for k, v in _LORA.items() if k != "use_nora"}, "exactly the keys"),
        ({**_LORA, "r": 0}, "positive integer"),
        ({**_LORA, "r": True}, "positive integer"),
        ({**_LORA, "lora_alpha": "8"}, "finite number"),
        ({**_LORA, "use_nora": 1}, "must be a bool"),
        ({**_LORA, "target_modules": []}, "target_modules"),
        ({**_LORA, "target_modules": ["q_proj"]}, "target_modules"),
        ({**_LORA, "target_modules": "r_proj"}, "target_modules"),
    ],
)
def test_config_rejects_invalid_rwkv_lora(rwkv_lora, message) -> None:
    irwkv_config = _config()
    irwkv_config.model_config.hf_config.rwkv_lora = rwkv_lora
    with pytest.raises(ValueError, match=message):
        RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)


@pytest.mark.usefixtures("_single_rank")
def test_lora_registers_fp32_peft_adapters_loaded_in_place() -> None:
    base = RwkvForCausalLM(irwkv_config=_tiny_model_irwkv_config(None))
    model = RwkvForCausalLM(irwkv_config=_tiny_model_irwkv_config(_LORA))
    adapters = {
        name: param for name, param in model.named_parameters() if ".lora_" in name
    }
    expected = {"lm_head.lora_A.weight": (4, 128), "lm_head.lora_B.weight": (256, 4)}
    for layer in range(2):
        for module, in_features in (
            ("linear_attn.r_proj", 128),
            ("linear_attn.o_proj", 128),
            ("mlp.value", 512),
        ):
            prefix = f"model.layers.{layer}.{module}"
            expected[f"{prefix}.lora_A.weight"] = (4, in_features)
            expected[f"{prefix}.lora_B.weight"] = (128, 4)

    assert {name for name, _ in base.named_parameters()} == {
        name for name, _ in model.named_parameters()
    } - set(adapters)
    assert {name: tuple(param.shape) for name, param in adapters.items()} == expected
    assert all(param.dtype == torch.float32 for param in adapters.values())
    assert all(not param.any() for param in adapters.values())
    assert model.lm_head.lora_scale == 2.0
    assert model.lm_head.use_nora

    loaded = model.load_weights(
        (name, param.T if name.endswith(".mlp.value.weight") else param)
        for name, param in base.named_parameters()
    )
    assert loaded == {name for name, _ in base.named_parameters()}
    assert all(not param.any() for param in adapters.values())

    weights = {name: torch.randn(shape) for name, shape in expected.items()}
    first_name = next(iter(weights))
    assert model.load_weights([(first_name, weights[first_name])]) == {first_name}
    storage = {name: param.data_ptr() for name, param in adapters.items()}
    assert model.load_weights(weights.items()) == set(expected)
    for name, param in adapters.items():
        assert param.data_ptr() == storage[name]
        assert param.dtype == torch.float32
        assert torch.equal(param, weights[name])


@pytest.mark.usefixtures("_single_rank")
def test_lora_snapshots_switch_without_replacing_buffers(tmp_path) -> None:
    from safetensors.torch import save_file

    from irwkv.lora.request import LoRARequest
    from irwkv.lora.worker_manager import WorkerLoRAManager

    model = RwkvForCausalLM(irwkv_config=_tiny_model_irwkv_config(_LORA))
    manager = WorkerLoRAManager(model, max_cpu_loras=1)
    pointers = (
        model.lm_head.lora_A.weight.data_ptr(),
        model.lm_head.lora_B.weight.data_ptr(),
    )
    for index in (1, 2):
        path = tmp_path / str(index)
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 4,
                    "lora_alpha": 8,
                    "target_modules": ["lm_head"],
                    "use_nora": True,
                }
            )
        )
        save_file(
            {
                "base_model.model.lm_head.lora_A.weight": torch.full((4, 128), index),
                "base_model.model.lm_head.lora_B.weight": torch.full((256, 4), index),
            },
            str(path / "adapter_model.safetensors"),
        )
        assert manager.add_adapter(
            LoRARequest("tenant", index, str(path), load_inplace=index == 2)
        )

    manager.set_active_adapter(1)
    assert torch.all(model.lm_head.lora_B.weight == 1)
    manager.set_active_adapter(2)
    assert torch.all(model.lm_head.lora_B.weight == 2)
    assert manager.remove_adapter(1)
    manager.set_active_adapter(None)
    assert not model.lm_head.lora_B.weight.any()
    assert pointers == (
        model.lm_head.lora_A.weight.data_ptr(),
        model.lm_head.lora_B.weight.data_ptr(),
    )


@pytest.mark.usefixtures("_single_rank")
@pytest.mark.parametrize("mode", ["layerwise", "direct"])
@torch.no_grad()
def test_checkpoint_reload_preserves_runtime_weights(monkeypatch, mode) -> None:
    from irwkv.model_executor.model_loader.reload import layerwise

    monkeypatch.setenv("IRWKV_RELOAD_MODE", mode)
    warning = MagicMock()
    monkeypatch.setattr(layerwise.logger, "warning", warning)
    flash: Any = _fake_flashrwkv2()
    flash.infer_embedding_ln0_forward_varlen = lambda x, weight, bias, eps: (
        torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", flash)
    model = RwkvForCausalLM(irwkv_config=_tiny_model_irwkv_config(None))
    record_metadata_for_reloading(model)
    weights = {
        name: torch.randn_like(param.T if name.endswith(".mlp.value.weight") else param)
        for name, param in model.named_parameters()
    }
    model.load_weights(weights.items())
    if mode == "direct":
        record_checkpoint_layout(model)
    model.process_weights_after_loading()
    folded_buffer = model.model._embedding_norm_folded_buffer
    pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
    buffer_pointers = {
        name: buf.data_ptr()
        for name, buf in model.named_buffers()
        if "canonical" in name or "layer_zero" in name
    }
    model_config = SimpleNamespace(
        dtype=torch.bfloat16, word_embeddings_untied_by_checkpoint=False
    )

    for _ in range(2):
        weights = {name: torch.randn_like(weight) for name, weight in weights.items()}
        items = list(weights.items())
        start_reload(model)
        split = len(items) // 2
        model.load_weights(items[:split])
        model.load_weights(items[split:])
        finish_reload(model, model_config, torch.device("cpu"))

        assert model.model._embedding_norm_folded_buffer is folded_buffer
        assert folded_buffer.item()
        assert not any(
            call.args == ("%s: Failed to load weights", "RwkvModel")
            for call in warning.call_args_list
        )
        assert {name: p.data_ptr() for name, p in model.named_parameters()} == pointers
        assert {
            name: buf.data_ptr()
            for name, buf in model.named_buffers()
            if "canonical" in name or "layer_zero" in name
        } == buffer_pointers
        expected_embedding = torch.nn.functional.layer_norm(
            weights["model.embed_tokens.weight"],
            (model.config.hidden_size,),
            weights["model.embedding_norm.weight"],
            weights["model.embedding_norm.bias"],
            model.config.layer_norm_epsilon,
        )
        torch.testing.assert_close(model.model.embed_tokens.weight, expected_embedding)
        for index, layer in enumerate(model.model.layers):
            value = weights[f"model.layers.{index}.mlp.value.weight"]
            torch.testing.assert_close(layer.mlp.value.weight, value.T)
            names = ("w1", "a1", "g1", "w2", "a2", "g2")
            if index:
                names += ("v1", "v2")
            for name in names:
                torch.testing.assert_close(
                    getattr(layer.linear_attn, f"{name}_canonical"),
                    weights[f"model.layers.{index}.linear_attn.{name}"].T,
                )

    attention = model.model.layers[1].linear_attn
    unchanged = attention.a0.clone()
    weight = torch.randn_like(attention.w1)
    start_reload(model)
    model.load_weights([("model.layers.1.linear_attn.w1", weight)])
    finish_reload(model, model_config, torch.device("cpu"))
    torch.testing.assert_close(attention.a0, unchanged)
    torch.testing.assert_close(attention.w1_canonical, weight.T)

    start_reload(model)
    model.load_weights(
        [("model.embedding_norm.weight", weights["model.embedding_norm.weight"])]
    )
    with pytest.raises(ValueError, match="unfolded embedding weight"):
        finish_reload(model, model_config, torch.device("cpu"))
    assert not weight_update_active()
    start_reload(model)
    model.load_weights(weights.items())
    finish_reload(model, model_config, torch.device("cpu"))

    start_reload(model)
    model.load_weights([("model.layers.1.linear_attn.w1", weight)])
    abort_reload(model)
    assert not weight_update_active()
    start_reload(model)
    model.load_weights(weights.items())
    finish_reload(model, model_config, torch.device("cpu"))
    torch.testing.assert_close(attention.w1, weights["model.layers.1.linear_attn.w1"])


def _adapter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: id(value) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
        if "lora" in key or "nora" in key
    }


@pytest.mark.usefixtures("_single_rank")
@pytest.mark.parametrize("rwkv_lora", [None, _LORA])
def test_lora_passes_raw_adapters_to_flashrwkv2(monkeypatch, rwkv_lora) -> None:
    calls: dict[str, dict[str, Any]] = {}
    hidden = torch.zeros(1, 128, dtype=torch.bfloat16)

    def record(name: str, result: object):
        def operator(*args, **kwargs):
            calls[name] = kwargs
            return result

        return operator

    module: Any = _fake_flashrwkv2()
    for name, result in (
        ("infer_tmix_postnorm_tokenshift_forward_varlen", (hidden,) * 7),
        ("infer_tmix_wkv_prepare_forward_varlen", (hidden,) * 8),
        ("infer_tmix_wkv7_recurrent_fp32io16_forward_varlen", hidden),
        ("infer_tmix_readout_forward_varlen", hidden),
        ("infer_cmix_forward_varlen", (hidden, hidden)),
        ("infer_head_linear_all_forward_varlen", torch.zeros(1, 256)),
    ):
        setattr(module, name, record(name, result))
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)

    model = RwkvForCausalLM(irwkv_config=_tiny_model_irwkv_config(rwkv_lora))
    layer = model.model.layers[1]
    metadata = SimpleNamespace(
        validated_metadata=None,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        state_indices=torch.tensor([0], dtype=torch.int32),
        max_seqlen=1,
        batch_size=1,
    )
    layer.linear_attn.process_weights_after_loading()
    layer.linear_attn(
        hidden,
        hidden,
        layer.input_layernorm,
        hidden,
        hidden,
        object(),
        metadata,
        torch.float32,
    )
    layer.mlp(hidden, hidden, layer.post_attention_layernorm, hidden, metadata)
    model.compute_logits(hidden)

    adapter_calls = {name: _adapter_kwargs(kwargs) for name, kwargs in calls.items()}
    assert len(adapter_calls) == 6
    if rwkv_lora is None:
        assert all(not kwargs for kwargs in adapter_calls.values())
        return
    attention, feed_forward = layer.linear_attn, layer.mlp
    assert adapter_calls["infer_tmix_wkv_prepare_forward_varlen"] == {
        "receptance_lora_a": id(attention.r_proj.lora_A.weight),
        "receptance_lora_b": id(attention.r_proj.lora_B.weight),
        "receptance_lora_scale": 2.0,
        "receptance_use_nora": True,
    }
    assert adapter_calls["infer_tmix_readout_forward_varlen"] == {
        "output_lora_a": id(attention.o_proj.lora_A.weight),
        "output_lora_b": id(attention.o_proj.lora_B.weight),
        "output_lora_scale": 2.0,
        "output_use_nora": True,
    }
    assert adapter_calls["infer_cmix_forward_varlen"] == {
        "value_lora_a": id(feed_forward.value.lora_A.weight),
        "value_lora_b": id(feed_forward.value.lora_B.weight),
        "value_lora_scale": 2.0,
        "value_use_nora": True,
    }
    assert adapter_calls["infer_head_linear_all_forward_varlen"] == {
        "lora_a": id(model.lm_head.lora_A.weight),
        "lora_b": id(model.lm_head.lora_B.weight),
        "lora_scale": 2.0,
        "use_nora": True,
    }


def test_config_accepts_pipeline_parallel_size_two() -> None:
    irwkv_config = _config()
    irwkv_config.parallel_config.pipeline_parallel_size = 2
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    assert irwkv_config.parallel_config.pipeline_parallel_size == 2


def test_config_uses_eager_cuda_graph_mode_for_diagnostics() -> None:
    irwkv_config = _config()
    irwkv_config.model_config.enforce_eager = True
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    assert irwkv_config.compilation_config.mode == CompilationMode.NONE
    assert irwkv_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_config_rejects_tensor_parallelism() -> None:
    irwkv_config = _config()
    irwkv_config.parallel_config.tensor_parallel_size = 2
    try:
        RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    except ValueError as error:
        assert "tensor parallelism" in str(error)
    else:
        raise AssertionError("tensor parallelism must be rejected")


def test_config_rejects_non_fp32_head_output() -> None:
    irwkv_config = _config()
    irwkv_config.model_config.hf_config.head_dtype = "model"
    try:
        RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    except ValueError as error:
        assert "float32 lm_head output" in str(error)
    else:
        raise AssertionError("RWKV must produce float32 logits directly")


def test_config_preserves_explicit_cudagraph_sizes() -> None:
    irwkv_config = _config()
    irwkv_config.compilation_config.cudagraph_capture_sizes = [1, 32, 96]

    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)

    assert irwkv_config.compilation_config.max_cudagraph_capture_size is None
    assert irwkv_config.compilation_config.cudagraph_capture_sizes == [1, 32, 96]


def test_config_rejects_unsupported_rapid_sampling_modes() -> None:
    for field, value, message in (
        ("logprobs_mode", "processed_logprobs", "raw logprobs"),
        ("return_sampling_mask", True, "sampling masks"),
        ("enable_trace_replay", True, "trace replay"),
    ):
        irwkv_config = _config()
        setattr(irwkv_config.model_config, field, value)
        try:
            RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{field} must be rejected")


def test_sampling_params_verify_rejects_unsupported_rwkv_controls() -> None:
    for model_type, sampling_params, message in (
        ("rwkv", SamplingParams(repetition_penalty=1.1), "repetition_penalty"),
        ("rwkv", SamplingParams(min_p=0.1), "min_p"),
        ("qwen2", SamplingParams(penalty_decay=0.9), "penalty_decay"),
    ):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            max_logprobs=20,
            get_vocab_size=lambda: 64,
            logits_processors=None,
        )
        try:
            sampling_params.verify(model_config, None, None)
        except IRWKVValidationError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{message} must be rejected")


def test_rapid_sampler_preserves_preempted_request_state() -> None:
    sampler = object.__new__(RwkvSampler)
    sampler.req_states = SimpleNamespace(req_id_to_index={"request": 1})
    sampler._rapid_sampling_states = torch.arange(8, dtype=torch.int8).view(2, 4)
    sampler._rapid_penalties = torch.arange(12, dtype=torch.float32).view(2, 6)
    sampler._preserved_requests = {}
    sampler._new_requests = []

    sampler.preempt_request("request")
    state, penalties = sampler._preserved_requests["request"]
    sampler._rapid_sampling_states[1].zero_()
    sampler._rapid_penalties[1].zero_()

    assert torch.equal(state, torch.tensor([4, 5, 6, 7], dtype=torch.int8))
    assert torch.equal(
        penalties,
        torch.tensor([6, 7, 8, 9, 10, 11], dtype=torch.float32),
    )
    sampler.remove_request("request")
    assert "request" in sampler._preserved_requests

    from irwkv.v1.worker.gpu.model_runner import GPUModelRunner

    sampler.req_states.req_id_to_index.clear()
    sampler.req_states.add_request = lambda **kwargs: (
        sampler.req_states.req_id_to_index.update({kwargs["req_id"]: 0})
    )
    sampler.req_states.apply_staged_writes = lambda: None
    runner = MagicMock(spec=GPUModelRunner)
    runner.req_states = sampler.req_states
    runner.model_state = SimpleNamespace(
        add_request=lambda idx, data: sampler.stage_request(idx, data.req_id),
        apply_staged_writes=lambda: None,
    )
    runner._remove_request.side_effect = lambda req_id: sampler.remove_request(req_id)
    runner.sampler = None
    runner.block_tables = MagicMock()
    runner.is_last_pp_rank = False
    runner.encoder_cache = None
    request = SimpleNamespace(
        req_id="request",
        prefill_token_ids=[1],
        prompt_len=1,
        sampling_params=None,
        num_computed_tokens=0,
        block_ids=[],
        lora_request=None,
    )
    GPUModelRunner.add_requests(runner, SimpleNamespace(scheduled_new_reqs=[request]))
    runner._remove_request.assert_not_called()
    assert "request" not in sampler._preserved_requests
    assert len(sampler._new_requests) == 1
    req_idx, preserved = sampler._new_requests[0]
    assert req_idx == 0
    assert preserved is not None
    assert preserved[0] is state
    assert preserved[1] is penalties

    writes = SimpleNamespace(apply_staged_writes=lambda: None, copy_to_uva=lambda: None)
    for name in (
        "sampling_states",
        "_logit_bias_state",
        "_bad_words_state",
        "logprob_token_ids_state",
        "thinking_budget_state",
        "_presence_penalty",
        "_frequency_penalty",
        "_penalty_decay",
    ):
        setattr(sampler, name, writes)
    sampler.apply_staged_writes()
    assert torch.equal(sampler._rapid_sampling_states[0], state)
    assert torch.equal(sampler._rapid_penalties[0], penalties)
    assert not sampler._rapid_sampling_states[1].any()
    assert not sampler._rapid_penalties[1].any()

    sampler.req_states.req_id_to_index.clear()
    sampler._preserved_requests["request"] = (state, penalties)
    sampler.remove_request("request")
    assert "request" not in sampler._preserved_requests


def _graph_sampler(monkeypatch, calls: list[str]) -> tuple[RwkvSampler, Any]:
    monkeypatch.setattr(
        "irwkv.v1.worker.gpu.model_states.rwkv.get_num_sampled_and_rejected",
        lambda *args: (
            torch.tensor([1, 0, 1, 1], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        ),
    )
    graph = object.__new__(_RwkvSamplingGraph)
    graph.hidden_states = torch.empty(4, 3)
    graph.slot_indices = torch.empty(4, dtype=torch.int32)
    graph.presence_penalty = torch.empty(4)
    graph.frequency_penalty = torch.empty(4)
    graph.penalty_decay = torch.empty(4)
    graph.temperature = torch.empty(4)
    graph.top_k = torch.empty(4, dtype=torch.int32)
    graph.top_p = torch.empty(4)
    graph.num_active = torch.empty(1, dtype=torch.int32)
    graph.logits = torch.zeros(4, 8)
    graph.sampled = torch.tensor([10, 11, 12, -1], dtype=torch.int32)
    graph.logits_graph = SimpleNamespace(replay=lambda: calls.append("logits"))
    graph.sample_graph = SimpleNamespace(replay=lambda: calls.append("sample"))
    sampler = object.__new__(RwkvSampler)
    sampler.compute_nans = False
    sampler.get_logprobs_dims = lambda idx_mapping_np: None
    sampler.needs_logits_processing = np.zeros(4, dtype=bool)
    sampler._sampling_graphs = {4: graph}
    active_rows = np.empty(4, dtype=np.int64)
    active_slots = np.empty(4, dtype=np.int32)
    graph_cu_num_logits = np.full(5, -1, dtype=np.int32)
    sampler._graph_cu_num_logits = SimpleNamespace(
        np=graph_cu_num_logits,
        copy_to_uva=lambda size: torch.from_numpy(graph_cu_num_logits[:size].copy()),
    )
    sampler._active_rows = SimpleNamespace(
        np=active_rows,
        copy_to_uva=lambda size: torch.from_numpy(active_rows[:size]),
    )
    sampler._active_slots = SimpleNamespace(
        np=active_slots,
        copy_to_uva=lambda size: torch.from_numpy(active_slots[:size]),
    )
    sampler._presence_penalty = SimpleNamespace(gpu=torch.arange(4).float())
    sampler._frequency_penalty = SimpleNamespace(gpu=torch.arange(4).float() + 4)
    sampler._penalty_decay = SimpleNamespace(gpu=torch.arange(4).float() + 8)
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(gpu=torch.arange(4).float() + 12),
        top_k=SimpleNamespace(gpu=torch.arange(4, dtype=torch.int32) + 16),
        top_p=SimpleNamespace(gpu=torch.arange(4).float() + 20),
    )
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([2, 8, 4, 1]))
    )
    return sampler, graph


def _graph_input_batch(num_computed_prefill_tokens: list[int]) -> Any:
    return SimpleNamespace(
        req_ids=["a", "b", "c", "d"],
        num_reqs=4,
        idx_mapping_np=np.array([2, 1, 3, 0]),
        idx_mapping=torch.tensor([2, 1, 3, 0]),
        num_computed_prefill_tokens_np=np.array(num_computed_prefill_tokens),
        num_scheduled_tokens=np.array([1, 2, 1, 1]),
        prefill_len_np=np.array([2, 8, 4, 1]),
        seq_lens=torch.tensor([2, 4, 4, 1]),
        cu_num_logits=torch.tensor([0, 1, 2, 3, 4]),
    )


def test_rapid_sampler_graph_compacts_completed_prefills(monkeypatch) -> None:
    calls: list[Any] = []
    sampler, graph = _graph_sampler(monkeypatch, calls)
    input_batch = _graph_input_batch([1, 2, 3, 0])
    hidden_states = torch.arange(12, dtype=torch.bfloat16).view(4, 3)

    output = sampler.sample_hidden_states(hidden_states, input_batch, None)

    assert output is not None
    assert torch.equal(output.sampled_token_ids[:, 0], torch.tensor([10, 0, 11, 12]))
    assert calls == ["logits", "sample"]
    assert graph.num_active.item() == 3
    assert torch.equal(graph.hidden_states[:3], hidden_states[[0, 2, 3]])
    assert torch.equal(graph.slot_indices[:3], torch.tensor([2, 3, 0]))
    assert torch.equal(graph.presence_penalty[:3], torch.tensor([2.0, 3.0, 0.0]))
    assert torch.equal(graph.top_k[:3], torch.tensor([18, 19, 16]))


def _grammar_worker(calls: list[Any]) -> Any:
    def apply_grammar_bitmask(logits, input_batch, req_ids, bitmask) -> None:
        cu_num_logits = input_batch.cu_num_logits.tolist()
        assert cu_num_logits == input_batch.cu_num_logits_np.tolist()
        calls.append(("grammar", logits, cu_num_logits, bitmask))

    return SimpleNamespace(apply_grammar_bitmask=apply_grammar_bitmask)


@pytest.mark.parametrize(
    ("num_computed_prefill_tokens", "cu_num_logits"),
    [
        # Every request samples: one graph row per batch row.
        ([1, 6, 3, 0], [0, 1, 2, 3, 4]),
        # Request "b" is mid-prefill and has no graph row, so "c" and "d"
        # shift up one row.
        ([1, 2, 3, 0], [0, 1, 1, 2, 3]),
    ],
)
def test_rapid_sampler_graph_masks_grammar_between_logits_and_sampling(
    monkeypatch, num_computed_prefill_tokens: list[int], cu_num_logits: list[int]
) -> None:
    """The bitmask is applied between the logits and sampling replays, with
    ``cu_num_logits`` as full cumulative offsets into the compacted graph rows,
    including in mixed prefill batches."""
    calls: list[Any] = []
    sampler, graph = _graph_sampler(monkeypatch, calls)
    input_batch = _graph_input_batch(num_computed_prefill_tokens)
    bitmask = np.zeros((3, 1), dtype=np.int32)
    grammar_output = SimpleNamespace(
        structured_output_request_ids=["a", "c", "d"], grammar_bitmask=bitmask
    )

    output = sampler.sample_hidden_states(
        torch.zeros(4, 3), input_batch, grammar_output, _grammar_worker(calls)
    )

    assert output is not None
    assert calls == [
        "logits",
        ("grammar", graph.logits, cu_num_logits, bitmask),
        "sample",
    ]
    assert torch.equal(input_batch.cu_num_logits, torch.tensor([0, 1, 2, 3, 4]))


@pytest.mark.parametrize("factory", [SamplingParams, SamplingParams.from_optional])
def test_penalty_decay_preserves_positional_sampling_arguments(factory) -> None:
    params = factory(1, 0.0, 0.0, 1.0, 0.5)
    assert params.temperature == 0.5
    assert params.penalty_decay == 1.0
    assert factory(penalty_decay=0.9).penalty_decay == 0.9


@pytest.mark.parametrize("model_type", ["rwkv", "qwen2"])
def test_rwkv_generation_defaults_do_not_leak_to_other_models(model_type) -> None:
    model_config = SimpleNamespace(
        generation_config="irwkv",
        override_generation_config={"stop_strings": ["END"], "penalty_decay": 0.9},
        hf_config=SimpleNamespace(model_type=model_type),
    )
    expected = {"stop": ["END"], "penalty_decay": 0.9} if model_type == "rwkv" else {}
    assert ModelConfig.get_diff_sampling_param(model_config) == expected


def test_rapid_sampler_rejects_loaded_custom_processors() -> None:
    sampler = SimpleNamespace(logits_processors=[object() for _ in range(4)])
    with pytest.raises(ValueError, match="custom logits processors"):
        RwkvSampler(sampler, torch.nn.Identity(), capture_graphs=False)


@pytest.mark.parametrize("mixed_prefill", [False, True])
def test_rapid_sampler_processing_preserves_raw_logits(mixed_prefill) -> None:
    sampler = object.__new__(RwkvSampler)
    slots = np.arange(2)
    for name, dtype in (("_active_rows", np.int64), ("_active_slots", np.int32)):
        buffer = np.empty(2, dtype=dtype)
        setattr(
            sampler,
            name,
            SimpleNamespace(
                np=buffer,
                copy_to_uva=lambda size, buf=buffer: torch.from_numpy(buf[:size]),
            ),
        )
    values = SimpleNamespace(gpu=torch.ones(2))
    sampler._presence_penalty = sampler._frequency_penalty = values
    sampler._penalty_decay = values
    sampler.sampling_states = SimpleNamespace(
        temperature=values, top_k=values, top_p=values
    )
    sampler.needs_logits_processing = np.ones(2, dtype=bool)
    sampler._logit_bias_state = SimpleNamespace(
        apply=lambda logits, ctx: logits.add_(torch.tensor([0.0, 10.0]))
    )
    sampler._bad_words_state = SimpleNamespace(apply=lambda logits, ctx: logits)
    sampler.thinking_budget_state = SimpleNamespace(apply=lambda logits, ctx: None)
    sampler._sample = lambda logits, *args, **kwargs: logits.argmax(dim=-1)
    sampler._rapid_penalties = sampler._rapid_sampling_states = torch.zeros(2, 2)
    sampler._input_batch = SimpleNamespace(
        num_reqs=2,
        num_computed_prefill_tokens_np=np.zeros(2),
        num_scheduled_tokens=np.ones(2),
        prefill_len_np=np.array([1, 2 if mixed_prefill else 1]),
    )
    logits = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    raw = logits.clone()
    sampled, returned = sampler.sample(
        logits,
        torch.from_numpy(slots),
        torch.from_numpy(slots),
        slots,
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        np.ones(2),
        return_logprobs=True,
    )
    assert torch.equal(sampled, torch.tensor([1, 0 if mixed_prefill else 1]))
    assert torch.equal(logits, raw)
    assert torch.equal(returned, raw)


def test_config_accepts_async_scheduling() -> None:
    irwkv_config = _config()
    irwkv_config.scheduler_config.async_scheduling = True
    RwkvForCausalLMConfig.verify_and_update_config(irwkv_config)
    assert irwkv_config.scheduler_config.async_scheduling


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL"),
    reason="IRWKV_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize(
    (
        "use_nora",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enable_prefix_caching",
    ),
    [
        pytest.param(True, 2, 1, True, id="True-pp2-True"),
        pytest.param(False, 1, 2, True, id="False-dp2-True"),
        pytest.param(False, 1, 1, False, id="False-single-False"),
    ],
)
@pytest.mark.asyncio
async def test_real_model_lora_tenant_isolation(
    tmp_path,
    enable_prefix_caching: bool,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    use_nora: bool,
) -> None:
    from shutil import copyfile

    import httpx
    from safetensors.torch import save_file

    from tests.utils import RemoteOpenAIServer

    if (
        max(pipeline_parallel_size, data_parallel_size) > 1
        and torch.accelerator.device_count() < 2
    ):
        pytest.skip("PP2 and DP2 require two GPUs")

    model_path = os.environ["IRWKV_RWKV_TEST_MODEL"]
    with open(os.path.join(model_path, "config.json")) as file:
        config = json.load(file)
    hidden_size, vocab_size = config["hidden_size"], config["vocab_size"]
    generator = torch.Generator().manual_seed(123)
    a_head = torch.randn((4, hidden_size), generator=generator)
    a_receptance = torch.randn((4, hidden_size), generator=generator)
    paths = []
    for direction in (1, -1):
        path = tmp_path / str(direction)
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 4,
                    "lora_alpha": 4,
                    "target_modules": ["r_proj", "lm_head"],
                    "use_nora": use_nora,
                }
            )
        )
        b_head = torch.zeros((vocab_size, 4))
        b_head[42, 0], b_head[43, 0] = 10000 * direction, -10000 * direction
        receptance = "base_model.model.model.layers.0.linear_attn.r_proj"
        save_file(
            {
                "base_model.model.lm_head.lora_A.weight": a_head,
                "base_model.model.lm_head.lora_B.weight": b_head,
                f"{receptance}.lora_A.weight": a_receptance,
                f"{receptance}.lora_B.weight": torch.full(
                    (hidden_size, 4), 0.001 * direction
                ),
            },
            str(path / "adapter_model.safetensors"),
        )
        paths.append(path)

    args = [
        "--served-model-name",
        "base",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "4",
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--api-server-count",
        "1",
        "--max-cpu-loras",
        "2",
        "--gpu-memory-utilization",
        "0.25",
        "--num-gpu-blocks-override",
        "64",
        "--block-size",
        "32",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--enable-prefix-caching"
        if enable_prefix_caching
        else "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--compilation-config",
        '{"cudagraph_capture_sizes":[1,2,4]}',
    ]
    overrides = {
        "rwkv_lora": {
            "r": 4,
            "lora_alpha": 4,
            "use_nora": use_nora,
            "target_modules": ["r_proj", "lm_head"],
        }
    }
    with RemoteOpenAIServer(
        model_path,
        args,
        env_dict={
            "IRWKV_ALLOW_RUNTIME_LORA_UPDATING": "1",
            "IRWKV_RWKV_TUNING_MODE": os.environ.get(
                "IRWKV_RWKV_TUNING_MODE", "inspect"
            ),
        },
        override_hf_configs=overrides,
        max_wait_seconds=300,
    ) as server:
        async with httpx.AsyncClient(base_url=server.url_root, timeout=120) as client:

            async def load(name: str, path, inplace: bool = False):
                response = await client.post(
                    "/v1/load_lora_adapter",
                    json={
                        "lora_name": name,
                        "lora_path": str(path),
                        "load_inplace": inplace,
                    },
                )
                assert response.status_code == 200, response.text

            async def generate(name: str, max_tokens: int = 4):
                response = await client.post(
                    "/v1/completions",
                    json={
                        "model": name,
                        "prompt": list(range(1, 74)),
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "logprobs": 1,
                        "return_token_ids": True,
                        "ignore_eos": True,
                        "session_id": name,
                    },
                )
                assert response.status_code == 200, response.text
                return response.json()["choices"][0]["token_ids"]

            if pipeline_parallel_size == 2:
                response = await client.post(
                    "/v1/load_lora_adapter",
                    json={
                        "lora_name": "tenant-a",
                        "lora_path": str(tmp_path / "absent"),
                    },
                )
                assert response.status_code == 404, response.text

            await load("tenant-a", paths[0])
            await load("tenant-b", paths[1])
            baseline = await generate("base")
            first = await generate("tenant-a")
            second = await generate("tenant-b")
            assert first != second
            mixed = await asyncio.gather(
                *(
                    generate(name)
                    for name in ("tenant-a", "tenant-b", "base", "tenant-a", "tenant-b")
                )
            )
            assert mixed == [first, second, baseline, first, second]
            expected_stream = await generate("tenant-a", max_tokens=256)
            async with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "tenant-a",
                    "prompt": list(range(1, 74)),
                    "max_tokens": 256,
                    "temperature": 0,
                    "return_token_ids": True,
                    "ignore_eos": True,
                    "stream": True,
                    "session_id": "tenant-a",
                },
            ) as response:
                assert response.status_code == 200
                stream_tokens: list[int] = []
                reloaded = False
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    choice = json.loads(line[6:])["choices"][0]
                    stream_tokens.extend(choice["token_ids"] or [])
                    if stream_tokens and not reloaded:
                        assert choice["finish_reason"] is None
                        copyfile(
                            paths[1] / "adapter_model.safetensors",
                            paths[0] / "adapter_model.safetensors",
                        )
                        await load("tenant-a", paths[0], inplace=True)
                        reloaded = True
                assert reloaded
                assert stream_tokens == expected_stream
            assert await generate("tenant-a") == second
            response = await client.post(
                "/v1/unload_lora_adapter", json={"lora_name": "tenant-a"}
            )
            assert response.status_code == 200
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "tenant-a",
                    "prompt": [1, 2],
                    "max_tokens": 1,
                },
            )
            assert response.status_code == 404
            assert await generate("tenant-b") == second
            assert await generate("base") == baseline


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL"),
    reason="IRWKV_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize(
    ("prefix_caching", "prefix_match_unit"),
    [(False, None), (True, None), (True, 32), (True, 8)],
)
@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_real_model_batching_preserves_tokens_and_logprobs(
    irwkv_runner, prefix_caching, prefix_match_unit, state_dtype, temperature
) -> None:
    """Each state mode agrees across prefill, batching and graph execution."""
    model_path = os.environ["IRWKV_RWKV_TEST_MODEL"]
    params = SamplingParams(
        temperature=temperature,
        top_k=1 if temperature == 0 else 20,
        seed=20261026,
        max_tokens=4,
        ignore_eos=True,
        logprobs=3,
        presence_penalty=0,
        frequency_penalty=0,
        penalty_decay=1,
        detokenize=False,
    )
    prompt_token_ids = list(range(1, 74))
    prompt = {"prompt_token_ids": prompt_token_ids}
    reference = None
    for enforce_eager, token_budget in ((True, 128), (False, 32)):
        with irwkv_runner(
            model_path,
            dtype="bfloat16",
            max_model_len=128,
            max_num_batched_tokens=token_budget,
            max_num_seqs=4,
            gpu_memory_utilization=0.8,
            num_gpu_blocks_override=64,
            block_size=32,
            mamba_ssm_cache_dtype=state_dtype,
            enable_chunked_prefill=True,
            enable_prefix_caching=prefix_caching,
            prefix_match_unit=prefix_match_unit,
            enforce_eager=enforce_eager,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
        ) as runner:
            cold = runner.llm.generate([prompt], params)[0]
            hit = runner.llm.generate(
                [{"prompt_token_ids": [7, 8, 9]}, prompt], params
            )[1]
            assert cold.num_cached_tokens == 0
            # The last prompt token is always recomputed to produce logits.
            max_hit = len(prompt_token_ids) - 1
            hit_unit = prefix_match_unit or 32
            expected_hit = max_hit // hit_unit * hit_unit if prefix_caching else 0
            assert hit.num_cached_tokens == expected_hit
            single, mixed = cold.outputs[0], hit.outputs[0]
        assert len(single.token_ids) == 4
        assert mixed.token_ids == single.token_ids
        assert mixed.logprobs == single.logprobs
        if reference is not None:
            assert single.token_ids == reference.token_ids
            assert single.logprobs == reference.logprobs
        reference = single


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL"),
    reason="IRWKV_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize("backend", ["nccl", "ipc"])
def test_real_model_native_weight_reload(irwkv_runner, monkeypatch, backend) -> None:
    """Two changed checkpoints update captured graphs after cache reset."""
    from safetensors import safe_open

    from irwkv.distributed.weight_transfer import (
        ParamMeta,
        WeightSource,
        WeightTransferTrainerFactory,
    )
    from irwkv.distributed.weight_transfer.clients import _json_safe_update_payload
    from irwkv.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
    from irwkv.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
    from irwkv.utils.network_utils import get_open_port

    if backend == "nccl" and torch.accelerator.device_count() < 2:
        pytest.skip("NCCL weight transfer requires two GPUs")
    monkeypatch.setenv("IRWKV_ALLOW_INSECURE_SERIALIZATION", "1")
    model_path = os.environ["IRWKV_RWKV_TEST_MODEL"]
    files = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )

    class CheckpointSource(WeightSource):
        round_id = 0

        def metadata(self):
            result = []
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for name in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(name)
                        result.append(
                            ParamMeta(name, tensor.dtype, tuple(tensor.shape))
                        )
            return result

        def __iter__(self):
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for name in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(name).to("cuda")
                        if name == "lm_head.weight":
                            tensor = tensor * (-1 if self.round_id == 1 else 0.5)
                        elif name in {
                            "model.embedding_norm.weight",
                            "model.layers.1.linear_attn.w1",
                            "model.layers.1.mlp.value.weight",
                        }:
                            tensor = tensor * (0.75 if self.round_id == 1 else 0.5)
                        yield name, tensor

    class Client:
        def __init__(self, llm):
            self.llm = llm

        def init_weight_transfer_engine(self, init_info):
            self.llm.collective_rpc(
                "init_weight_transfer_engine",
                timeout=600,
                kwargs={"init_info": init_info},
            )

        def start_weight_update(self):
            self.llm.collective_rpc("start_weight_update", timeout=600)

        def update_weights(self, update_info):
            self.llm.collective_rpc(
                "update_weights",
                timeout=600,
                kwargs={"update_info": _json_safe_update_payload(update_info)},
            )

        def finish_weight_update(self, weight_version=None):
            self.llm.collective_rpc("finish_weight_update", timeout=600)

    def snapshot(worker):
        model = worker.model_runner.get_model()
        layer = model.model.layers[1]
        manager = worker.model_runner.cudagraph_manager
        sampler = worker.model_runner.model_state._sampler
        importer = getattr(worker.weight_transfer_engine, "_packed_importer", None)
        return {
            "pointers": (
                model.model.embed_tokens.weight.data_ptr(),
                model.lm_head.weight.data_ptr(),
                layer.mlp.value.weight.data_ptr(),
                layer.linear_attn.w1_canonical.data_ptr(),
            ),
            "graph_ids": tuple(id(graph) for graph in manager.graphs.values()),
            "sampler_graph_ids": tuple(
                id(graph) for graph in sampler._sampling_graphs.values()
            ),
            "canonical_equal": torch.equal(
                layer.linear_attn.w1_canonical, layer.linear_attn.w1.T
            ),
            "w1_sum": layer.linear_attn.w1.float().abs().sum().item(),
            "value_sum": layer.mlp.value.weight.float().abs().sum().item(),
            "embedding": model.model.embed_tokens.weight[1, :16].float().tolist(),
            "ipc_released": importer is None or importer._entry is None,
        }

    def set_graph_enabled(worker, enabled):
        worker.model_runner.cudagraph_manager._graphs_captured = enabled

    params = SamplingParams(
        temperature=0,
        top_k=1,
        max_tokens=4,
        ignore_eos=True,
        logprobs=1,
        presence_penalty=0,
        frequency_penalty=0,
        penalty_decay=1,
        detokenize=False,
    )
    fast_params = params.clone()
    fast_params.logprobs = None
    prompt = {"prompt_token_ids": list(range(1, 74))}
    with irwkv_runner(
        model_path,
        dtype="bfloat16",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        gpu_memory_utilization=0.25,
        num_gpu_blocks_override=64,
        block_size=32,
        mamba_ssm_cache_dtype="float32",
        enable_prefix_caching=True,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        weight_transfer_config={"backend": backend},
    ) as runner:
        llm = runner.llm
        baseline = llm.generate([prompt], params, use_tqdm=False)[0]
        assert llm.generate([prompt], params, use_tqdm=False)[0].num_cached_tokens > 0
        before = llm.collective_rpc(snapshot)[0]
        assert before["graph_ids"] and before["sampler_graph_ids"]
        torch.accelerator.set_device_index(1 if backend == "nccl" else 0)
        source = CheckpointSource()
        if backend == "nccl":
            info = NCCLTrainerInitInfo(
                rank=0,
                master_address="127.0.0.1",
                master_port=get_open_port(),
                world_size=2,
                packed_buffer_size_bytes=512 * 1024**2,
            )
        else:
            info = IPCTrainerInitInfo(
                rank=0, packed=True, packed_buffer_size_bytes=512 * 1024**2
            )
        trainer = WeightTransferTrainerFactory.trainer_init(
            info, client=Client(llm), source=source
        )
        try:
            previous = baseline
            for round_id in (1, 2):
                source.round_id = round_id
                trainer.send_weights()
                after = llm.collective_rpc(snapshot)[0]
                assert after["pointers"] == before["pointers"]
                assert after["graph_ids"] == before["graph_ids"]
                assert after["sampler_graph_ids"] == before["sampler_graph_ids"]
                assert after["canonical_equal"] and after["ipc_released"]
                scale = 0.75 if round_id == 1 else 0.5
                assert after["w1_sum"] == pytest.approx(
                    before["w1_sum"] * scale, rel=1e-4
                )
                assert after["value_sum"] == pytest.approx(
                    before["value_sum"] * scale, rel=1e-4
                )
                assert after["embedding"] != before["embedding"]

                assert llm.reset_prefix_cache()
                graph = llm.generate([prompt], params, use_tqdm=False)[0]
                assert graph.num_cached_tokens == 0
                assert llm.reset_prefix_cache()
                fast = llm.generate([prompt], fast_params, use_tqdm=False)[0]
                assert fast.outputs[0].token_ids == graph.outputs[0].token_ids
                assert llm.reset_prefix_cache()
                llm.collective_rpc(set_graph_enabled, args=(False,))
                eager = llm.generate([prompt], params, use_tqdm=False)[0]
                llm.collective_rpc(set_graph_enabled, args=(True,))
                assert eager.outputs[0].token_ids == graph.outputs[0].token_ids
                for token, graph_probs, eager_probs in zip(
                    graph.outputs[0].token_ids,
                    graph.outputs[0].logprobs,
                    eager.outputs[0].logprobs,
                ):
                    assert graph_probs[token].logprob == pytest.approx(
                        eager_probs[token].logprob, abs=1e-3
                    )
                assert graph.outputs[0].token_ids != previous.outputs[0].token_ids
                previous = graph
        finally:
            trainer.shutdown()


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL"),
    reason="IRWKV_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
def test_real_model_continuation_matches_full_history_prefill(
    irwkv_runner, state_dtype
) -> None:
    """Resuming the next turn from the previous turn's committed state matches
    prefilling the whole history: S(P + delta) == Advance(S(P), delta)."""
    model_path = os.environ["IRWKV_RWKV_TEST_MODEL"]
    params = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=40,
        ignore_eos=True,
        logprobs=3,
        presence_penalty=0,
        frequency_penalty=0,
        penalty_decay=1,
        detokenize=False,
    )
    prompt = list(range(1, 66))
    replies = {}
    continuations = {}
    for prefix_caching in (False, True):
        with irwkv_runner(
            model_path,
            dtype="bfloat16",
            max_model_len=256,
            max_num_seqs=4,
            gpu_memory_utilization=0.8,
            num_gpu_blocks_override=64,
            mamba_ssm_cache_dtype=state_dtype,
            enable_chunked_prefill=True,
            enable_prefix_caching=prefix_caching,
        ) as runner:
            reply = runner.llm.generate([{"prompt_token_ids": prompt}], params)[0]
            history = [*prompt, *reply.outputs[0].token_ids, *range(70, 80)]
            replies[prefix_caching] = reply.outputs[0]
            continuations[prefix_caching] = runner.llm.generate(
                [{"prompt_token_ids": history}], params
            )[0]

    assert replies[True].token_ids == replies[False].token_ids
    # The reply's last token is sampled but never consumed, so the first turn
    # commits 65 + 40 - 1 = 104 tokens, last aligned at 96.
    assert continuations[True].num_cached_tokens == 96
    assert not continuations[False].num_cached_tokens
    resumed = continuations[True].outputs[0]
    prefilled = continuations[False].outputs[0]
    assert resumed.token_ids == prefilled.token_ids
    assert resumed.logprobs == prefilled.logprobs


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL") or torch.accelerator.device_count() < 2,
    reason="a local RWKV checkpoint and two GPUs are required",
)
def test_real_model_uneven_pipeline_matches_single_gpu(irwkv_runner, monkeypatch):
    model_path = os.environ["IRWKV_RWKV_TEST_MODEL"]
    with open(os.path.join(model_path, "config.json")) as config_file:
        num_layers = json.load(config_file)["num_hidden_layers"]
    first_stage = num_layers // 2 - 1
    params = SamplingParams(
        temperature=0,
        max_tokens=8,
        ignore_eos=True,
        logprobs=1,
        detokenize=False,
    )
    prompts = [
        {"prompt_token_ids": list(range(1, 34))},
        {"prompt_token_ids": list(range(7, 54))},
    ]
    outputs = []
    monkeypatch.delenv("IRWKV_PP_LAYER_PARTITION", raising=False)
    for pp_size in (1, 2):
        if pp_size == 2:
            monkeypatch.setenv(
                "IRWKV_PP_LAYER_PARTITION", f"{first_stage},{num_layers - first_stage}"
            )
        with irwkv_runner(
            model_path,
            dtype="bfloat16",
            pipeline_parallel_size=pp_size,
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=4,
            gpu_memory_utilization=0.7,
            num_gpu_blocks_override=16,
            enable_prefix_caching=False,
            enforce_eager=True,
        ) as runner:
            outputs.append(runner.llm.generate(prompts, params))

    for single, pipeline in zip(*outputs):
        expected, actual = single.outputs[0], pipeline.outputs[0]
        assert actual.token_ids == expected.token_ids
        for token, expected_probs, actual_probs in zip(
            expected.token_ids, expected.logprobs, actual.logprobs
        ):
            assert actual_probs[token].logprob == pytest.approx(
                expected_probs[token].logprob, abs=1e-3
            )


def test_reasoning_end_handles_markers_split_across_tokens():
    from irwkv.reasoning import ReasoningParserManager
    from tests.parser.engine.conftest import make_mock_tokenizer

    tokenizer = make_mock_tokenizer(
        {"<": 1, "think": 2, ">": 3, "</": 4, "answer": 5},
        special_tokens=[],
    )
    parser = ReasoningParserManager.get_reasoning_parser("rwkv")(tokenizer)
    tokens = [1, 2, 3, 5, 4, 2, 3, 5]

    assert not parser.is_reasoning_end(tokens[:6])
    assert parser.is_reasoning_end(tokens)
    assert not parser.is_reasoning_end_streaming(tokens[:6], tokens[5:6])
    assert parser.is_reasoning_end_streaming(tokens[:7], tokens[6:7])
    assert parser.extract_content_ids(tokens) == [5]
    assert not parser.is_reasoning_end(tokens + [1, 2, 3])
    assert not parser.is_reasoning_end(tokens + [1, 2])


def test_structured_output_waits_for_split_reasoning_end():
    from irwkv.reasoning import ReasoningParserManager
    from irwkv.v1.structured_output import ConstraintBounds, StructuredOutputManager
    from tests.parser.engine.conftest import make_mock_tokenizer

    tokenizer = make_mock_tokenizer(
        {"<": 1, "think": 2, ">": 3, "</": 4, "answer": 5},
        special_tokens=[],
    )
    parser = ReasoningParserManager.get_reasoning_parser("rwkv")(tokenizer)
    manager = SimpleNamespace(enable_in_reasoning=False, _get_reasoner=lambda _: parser)
    request = SimpleNamespace(
        structured_output_request=SimpleNamespace(reasoning_ended=False),
        all_token_ids=[1, 2, 3, 5, 4],
    )

    bounds = StructuredOutputManager._get_constraint_bounds(manager, request, [2, 3, 5])
    assert bounds == ConstraintBounds(2, 2)


def test_reasoning_parser_streams_text_across_split_markers():
    from irwkv.reasoning import ReasoningParserManager
    from tests.parser.engine.conftest import make_mock_tokenizer

    tokenizer = make_mock_tokenizer(
        {"thought": 1, "</": 2, "think": 3, ">": 4, "answer": 5},
        special_tokens=[],
    )
    parser = ReasoningParserManager.get_reasoning_parser("rwkv")(tokenizer)
    tokens = [1, 2, 3, 4, 5]
    assert parser.extract_reasoning("thought</think>answer", MagicMock()) == (
        "thought",
        "answer",
    )

    previous_text = ""
    reasoning, content = "", ""
    for i, token in enumerate(tokens):
        current_text = tokenizer.decode(tokens[: i + 1])
        delta = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            current_text[len(previous_text) :],
            tokens[:i],
            tokens[: i + 1],
            [token],
        )
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        previous_text = current_text
    delta = parser.finish_streaming()
    if delta:
        reasoning += delta.reasoning or ""
        content += delta.content or ""
    assert (reasoning, content) == ("thought", "answer")


@pytest.mark.parametrize(
    ("prompt", "output", "expected_reasoning"),
    [
        ([1, 2], [3, 5, 4, 2, 3, 6], "thought"),
        ([1, 2, 3, 4, 2], [3, 6], None),
    ],
)
def test_reasoning_parser_completes_marker_from_prompt(
    prompt, output, expected_reasoning
):
    from irwkv.parser.parser_manager import ParserManager
    from tests.parser.engine.conftest import make_mock_tokenizer

    tokenizer = make_mock_tokenizer(
        {"<": 1, "think": 2, ">": 3, "</": 4, "thought": 5, "answer": 6},
        special_tokens=[],
    )
    parser_cls = ParserManager.get_parser(reasoning_parser_name="rwkv")
    assert parser_cls is not None
    request = MagicMock(tool_choice="none", include_reasoning=True)

    parser = parser_cls(tokenizer)
    parser.set_prompt_token_ids(prompt)
    reasoning, content, _ = parser.parse(tokenizer.decode(output), request)
    assert (reasoning, content) == (expected_reasoning, "answer")
    assert parser.count_reasoning_tokens(output) == (1 if expected_reasoning else 0)

    parser = parser_cls(tokenizer)
    reasoning, content = "", ""
    for token in output:
        delta = parser.parse_delta(
            tokenizer.decode([token]), [token], request, prompt, finished=False
        )
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
    assert (reasoning or None, content) == (expected_reasoning, "answer")


def test_structured_output_keeps_content_merged_with_reasoning_end():
    from irwkv.reasoning import ReasoningParserManager
    from irwkv.v1.structured_output import StructuredOutputManager
    from irwkv.v1.structured_output.backend_types import StructuredOutputGrammar
    from tests.parser.engine.conftest import make_mock_tokenizer

    tokenizer = make_mock_tokenizer(
        {"<": 1, "think": 2, ">": 3, "</": 4, ">{": 7, "}": 8, "{": 9},
        special_tokens=[],
    )
    tokenizer.encode.side_effect = lambda text, **kwargs: [9] if text == "{" else []
    parser = ReasoningParserManager.get_reasoning_parser("rwkv")(tokenizer)
    grammar = MagicMock(spec=StructuredOutputGrammar)
    grammar.accept_tokens.return_value = True
    grammar.validate_tokens.side_effect = lambda tokens: tokens
    grammar.is_terminated.return_value = False
    grammar.fill_bitmask.side_effect = lambda mask, index: mask[index].fill_(123)
    request = SimpleNamespace(
        request_id="rwkv",
        use_structured_output=True,
        structured_output_request=SimpleNamespace(
            reasoning_ended=False, grammar=grammar
        ),
        all_token_ids=[1, 2, 3, 4, 2],
    )
    manager = StructuredOutputManager.__new__(StructuredOutputManager)
    manager.enable_in_reasoning = False
    manager._get_reasoner = lambda _: parser
    manager._grammar_bitmask = torch.zeros((1, 1), dtype=torch.int32)
    manager._full_mask = -1
    manager.fill_bitmask_parallel_threshold = 8
    masks = manager.grammar_bitmask({"rwkv": request}, ["rwkv"])
    assert masks[:, 0].tolist() == [-1]
    request.all_token_ids.append(7)
    assert manager.accept_tokens(request, [7])
    grammar.accept_tokens.assert_called_once_with("rwkv", [9])
    assert request.structured_output_request.reasoning_ended
    masks = manager.grammar_bitmask({"rwkv": request}, ["rwkv"])
    assert masks[:, 0].tolist() == [123]


def test_guidance_uses_rwkv_byte_vocabulary():
    from transformers import TokenizersBackend

    from irwkv.v1.structured_output.backend_guidance import GuidanceBackend

    tokenizer = MagicMock(spec=TokenizersBackend)
    tokenizer.backend_tokenizer.model = type("RwkvTrie", (), {})()
    tokenizer.get_vocab.return_value = {
        "<eos>": 0,
        "Ā": 1,
        "Ã": 2,
        "A": 3,
        "Ġ": 4,
        "©": 5,
    }
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = None
    tokenizer.all_special_ids = [0]
    tokenizer.__len__.return_value = 6

    def encode(text, **kwargs):
        if not isinstance(text, str):
            raise TypeError("expected text")
        return [3] if text == "A" else []

    tokenizer.encode.side_effect = encode
    guidance_tokenizer = GuidanceBackend._build_ll_tokenizer(tokenizer, 8)
    assert guidance_tokenizer.vocab_size == 8
    assert guidance_tokenizer.decode_bytes([1, 2, 5, 3, 4]) == b"\x00\xc3\xa9A "
    assert guidance_tokenizer.tokenize_str("A") == [3]


def test_lm_format_enforcer_accepts_a_constrained_choice():
    from irwkv.config import IrwkvConfig
    from irwkv.v1.structured_output.backend_lm_format_enforcer import (
        LMFormatEnforcerBackend,
    )
    from irwkv.v1.structured_output.backend_types import StructuredOutputOptions

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.all_special_ids = [0]
    tokenizer.encode.return_value = [1]
    vocabulary = ["", "0", "yes", "no"]
    tokenizer.decode.side_effect = lambda ids: "".join(vocabulary[i] for i in ids)

    backend = LMFormatEnforcerBackend(IrwkvConfig(), tokenizer, len(vocabulary))
    grammar = backend.compile_grammar(StructuredOutputOptions.CHOICE, '["yes"]')

    assert grammar.accept_tokens("request", [2])
    grammar.reset()
    assert not grammar.accept_tokens("request", [3])
