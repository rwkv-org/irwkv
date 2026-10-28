# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 warmup reserves the KV blocks the scheduler would reserve."""

from types import SimpleNamespace

import pytest
import torch

from irwkv.platforms import current_platform
from irwkv.utils.math_utils import cdiv
from irwkv.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from irwkv.v1.worker.gpu.warmup import (
    _reserved_block_count,
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)
from tests.v1.core.test_prefix_caching import make_kv_cache_manager
from tests.v1.core.utils import create_requests

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024

# `warmup_kernels` ends on `torch.accelerator.synchronize()`.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="warmup synchronizes on the accelerator"
)


def _attention_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )


def _circular_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["circular"],
        CircularBufferSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )


def _mamba_group(mamba_cache_mode: str) -> KVCacheGroupSpec:
    # Name carries the mode so several groups can coexist in one config.
    return KVCacheGroupSpec(
        [f"mamba_{mamba_cache_mode}"],
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode=mamba_cache_mode,
        ),
    )


def _make_runner(
    kv_cache_groups: list[KVCacheGroupSpec],
    decode_query_len: int = 1,
) -> SimpleNamespace:
    """Stub model runner exposing only what the warmup entry points read."""
    return SimpleNamespace(
        rejection_sampler=None,
        decode_query_len=decode_query_len,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=kv_cache_groups, num_blocks=1024
        ),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )


class _StepRecorder:
    """Rebuilds each request's per-group block holdings from the warmup steps."""

    def __init__(self) -> None:
        # (blocks held per group, num_computed_tokens, num_scheduled_tokens)
        self.steps: list[tuple[list[int], int, int]] = []
        self._held: dict[str, list[int]] = {}

    def execute_model(self, scheduler_output) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            self._held[new_req.req_id] = [len(ids) for ids in new_req.block_ids]
            self._record(new_req.req_id, new_req.num_computed_tokens, scheduler_output)
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                self._held[req_id] = [
                    held + len(ids)
                    for held, ids in zip(self._held[req_id], new_block_ids)
                ]
            self._record(req_id, cached.num_computed_tokens[i], scheduler_output)

    def _record(self, req_id: str, num_computed: int, scheduler_output) -> None:
        self.steps.append(
            (
                list(self._held[req_id]),
                num_computed,
                scheduler_output.num_scheduled_tokens[req_id],
            )
        )

    def sample_tokens(self, grammar_output=None) -> None:
        return None


def _assert_covers_scheduled_tokens(steps: list[tuple[list[int], int, int]]) -> None:
    assert steps, "warmup ran no steps"
    for num_blocks, num_computed, num_scheduled in steps:
        num_tokens = min(num_computed + num_scheduled, MAX_MODEL_LEN)
        assert num_blocks[0] >= cdiv(num_tokens, BLOCK_SIZE), (
            f"{num_blocks[0]} blocks for {num_computed}+{num_scheduled} tokens "
            "do not cover the scheduled token range"
        )


@pytest.mark.parametrize("decode_query_len", [1, 2, 4])
def test_warmup_kernels_reserves_scheduled_blocks(decode_query_len):
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner([_attention_group()], decode_query_len),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    _assert_covers_scheduled_tokens(recorder.steps)


def test_mixed_warmup_reserves_scheduled_blocks():
    recorder = _StepRecorder()

    assert run_mixed_prefill_decode_warmup(
        _make_runner([_attention_group()]),
        worker_execute_model=recorder.execute_model,
        worker_sample_tokens=recorder.sample_tokens,
        num_tokens=128,
    )

    _assert_covers_scheduled_tokens(recorder.steps)


def _hybrid_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    """Attention, circular, and Mamba groups exercise every reservation branch."""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            _attention_group(),
            _circular_group(),
            _mamba_group("none"),
            _mamba_group("align"),
        ],
    )


def test_reserved_block_count_matches_real_kv_cache_manager():
    """`_reserved_block_count` predicts real allocations for every cache group."""
    decode_query_len = 1

    kv_cache_config = _hybrid_kv_cache_config(num_blocks=256)
    specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=MAX_MODEL_LEN,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=decode_query_len + 1, block_size=BLOCK_SIZE
    )

    def _step(num_new_tokens: int) -> None:
        computed_blocks, num_new_computed, _ = manager.get_computed_blocks(request)
        blocks = manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed,
            computed_blocks,
        )
        assert blocks is not None, "block pool exhausted"
        held = manager.get_block_ids(request.request_id)
        num_tokens = request.num_computed_tokens + num_new_tokens
        for spec, group_held in zip(specs, held):
            expected = _reserved_block_count(
                num_tokens,
                spec,
                max_model_len=MAX_MODEL_LEN,
                max_encoder_len=0,
            )
            assert len(group_held) == expected, (
                f"{type(spec).__name__} "
                f"mode={getattr(spec, 'mamba_cache_mode', None)}: real manager "
                f"holds {len(group_held)} blocks for {num_tokens} tokens, "
                f"_reserved_block_count predicts {expected}"
            )
        request.num_computed_tokens = num_tokens

    # Prefill followed by decode steps.
    _step(decode_query_len + 1)
    _step(1)
    _step(1)
    _step(1)

    manager.free(request)
