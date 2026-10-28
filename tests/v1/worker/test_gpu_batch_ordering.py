# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch ordering and classification in Model Runner V2 (irwkv.v1.worker.gpu).

split_decodes_and_prefills assumes decode -> short_extend -> prefill request
ordering. With spec decode (decode_query_len > 1), a shorter chunked-prefill
tail sorted in front of the uniform decodes misclassifies every decode as a
prefill.

Conversely, a prompt chunk of exactly decode_query_len tokens has a decode
batch's shape, and must not be classified as a uniform decode batch.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from irwkv.v1.attention.backend import CommonAttentionMetadata
from irwkv.v1.worker.gpu.model_runner import GPUModelRunner, sort_batch_req_ids
from irwkv.v1.worker.utils import get_uniform_decode_token_count


def _make_common_attn_metadata(query_lens: list[int]) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int32), 0, out=query_start_loc[1:]
    )
    seq_lens = torch.tensor([1000 + q for q in query_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        max_seq_len=int(seq_lens.max()),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        block_table_tensor=torch.zeros(num_reqs, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


def _make_runner(
    req_states: dict[str, tuple[int, int]], decode_query_len: int = 8
) -> Any:
    """Build a runner stub from {req_id: (num_computed_tokens, prefill_len)}."""
    prefill_lens = np.array([s[1] for s in req_states.values()], dtype=np.int32)
    num_computed = np.array([s[0] for s in req_states.values()], dtype=np.int32)
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = decode_query_len
    runner.pcp_manager = None
    runner.use_pp = False
    runner.req_states = SimpleNamespace(
        req_id_to_index={req_id: i for i, req_id in enumerate(req_states)},
        # The runner keeps this as min(num_computed_tokens, prefill_len).
        num_computed_prefill_tokens=np.minimum(num_computed, prefill_lens),
        prefill_len=SimpleNamespace(np=prefill_lens),
    )
    return runner


def _uniform_token_count(
    req_states: dict[str, tuple[int, int]],
    query_len: int,
    dummy_run: bool = False,
) -> int | None:
    """Classify via the runner's pre-dispatch gather, as execute_model does."""
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={req_id: query_len for req_id in req_states},
        total_num_scheduled_tokens=query_len * len(req_states),
    )
    runner = _make_runner(req_states, decode_query_len=query_len)
    _, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, dummy_run)
    return uniform_tok_count


def test_prompt_chunk_of_decode_query_len_is_not_uniform_decode():
    # Two prompt chunks whose query length coincides with the K+1 spec-decode
    # query length must reject the batch (issue #49918).
    batch = {f"d{i}": (16, 16) for i in range(6)}
    batch.update({f"p{i}": (8, 40) for i in range(2)})
    assert _uniform_token_count(batch, 8) is None

    # Same collision without spec decoding: 1-token prompt vs 1-token decode.
    assert _uniform_token_count({"p0": (0, 1)}, 1) is None

    # A fresh prompt of exactly decode_query_len tokens.
    assert _uniform_token_count({"p0": (0, 8)}, 8) is None


def test_one_token_prompt_tail_with_prior_context_is_uniform_decode():
    # Replaying the last prompt token over existing context (full prefix-cache
    # hit or P/D KV import) computes exactly like a decode.
    batch = {"d0": (16, 16), "d1": (16, 16), "p0": (128, 129)}
    assert _uniform_token_count(batch, 1) == 1

    # A fresh 1-token prompt alongside it has no prior state.
    batch["p1"] = (0, 1)
    assert _uniform_token_count(batch, 1) is None

    # Only one-token tails qualify, not K+1 chunks with prior context.
    assert _uniform_token_count({"d0": (16, 16), "p0": (120, 129)}, 8) is None


def test_dummy_batches_stay_uniform_decode():
    # Dummy batches have no request state to consult and must stay classified
    # by shape alone; rejecting them would make capture and replay disagree.
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={f"_dummy_req_{i}": 8 for i in range(4)},
        total_num_scheduled_tokens=32,
    )
    state, uniform_tok_count = _make_runner({}).gather_batch_req_state(
        scheduler_output, True
    )
    assert state is None
    assert uniform_tok_count == 8


def test_sort_batch_req_ids_puts_decodes_first():
    # decode_query_len == 1: plain ascending order (decodes first).
    num_tokens_per_req = {"p1": 100, "d1": 1, "p2": 7, "d2": 1}
    assert sort_batch_req_ids(num_tokens_per_req, 1) == ["d1", "d2", "p2", "p1"]


def test_uniform_decode_uses_state_index_not_batch_position():
    """The gather must read each request's own state, not its batch slot.

    Batch position and state index diverge as requests come and go; here the
    only prefilling request sits at state index 0, so a gather keyed on batch
    position would read its row and wrongly reject the two decodes.
    """
    runner: Any = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = 8
    runner.pcp_manager = None
    runner.use_pp = False
    # State arrays in state-index order: a prefilling request, then two decodes.
    runner.req_states = SimpleNamespace(
        req_id_to_index={"prefilling": 0, "decode_a": 1, "decode_b": 2},
        num_computed_prefill_tokens=np.array([8, 16, 16], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([40, 16, 16], dtype=np.int32)),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"decode_a": 8, "decode_b": 8},
        total_num_scheduled_tokens=16,
    )

    state, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, False)
    assert not state.has_prefill
    assert uniform_tok_count == 8

    # Scheduling the prefilling request alongside them must reject the batch.
    scheduler_output.num_scheduled_tokens = {
        "decode_a": 8,
        "prefilling": 8,
        "decode_b": 8,
    }
    scheduler_output.total_num_scheduled_tokens = 24
    state, uniform_tok_count = runner.gather_batch_req_state(scheduler_output, False)
    assert state.has_prefill
    assert uniform_tok_count is None


def test_uniform_decode_predicate():
    # Shape and decode-graph eligibility must both pass.
    assert get_uniform_decode_token_count(2, 16, 8, True) == 8
    assert get_uniform_decode_token_count(2, 16, 8, False) is None
    # 12 tokens over 2 requests is no shared query length.
    assert get_uniform_decode_token_count(2, 12, 8, True) is None
