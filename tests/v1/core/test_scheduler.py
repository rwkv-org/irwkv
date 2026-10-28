# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

from irwkv.config import (
    IrwkvConfig,
    SchedulerConfig,
)
from irwkv.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
from irwkv.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from irwkv.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from irwkv.sampling_params import SamplingParams, StructuredOutputsParams
from irwkv.utils.hashing import sha256
from irwkv.v1.core.encoder_cache_manager import EncoderCacheManager
from irwkv.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from irwkv.v1.core.sched.interface import PauseState
from irwkv.v1.core.sched.output import CachedRequestData, SchedulerOutput
from irwkv.v1.core.sched.scheduler import Scheduler
from irwkv.v1.engine import FinishReason
from irwkv.v1.engine.core import EngineCore
from irwkv.v1.executor.uniproc_executor import UniProcExecutor
from irwkv.v1.outputs import (
    ECConnectorOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    SamplingMaskLists,
)
from irwkv.v1.request import Request, RequestStatus
from irwkv.v1.structured_output import StructuredOutputGrammar

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_make_scheduled_encoder_input_stats_output_embeddings():
    scheduler = create_scheduler()
    mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="image",
            identifier="image-0",
            mm_position=PlaceholderRange(offset=0, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="video-0",
            mm_position=PlaceholderRange(offset=200, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="audio",
            identifier="audio-0",
            mm_position=PlaceholderRange(offset=400, length=49),
        ),
    ]
    scheduler.requests["req"] = Mock(mm_features=mm_features)

    stats = scheduler._make_scheduled_encoder_input_stats({"req": [0, 1, 2]})

    assert stats is not None
    assert stats.num_inputs == 3
    assert stats.output_tokens == 441


def test_scheduled_encoder_input_stats_disabled_without_iteration_logging(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_scheduled_encoder_input_stats_disabled_without_log_stats(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    scheduler.log_stats = False
    scheduler.observability_config.enable_logging_iteration_details = True
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


def test_max_num_active_seqs_caps_admission():
    """Admission is capped by max_num_active_seqs, not max_num_seqs."""
    scheduler = create_scheduler(max_num_seqs=16, max_num_active_seqs=3)
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 3
    assert len(scheduler.running) == 3
    assert len(scheduler.waiting) == 7
    # Execution capacity is untouched.
    assert scheduler.max_num_running_reqs == 16


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule(enable_prefix_caching: bool, prompt_logprobs: int | None):
    """Test scheduling.
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10, prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request


def test_scheduler_stats_route_to_existing_output_client():
    scheduler = create_scheduler()
    request = create_requests(num_requests=1)[0]
    request.client_index = 1
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
    )

    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert 0 not in engine_core_outputs
    assert engine_core_outputs[1].scheduler_stats is not None
    assert len(engine_core_outputs[1].outputs) == 1


def test_async_scheduling_pp_reschedules_after_pipeline_cadence(monkeypatch):
    """Respect the pipeline broadcast slot cadence for an in-flight request."""
    monkeypatch.setattr(
        "irwkv.config.parallel.current_platform.device_count", lambda: 2
    )
    scheduler = create_scheduler(async_scheduling=True, pipeline_parallel_size=2)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(req)

    _ = scheduler.schedule()
    assert req.num_output_placeholders > 0

    assert req.request_id not in scheduler.schedule().num_scheduled_tokens
    assert req.request_id in scheduler.schedule().num_scheduled_tokens


def test_cached_request_data_omits_resumed_token_ids():
    """The runner retains resumed token IDs in its request state."""
    from irwkv.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler = create_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=8)
    req.append_output_token_ids([101, 102, 103])
    cached = scheduler._make_cached_request_data(
        running_reqs=[],
        resumed_reqs=[req],
        num_scheduled_tokens={req.request_id: 1},
        req_to_new_blocks={req.request_id: KVCacheBlocks(blocks=((),))},
    )
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids == {}


@pytest.mark.parametrize("has_running", [True, False])
def test_schedule_prefills_gating(has_running: bool):
    """DP prefill-balancing gate: when `throttle_prefills` is True, a new
    WAITING (prefill) request is deferred ONLY if this rank has running work to
    protect. With no running requests, the prefill is admitted regardless (so a
    throttled step is never wasted as a dummy), and running/decode requests are
    unaffected. Once the cadence allows prefills again, the request is admitted.
    """
    scheduler = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192)

    if has_running:
        # Establish a running (decode) request via a prefill + output step.
        (running_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
        scheduler.add_request(running_req)
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["run0"],
                req_id_to_index={"run0": 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        assert len(scheduler.running) == 1

    # Add a new WAITING (prefill) request, with prefills gated off.
    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule(throttle_prefills=True)

    if has_running:
        # There is running work to protect, so the new prefill is deferred...
        assert "new0" not in output.num_scheduled_tokens
        assert new_req.status == RequestStatus.WAITING
        # ...while the running/decode request keeps being scheduled.
        assert "run0" in output.num_scheduled_tokens
        # When the cadence allows prefills again, the request is admitted.
        output = scheduler.schedule()

    # No running work to protect (or cadence now open): the prefill is admitted.
    assert "new0" in output.num_scheduled_tokens
    assert any(r.req_id == "new0" for r in output.scheduled_new_reqs)


def test_throttle_defers_inflight_prefill_chunk():
    """DP prefill balancing throttles ALL prefill compute on a throttled step,
    not just new admissions: an in-progress (chunked) prefill already in the
    running queue is also deferred, so the step runs decode-only, while a
    separate decode keeps being scheduled."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    # A short request that finishes prefill in one step -> a running decode.
    (decode_req,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(decode_req)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0"],
            req_id_to_index={"dec0": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )
    assert decode_req in scheduler.running and not decode_req.is_prefill_chunk

    # A long request (80 tokens, budget 50) -> prefilled in chunks.
    (chunk_req,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk_req)
    output = scheduler.schedule()  # first chunk of chk0 + decode of dec0
    assert output.num_scheduled_tokens["chk0"] > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0", "chk0"],
            req_id_to_index={"dec0": 0, "chk0": 1},
            sampled_token_ids=[[0], []],  # no token sampled for partial prefill
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )
    assert chunk_req.is_prefill_chunk  # still mid-prefill, in running

    # Throttled step: the in-flight prefill chunk is deferred, the decode runs.
    output = scheduler.schedule(throttle_prefills=True)
    assert "chk0" not in output.num_scheduled_tokens
    assert "dec0" in output.num_scheduled_tokens

    # When the cadence opens again, the prefill chunk resumes.
    output = scheduler.schedule()
    assert "chk0" in output.num_scheduled_tokens


def test_throttle_capacity_bound_guard_admits():
    """Saturation guard: if a cadence-aligned release step cannot drain the
    waiting prefill queue (it ran out of token budget), the throttle backs off on
    the next step so the backlog cannot grow into a TTFT avalanche -- prefills are
    admitted even though throttle_prefills is set."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=200, enable_chunked_prefill=True
    )
    a, b = create_requests(num_requests=2, num_tokens=200, req_ids=["a", "b"])
    scheduler.add_request(a)
    scheduler.add_request(b)

    # Release step (throttle off): `a` fills the 200-token budget; `b` cannot be
    # reached, so the waiting queue is not drained -> capacity-bound.
    output = scheduler.schedule()
    assert "a" in output.num_scheduled_tokens
    assert "b" not in output.num_scheduled_tokens
    assert scheduler.prefill_capacity_bound

    # Throttle. Because the previous release was capacity-bound, the guard backs
    # off and `b` is admitted rather than stalling the backlog.
    output = scheduler.schedule(throttle_prefills=True)
    assert "b" in output.num_scheduled_tokens


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_schedule_concurrent_partial_requests(enable_prefix_caching: bool):
    """Test scheduling behavior with concurrent partial requests.

    This test verifies that: there are multiple long prefill requests in the
    RUNNING state, and we can schedule them together.

    """
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert output1.scheduled_cached_reqs.num_reqs == 3
    assert len(output1.finished_req_ids) == 0
    assert output1.num_scheduled_tokens[requests[0].request_id] == 400
    assert output1.num_scheduled_tokens[requests[1].request_id] == 400
    assert output1.num_scheduled_tokens[requests[2].request_id] == 224

    # Schedule the third step. All three requests are running.
    # First and second requests are in the decode stage.
    # All the remaining tokens in the third request are processed.
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]] + [[] for _ in range(len(requests) - 2)],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert output2.scheduled_cached_reqs.num_reqs == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[requests[2].request_id] == 800 - 224 - 224


def test_long_prefill_threshold_ignored_when_alone():
    """A lone long prefill is not capped by the threshold: it has no other
    request to starve."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
    )
    request = create_requests(num_requests=1, num_tokens=2000)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == 1024


def test_long_prefill_threshold_applies_with_other_requests():
    """The threshold caps the prefill as soon as another request is queued."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
    )
    long_req = create_requests(num_requests=1, num_tokens=2000)[0]
    short_req = create_requests(num_requests=1, num_tokens=10, req_ids=["short"])[0]
    for request in [long_req, short_req]:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[long_req.request_id] == 400
    assert output.num_scheduled_tokens[short_req.request_id] == 10


def test_long_prefill_threshold_floored_by_fair_share():
    """With the adaptive flag, the effective threshold never falls below the
    fair share of the token budget: max_num_batched_tokens / num queued +
    running requests."""
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=100,
        long_prefill_token_threshold_adaptive=True,
    )
    long_req = create_requests(num_requests=1, num_tokens=2000)[0]
    short_req = create_requests(num_requests=1, num_tokens=10, req_ids=["short"])[0]
    for request in [long_req, short_req]:
        scheduler.add_request(request)

    output = scheduler.schedule()
    # 100 is below the fair share (1024 // 2 = 512), so the floor binds.
    assert output.num_scheduled_tokens[long_req.request_id] == 512
    assert output.num_scheduled_tokens[short_req.request_id] == 10


def test_update_from_output_routes_sampling_masks_by_request():
    """Each request receives the sampler row at its own batch index."""
    scheduler = create_scheduler()
    scheduler.return_sampling_mask = True
    requests = create_requests(num_requests=3, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 1 for req in requests},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[1], [3], [4]],
        logprobs=None,
        prompt_logprobs_dict={},
        sampling_masks=SamplingMaskLists(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), np.array([0, 2, 3, 6])
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)[0].outputs

    assert [out.request_id for out in outputs] == [req.request_id for req in requests]
    assert [out.new_sampling_mask.token_ids.tolist() for out in outputs] == [
        [1, 2],
        [3],
        [4, 5, 6],
    ]
    assert all(out.new_sampling_mask.offsets is None for out in outputs)


def test_update_from_output_routes_multi_position_sampling_masks():
    scheduler = create_scheduler()
    scheduler.return_sampling_mask = True
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 1 for req in requests},
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[1, 2], [3, 4, 5]],
        logprobs=None,
        prompt_logprobs_dict={},
        sampling_masks=SamplingMaskLists(
            token_ids=np.array([1, 6, 2, 3, 7, 8, 4, 5, 9], dtype=np.int32),
            offsets=np.array([0, 2, 3, 6, 7, 9]),
            cu_num_generated_tokens=[0, 2, 5],
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)[0].outputs

    assert [out.new_sampling_mask.to_nested_list() for out in outputs] == [
        [[1, 6], [2]],
        [[3, 7, 8], [4], [5, 9]],
    ]


def test_check_stop_min_tokens():
    """Test that sampled stop tokens still finish under min_tokens."""
    from irwkv.v1.core.sched.utils import check_stop

    # Test case 1: regular tokens below min_tokens should keep running.
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
    )
    request.append_output_token_ids([10, 11, 12])

    result = check_stop(request, max_model_len=100)
    assert result is False, "Should not stop when num_output_tokens<min_tokens"

    # Test case 2: if EOS is sampled, it should stop even below min_tokens.
    request_eos = Request(
        request_id="eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
    )
    request_eos.append_output_token_ids([10, 11, EOS_TOKEN_ID])
    result = check_stop(request_eos, max_model_len=100)
    assert result is True, "Sampled EOS should stop even below min_tokens"
    assert request_eos.status == RequestStatus.FINISHED_STOPPED

    # Test case 3: ignore_eos should still prevent EOS-based stopping.
    sampling_params_ignore_eos = SamplingParams(
        ignore_eos=True,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params_ignore_eos.update_from_generation_config({}, EOS_TOKEN_ID)
    request_ignore_eos = Request(
        request_id="ignore-eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_ignore_eos,
    )
    request_ignore_eos.append_output_token_ids([EOS_TOKEN_ID])
    result = check_stop(request_ignore_eos, max_model_len=100)
    assert result is False, "ignore_eos should still suppress EOS stopping"

    # Test case 4: min_tokens = 0 should follow normal stopping logic.
    sampling_params_no_min = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=0,
    )
    sampling_params_no_min.update_from_generation_config({}, EOS_TOKEN_ID)
    request_no_min = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_no_min,
    )
    request_no_min.append_output_token_ids([10, EOS_TOKEN_ID])

    result = check_stop(request_no_min, max_model_len=100)
    assert result is True, "Should stop on EOS when min_tokens=0"
    assert request_no_min.status == RequestStatus.FINISHED_STOPPED

    # Test case 5: if a stop token is sampled, it should stop immediately.
    sampling_params_stop = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
        stop_token_ids=[42],
    )
    sampling_params_stop.update_from_generation_config({}, EOS_TOKEN_ID)
    request_stop = Request(
        request_id="2",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_stop,
    )
    request_stop.append_output_token_ids([10, 11, 42])
    result = check_stop(request_stop, max_model_len=100)
    assert result is True, "Sampled stop tokens should stop even below min_tokens"
    assert request_stop.status == RequestStatus.FINISHED_STOPPED
    assert request_stop.stop_reason == 42


def test_check_stop_length_cap_below_min_tokens():
    """Length caps must finish a request even below min_tokens.

    Guards against requests hanging forever when min_tokens cannot be
    satisfied: min_tokens only defers EOS/stop tokens, so a request that
    reaches max_model_len (or a max_tokens clamped below min_tokens) must
    still finish as FINISHED_LENGTH_CAPPED instead of spinning in the
    scheduler with zero new tokens to schedule.
    """
    from irwkv.v1.core.sched.utils import check_stop

    # Reaching max_model_len below min_tokens must stop the request.
    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
    )
    # 7 output tokens: num_tokens == 10 == max_model_len, min_tokens unmet.
    request.append_output_token_ids([10, 11, 12, 13, 14, 15, 16])

    result = check_stop(request, max_model_len=10)
    assert result is True, "Should stop at max_model_len even below min_tokens"
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # Reaching max_tokens below min_tokens must stop the request. This can
    # happen when max_tokens is clamped to the context window after
    # SamplingParams validation (e.g. max_tokens=None on a long prompt).
    sampling_params_capped = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params_capped.max_tokens = 3
    sampling_params_capped.update_from_generation_config({}, EOS_TOKEN_ID)
    request_capped = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_capped,
    )
    request_capped.append_output_token_ids([10, 11, 12])

    result = check_stop(request_capped, max_model_len=100)
    assert result is True, "Should stop at max_tokens even below min_tokens"
    assert request_capped.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule_concurrent_batches(
    enable_prefix_caching: bool, prompt_logprobs: int | None
):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


def test_schedule_order():
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=3,
    )

    # long requests
    requests = create_requests(num_requests=2, num_tokens=800, req_ids=["1", "2"])
    # short requests
    requests += create_requests(num_requests=2, num_tokens=10, req_ids=["3", "4"])

    for request in requests:
        scheduler.add_request(request)

    scheduler_output1 = scheduler.schedule()

    assert len(scheduler_output1.scheduled_new_reqs) == 2


def test_preempt_during_execution():
    # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    # because block 0 is reserved as the null block.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.num_scheduled_tokens) == 1
    assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

    # Schedule the second request while the first request is still running.
    # This scenario can occur in certain cases, when max_concurrent_batches > 1
    # (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.num_scheduled_tokens) == 1
    assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

    # Get the output of the first request.
    model_runner_output0 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output0)

    # Schedule the first request again. This will cause the preemption
    # of the second request because the KV cache is full.
    _ = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert scheduler.running[0] == requests[0]
    assert requests[1].status == RequestStatus.PREEMPTED

    model_runner_output1 = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output1)

    # The second request (that is preempted) should be updated with the
    # sampled token id.
    assert len(requests[1].output_token_ids) == 1
    assert requests[1].output_token_ids[0] == 42


def test_preemption_re_records_prefix_cache_query():
    """A preempted request re-enters the lookup on resume, so its recomputation
    is counted again into the preempted stats."""
    scheduler = create_scheduler(enable_prefix_caching=True)
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.preempted_requests) == (1, 0)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )
    assert request.num_stale_output_tokens == 0
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None

    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert stats.preempted_requests == 1


def test_prefix_cache_stats_not_recorded_when_caching_disabled():
    """With prefix caching off there is no local lookup, so admitting a request
    records no phantom miss."""
    scheduler = create_scheduler(enable_prefix_caching=False)
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)

    scheduler.schedule()

    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)


def test_prefix_cache_stats_counted_once_for_retried_then_scheduled_request():
    """A real cache hit rejected once by allocate_slots and admitted on the next
    step is counted exactly once, hits included."""
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
    )

    # Seed the cache so the next request with the same prompt hits it.
    seed = create_requests(
        num_requests=1,
        num_tokens=block_size * 2,
        max_tokens=2,
        same_prompt=True,
        block_size=block_size,
        req_ids=["seed"],
    )[0]
    scheduler.add_request(seed)
    _step_until_done(
        scheduler,
        scheduler.schedule(),
        ModelRunnerOutput(
            req_ids=["seed"],
            req_id_to_index={"seed": 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )

    # The seeding step swapped in a fresh accumulator, so re-read it; the
    # retried request must be the only thing recorded from here on.
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    retried = create_requests(
        num_requests=1,
        num_tokens=block_size * 3,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["retried"],
    )[0]
    scheduler.add_request(retried)

    # Reject the first allocation attempt, then delegate to the real one.
    orig_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    allocate_results: list = []

    def spy_allocate_slots(*args, **kwargs):
        result = None if not allocate_results else orig_allocate_slots(*args, **kwargs)
        allocate_results.append(result)
        return result

    scheduler.kv_cache_manager.allocate_slots = spy_allocate_slots

    assert not scheduler.schedule().scheduled_new_reqs
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    assert "retried" in scheduler.schedule().num_scheduled_tokens
    assert allocate_results[0] is None and allocate_results[1] is not None
    assert (stats.requests, stats.queries, stats.hits) == (
        1,
        retried.num_tokens,
        block_size * 2,
    )


def test_scheduler_reset_prefix_cache():
    scheduler = create_scheduler(enable_prefix_caching=True)
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    # Initial scheduling, requests should be at the running state now
    _ = scheduler.schedule()

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request

    # Reset prefix cache should fail since there are still running requests
    # and they are taking KV cache
    assert not scheduler.reset_prefix_cache()

    # Pause completes pending model outputs before the caller resets the scheduler.
    for request in requests:
        request.num_in_flight_tokens = 0

    assert scheduler.reset_prefix_cache(reset_running_requests=True)

    # Verify requests moved from running to waiting
    assert len(scheduler.waiting) == len(requests)
    assert len(scheduler.running) == 0

    for i, request in enumerate(requests):
        assert scheduler.waiting[i] == request


def test_kv_cache_release_after_keep_pause_preserves_requests():
    """A completed keep pause permits release and recomputation of live requests."""
    scheduler = create_scheduler(max_num_seqs=1, enable_prefix_caching=True)
    running, waiting = create_requests(num_requests=2)
    for request in (running, waiting):
        scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[running.request_id],
            req_id_to_index={running.request_id: 0},
            sampled_token_ids=[[1]],
        ),
    )
    assert scheduler.kv_cache_manager.get_block_ids(running.request_id)[0]

    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = Mock(is_sleeping=False)
    core.mm_receiver_cache = None
    core.batch_queue = None

    assert core.pause_scheduler(mode="keep", clear_cache=False) is None
    assert scheduler.running == [running]
    assert list(scheduler.waiting) == [waiting]
    assert not scheduler.has_requests()

    core.release_kv_cache_memory()

    core.model_executor.discard.assert_called_once_with(("kv_cache",))
    assert scheduler.requests == {
        request.request_id: request for request in (running, waiting)
    }
    assert not scheduler.running
    assert list(scheduler.waiting) == [running, waiting]
    assert running.status == RequestStatus.PREEMPTED
    assert running.num_computed_tokens == 0
    assert list(running.output_token_ids) == [1]
    assert not scheduler.kv_cache_manager.get_block_ids(running.request_id)[0]
    assert scheduler.schedule().total_num_scheduled_tokens == 0

    assert core.wake_up(tags=["kv_cache"])
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {running.request_id: running.num_tokens}


@pytest.mark.parametrize(
    "sleeping_tags",
    [{"weights", "kv_cache"}, {"weights"}, {"kv_cache"}],
    ids=["full-sleep", "weights-only", "kv-only-or-repeated-release"],
)
def test_kv_cache_release_rejects_nonresident_memory(sleeping_tags):
    """Reject release without resetting caches or discarding more memory."""
    core = object.__new__(EngineCore)
    core.scheduler = create_scheduler()
    core.scheduler.set_pause_state(PauseState.PAUSED_ALL)
    core.batch_queue = None
    core._reset_caches = Mock()
    core.model_executor = object.__new__(UniProcExecutor)
    core.model_executor.sleeping_tags = sleeping_tags.copy()
    core.model_executor.collective_rpc = Mock()

    with pytest.raises(
        RuntimeError, match="requires all executor memory to be resident"
    ):
        core.release_kv_cache_memory()

    core._reset_caches.assert_not_called()
    core.model_executor.collective_rpc.assert_not_called()
    assert core.model_executor.sleeping_tags == sleeping_tags


def test_reset_connector_cache_no_connector_is_no_op_success():
    """``reset_connector_cache`` must return True when no connector is
    configured.

    Without this, ``reset_prefix_cache(reset_connector=True)`` returns
    ``False`` on every engine that doesn't have a KV connector configured —
    even when the local prefix cache reset succeeded — and any caller that
    interprets the return value as "did the reset I asked for succeed?"
    sees a spurious failure.
    """
    scheduler = create_scheduler(enable_prefix_caching=True)
    assert scheduler.connector is None

    # No-connector reset is treated as success.
    assert scheduler.reset_connector_cache() is True

    # End-to-end: reset_prefix_cache(reset_connector=True) on an idle
    # scheduler succeeds with or without a connector.
    assert scheduler.reset_prefix_cache(reset_connector=True) is True


def _model_output(scheduler, output, sampled):
    """Feed `sampled` (per-request list) back to the scheduler."""
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )


def test_scheduler_stats_waiting_queues():
    """Test that scheduler stats correctly report capacity-bound and blocked
    waiting requests."""
    # Create scheduler with limited capacity so we can have waiting requests
    scheduler = create_scheduler(max_num_batched_tokens=100)

    # Create requests: some will be scheduled, some will wait on capacity,
    # and some will be blocked by constraints
    all_requests = create_requests(num_requests=5, num_tokens=50)

    # Add 3 requests - only 2 can be scheduled (2 * 50 = 100 tokens)
    # The 3rd will remain in waiting queue (capacity constraint)
    for request in all_requests[:3]:
        scheduler.add_request(request)

    # Add 2 more that are blocked on grammar compilation
    for request in all_requests[3:]:
        request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        scheduler.add_request(request)

    # Schedule - this will schedule 2 requests, leaving 1 in waiting
    output = scheduler.schedule()

    # Verify: 2 scheduled, 1 still waiting on capacity, 2 blocked by constraints
    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.waiting) == 3
    assert len(scheduler.deferred_waiting) == 2

    # Call update_from_output() to get frontend-facing stat
    scheduled_req_ids = list(output.num_scheduled_tokens.keys())
    model_runner_output = ModelRunnerOutput(
        req_ids=scheduled_req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(scheduled_req_ids)},
        sampled_token_ids=[[1]] * len(scheduled_req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    assert engine_core_outputs and len(engine_core_outputs) > 0
    stats = engine_core_outputs[0].scheduler_stats
    assert stats is not None

    # Verify stats match queue lengths after scheduling
    assert stats.num_running_reqs == 2  # 2 were scheduled
    assert stats.num_waiting_reqs == 1  # 1 waiting on capacity
    assert stats.num_skipped_waiting_reqs == 2  # 2 blocked by constraints


def _assert_right_scheduler_output(
    output: SchedulerOutput,
    num_requests: int,
    expected_num_scheduled_tokens: int,
):
    """Check if SchedulerOutput is correct after remote KV cache hit."""
    # We should inject the kv_connector_metadata.
    assert len(output.kv_connector_metadata.requests) == num_requests

    # Only num_tokens - matched_num_new_tokens should be scheduled.
    for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
        assert num_scheduled_tokens == expected_num_scheduled_tokens


def _assert_right_kv_cache_manager(
    scheduler: Scheduler,
    requests: list[Request],
    num_tokens: int,
    block_size: int,
    num_requests: int,
    num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""
    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req in requests:
        blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks[req.request_id]
        hashes = req.block_hashes
        assert (
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block[req.request_id]
            == EXPECTED_TOTAL_BLOCKS
        )
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        == num_total_blocks - num_requests * BLOCKS_PER_REQ
    )


def _step_until_done(
    scheduler: Scheduler,
    output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
):
    """Loop over schedule(), update_from_output() until finished."""
    all_finished = False
    _ = scheduler.update_from_output(output, model_runner_output)
    while not all_finished:
        # Schedule + a few iterations until stopping.
        output = scheduler.schedule()
        assert len(scheduler.running)
        for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
            # We should be in the decode phase now.
            assert num_scheduled_tokens == 1
        if scheduler.connector is not None:
            assert len(output.kv_connector_metadata.requests) == 0
        if scheduler.ec_connector is not None:
            assert len(output.ec_connector_metadata.mm_datas) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)[0]
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def _num_waiting_requests(scheduler: Scheduler) -> int:
    return len(scheduler.waiting) + len(scheduler.kv_holding_waiting)


def _step_until_kv_transfer_finished(scheduler: Scheduler, req_ids: list[str]):
    """Cycle requests through a KV transfer cycle."""
    # Requests should first transition to WAITING_FOR_REMOTE_KVS
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0
    assert len(output.scheduled_new_reqs) == 0
    for req in scheduler.requests.values():
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # No model execution yet
    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    initial_ecos = scheduler.update_from_output(output, EMPTY_OUTPUT)

    # Simulate KV transfer completion using KVConnectorOutput.finished_recving
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        kv_connector_output=KVConnectorOutput(finished_recving=req_ids),
    )
    scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    for req_id in req_ids:
        assert req_id in scheduler.finished_recving_kv_req_ids

    return initial_ecos


def _runner_output(
    req_ids: list[str], finished_recving: set[str] | None = None
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1] for _ in req_ids],
        kv_connector_output=KVConnectorOutput(finished_recving=finished_recving),
    )


def test_pause_new_ignores_streaming_session_waiting_for_input():
    """A session idle in WAITING_FOR_STREAMING_REQ cannot progress, so it must
    not keep a PAUSED_NEW drain from completing."""
    stop_token = 7
    scheduler = create_scheduler()
    (session,) = create_requests(
        num_requests=1, num_tokens=4, stop_token_ids=[stop_token], req_ids=["s"]
    )
    session.resumable = True
    scheduler.add_request(session)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["s"], req_id_to_index={"s": 0}, sampled_token_ids=[[stop_token]]
        ),
    )
    assert session in scheduler.kv_holding_waiting

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0


def make_output(scheduler: Scheduler):
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={req.request_id: i for i, req in enumerate(scheduler.running)},
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
    )


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        )
        == 0
    )
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block
        )
        == 0
    )
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert num_free_blocks == (scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0
        # assert block._block_hash is None
    # assert (
    #     len(scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    #           ) == 0)


def test_memory_leak():
    """Test that we do not have a memory leak."""
    scheduler = create_scheduler(enable_prefix_caching=True)

    NUM_REQUESTS = 5
    NUM_TOKENS = 10
    MAX_TOKENS = 10
    requests = create_requests(
        num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS, max_tokens=MAX_TOKENS
    )

    # Add each request.
    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Iterate until done.
    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm no memory leak.
    assert_scheduler_empty(scheduler)


_none_hash_initialized = False


def create_requests_with_priority(
    num_requests: int,
    priorities: list[int],
    arrival_times: list[float] | None = None,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    starting_idx: int = 0,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
):
    """Create requests with specified priorities and arrival times."""
    assert len(priorities) == num_requests
    if arrival_times is not None:
        assert len(arrival_times) == num_requests
    else:
        arrival_times = [float(i) for i in range(num_requests)]

    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        prompt_logprobs=prompt_logprobs,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    requests = []

    if mm_hashes_list is not None:
        # NOTE: allow manual input; some mm items can have the same identifier
        # no. of mm_hashes and mm_positions for each request should be identical
        assert mm_positions is not None, (
            "mm_positions must be provided when mm_hashes_list is provided"
        )
        assert len(mm_hashes_list) == len(mm_positions) == num_requests
        assert [len(h) for h in mm_hashes_list] == [len(p) for p in mm_positions]

        # Since same identifier would imply they are identical encoder output
        # Verify mm items with identical identifier are having mm_position.length
        seen_hashes: dict[str, int] = {}

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i + starting_idx}" for i in range(num_requests)]

    for i in range(num_requests):
        mm_features = []

        for j, position in enumerate(
            mm_positions[i] if mm_positions is not None else []
        ):
            if mm_hashes_list is not None:
                identifier = mm_hashes_list[i][j]

                # Verify if position length is identical
                position_length = position.length
                if identifier in seen_hashes:
                    assert seen_hashes[identifier] == position_length, (
                        f"mm_hash '{identifier}' has inconsistent position lengths: "
                        f"previously {seen_hashes[identifier]}, now {position_length} "
                        f"at request {i}, position {j}"
                    )
                else:
                    seen_hashes[identifier] = position_length
            else:
                # Unique dummy hash for each mm item
                identifier = f"hash{i}_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

        prompt_token_ids = (
            [starting_idx] * num_tokens
            if same_prompt
            else [i + starting_idx] * num_tokens
        )
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            mm_features=mm_features if mm_features else None,
            arrival_time=arrival_times[i],
            priority=priorities[i],
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def test_priority_scheduling_basic_ordering():
    """Test that requests are scheduled in priority order
    (lower value = higher priority)."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
    )

    # Create requests with different priorities
    # Priority 0 (highest), 1, 2 (lowest)
    priorities = [2, 0, 1]  # Add in non-priority order
    arrival_times = [1.0, 2.0, 3.0]  # All different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-priority order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify priority order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in priority order:
    # req_1 (priority 0), req_2 (priority 1), req_0 (priority 2)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_arrival_time_tiebreaker():
    """Test that arrival time is used
    as tiebreaker when priorities are equal."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
    )

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1]  # All same priority
    arrival_times = [3.0, 1.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-arrival order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify arrival time order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in arrival time order:
    # req_1 (1.0), req_2 (2.0), req_0 (3.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_mixed_priority_and_arrival():
    """Test priority scheduling with mixed priorities and arrival times."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
    )

    # Create requests with mixed priorities and arrival times
    priorities = [2, 1, 1, 0]  # Mixed priorities
    arrival_times = [1.0, 3.0, 2.0, 4.0]  # Mixed arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 4

    # Expected order:
    # 1. req_3 (priority 0, arrival 4.0)
    # 2. req_2 (priority 1, arrival 2.0) - earlier arrival than req_1
    # 3. req_1 (priority 1, arrival 3.0)
    # 4. req_0 (priority 2, arrival 1.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["3", "2", "1", "0"]


def test_priority_scheduling_preemption():
    """Test that under KV block pressure the scheduler preempts the
    lowest-priority *running* request, not the highest-priority one.

    A low-priority request starts running first. Then a high-priority
    request arrives and is admitted to running.  When block pressure
    builds, the scheduler preempts the low-priority running request
    while keeping the high-priority one.

    Block math
    ----------
    block_size = 16, num_blocks = 6 (1 null → 5 usable).

    Phase 1: lo1 (priority 5, 32 tokens) → 2 blocks.  3 free.
             Decode → lo1 has 33 tokens (needs 3rd block on next schedule).
    Phase 2: hi1 (priority 0, 32 tokens) arrives.
             schedule() allocates lo1's 3rd block (3 used) and admits
             hi1 (2 blocks) → 5 used, 0 free. Both running.
             Decode → lo1 34 tokens, hi1 33 tokens.
    Phase 3: schedule() → hi1 needs 3rd block, 0 free → preemption.
             lo1 (priority 5) is preempted, hi1 (priority 0) survives.
    """
    block_size = 16
    num_blocks = 6  # 1 null block → 5 usable
    num_tokens = block_size * 2  # 32 tokens = exactly 2 blocks

    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=3,
        max_num_batched_tokens=200,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    # --- Phase 1: low-priority request starts running ---
    lo1 = create_requests_with_priority(
        num_requests=1,
        priorities=[5],
        arrival_times=[1.0],
        num_tokens=num_tokens,
        req_ids=["lo1"],
    )[0]
    scheduler.add_request(lo1)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Decode: lo1 now has 33 tokens (crosses 32-token boundary).
    model_output = ModelRunnerOutput(
        req_ids=["lo1"],
        req_id_to_index={"lo1": 0},
        sampled_token_ids=[[100]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 2: high-priority request arrives AFTER lo1 is running ---
    hi1 = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[2.0],
        num_tokens=num_tokens,
        req_ids=["hi1"],
    )[0]
    scheduler.add_request(hi1)

    # schedule(): lo1 gets its 3rd block (3 used), hi1 admitted (5 used,
    # 0 free).  Both are now running.
    output = scheduler.schedule()
    assert any(r.req_id == "hi1" for r in output.scheduled_new_reqs)
    assert len(scheduler.running) == 2

    # Decode: lo1 → 34 tokens, hi1 → 33 tokens.
    model_output = ModelRunnerOutput(
        req_ids=["lo1", "hi1"],
        req_id_to_index={"lo1": 0, "hi1": 1},
        sampled_token_ids=[[101], [100]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 3: preemption with mixed-priority running requests ---
    # hi1 needs a 3rd block but 0 are free.  The scheduler picks the
    # lowest-priority running request to preempt:
    #   max(running, key=(priority, arrival_time)) → lo1 (5 > 0).
    output = scheduler.schedule()

    lo1_req = scheduler.requests["lo1"]
    assert lo1_req.status == RequestStatus.PREEMPTED, (
        "Expected low-priority 'lo1' to be preempted"
    )
    assert any(req.request_id == "hi1" for req in scheduler.running), (
        "High-priority 'hi1' should still be running"
    )


def test_priority_scheduling_no_preemption_when_space_available():
    """Test that preemption doesn't happen
    when there's space for new requests."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=3,  # Allow 3 concurrent requests
        max_num_batched_tokens=200,  # Sufficient token budget
    )

    # Add two low-priority running requests
    low_priority_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=30,
        req_ids=["lo1", "lo2"],
    )

    for request in low_priority_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in low_priority_requests],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(low_priority_requests)
        },
        sampled_token_ids=[[100] for _ in low_priority_requests],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(output, model_output)

    # Add high-priority request
    high_priority_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=30,
        req_ids=["hi1"],
    )[0]

    scheduler.add_request(high_priority_request)

    # Schedule - should not preempt since there's space
    output = scheduler.schedule()

    # Should schedule the new request without preemption
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.running) == 3  # All three requests running
    assert len(scheduler.waiting) == 0  # No requests waiting


def test_priority_scheduling_preemption_victim_selection():
    """Test that the correct victim is selected for
    preemption based on priority and arrival time."""
    # This test verifies the priority-based victim selection logic
    # by checking the waiting queue order after adding requests with different
    # priorities
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=1,  # Force sequential processing to test priority order
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[3, 2, 0],  # Different priorities: low, medium, high
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_2, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "2"  # Highest priority

    # Verify the waiting queue has the remaining requests in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify priority order
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_1 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["1", "0"]


def test_priority_scheduling_equal_priority_preemption():
    """Test arrival time tiebreaker when requests have equal priority."""
    # This test verifies that arrival time is used as a tiebreaker for equal
    # priorities
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=1,  # Force sequential processing
    )

    # Create requests with same priority but different arrival times
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[2, 2, 2],  # Same priority
        arrival_times=[3.0, 1.0, 2.0],  # Different arrival times
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should schedule the request with earliest arrival time
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "1"  # Earliest arrival (1.0)

    # Verify the waiting queue has remaining requests in arrival time order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify arrival time order
    waiting_requests = list(scheduler.waiting)

    waiting_arrival_times = [req.arrival_time for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (arrival 2.0) then req_0 (arrival 3.0)
    assert waiting_arrival_times == [2.0, 3.0]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_waiting_queue_order():
    """Test that the waiting queue maintains priority order."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Create multiple requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_3, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "3"

    # Verify waiting queue has remaining requests in priority order
    assert len(scheduler.waiting) == 3

    # Extract requests from waiting queue
    # (it's a heap, so we need to pop to see order)
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be ordered by priority: req_1 (1), req_2 (2), req_0 (3)
    assert waiting_req_ids == ["1", "2", "0"]
    assert waiting_priorities == [1, 2, 3]


def test_priority_scheduling_fcfs_fallback():
    """Test that FCFS behavior is maintained when all
    requests have same priority."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
    )

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1, 1]  # All same priority
    arrival_times = [4.0, 1.0, 3.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule
    output = scheduler.schedule()

    # Should schedule all requests in arrival time order
    assert len(output.scheduled_new_reqs) == 4
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]

    # Expected order by arrival time:
    # req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0)
    assert scheduled_req_ids == ["1", "3", "2", "0"]


def test_priority_scheduling_with_limited_slots():
    """Test priority scheduling when max_num_seqs limits concurrent requests."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=2,  # Only allow 2 concurrent requests
        max_num_batched_tokens=1000,  # Plenty of token budget
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the 2 highest priority requests
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2

    # Should schedule req_3 (priority 0) and req_1 (priority 1)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert "3" in scheduled_req_ids  # Priority 0
    assert "1" in scheduled_req_ids  # Priority 1

    # Remaining requests should be in waiting queue in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify order
    waiting_requests = list(scheduler.waiting)
    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_heap_property():
    """Test that the waiting queue maintains heap
    property for priority scheduling."""
    scheduler = create_scheduler(
        scheduling_policy="priority",
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Add requests in random priority order
    priorities = [5, 1, 8, 3, 2, 7, 4, 6]
    arrival_times = [float(i) for i in range(len(priorities))]
    requests = create_requests_with_priority(
        num_requests=len(priorities),
        priorities=priorities,
        arrival_times=arrival_times,
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule one request at a time and verify priority order
    scheduled_priorities = []

    while scheduler.waiting:
        output = scheduler.schedule()
        if output.scheduled_new_reqs:
            req = output.scheduled_new_reqs[0]
            scheduled_priorities.append(requests[int(req.req_id)].priority)

            # Simulate completion to make room for next request
            model_output = ModelRunnerOutput(
                req_ids=[req.req_id],
                req_id_to_index={req.req_id: 0},
                sampled_token_ids=[[100]],
                logprobs=None,
                prompt_logprobs_dict={},
            )
            scheduler.update_from_output(output, model_output)

            # Finish the request to make room for the next one
            scheduler.finish_requests(req.req_id, RequestStatus.FINISHED_STOPPED)

    # Verify requests were scheduled in priority order (lowest value first)
    expected_priorities = sorted(priorities)
    assert scheduled_priorities == expected_priorities


def test_schedule_skip_tokenizer_init():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    requests = create_requests(num_requests=5)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)


def test_schedule_skip_tokenizer_init_structured_output_request():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    structured_outputs_params = StructuredOutputsParams(regex="[0-9]+")
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=16,
        structured_outputs=structured_outputs_params,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1
    assert len(scheduler.deferred_waiting) == 1


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_compile_error_finishes_only_request(async_grammar: bool):
    scheduler = create_scheduler()
    manager = scheduler.structured_output_manager
    manager.backend = Mock()
    manager.backend.compile_grammar.side_effect = RuntimeError(
        "forced FSM compilation error"
    )
    manager._use_async_grammar_compilation = async_grammar
    if async_grammar:
        manager.executor = ThreadPoolExecutor(max_workers=1)

    sampling_params = SamplingParams(
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="grammar-error",
        prompt_token_ids=[0, 1],
        sampling_params=sampling_params,
    )

    manager.grammar_init(request)
    assert request.structured_output_request is not None
    grammar_future = request.structured_output_request._grammar
    assert isinstance(grammar_future, Future)
    assert isinstance(grammar_future.exception(timeout=5), RuntimeError)
    if async_grammar:
        manager.executor.shutdown()

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    output = engine_core_outputs[0].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR
    assert output.stop_reason is None

    healthy_request = create_requests(num_requests=1, req_ids=["healthy-request"])[0]
    scheduler.add_request(healthy_request)
    next_output = scheduler.schedule()
    assert [req.req_id for req in next_output.scheduled_new_reqs] == [
        healthy_request.request_id
    ]


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(Scheduler)
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
    )
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.accept_tokens.return_value = False
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = set()
    scheduler.encoder_cache_mismatch_reqs = set()
    scheduler.irwkv_config = Mock()
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req: Request, delay_free_blocks: bool = False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    scheduler.structured_output_manager.accept_tokens.assert_called_once_with(
        request, [123]
    )
    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running
    scheduler._free_request.assert_called_once_with(request)
    assert len(engine_core_outputs[0].outputs) == 1
    engine_core_output = engine_core_outputs[0].outputs[0]
    assert engine_core_output.request_id == request.request_id
    assert engine_core_output.new_token_ids == [123]
    assert engine_core_output.finish_reason == FinishReason.ERROR


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "is_encoder_decoder", "expect_enabled"),
    [
        (True, False, True),
        (False, False, False),
        # Encoder-decoder models should always have it disabled
        (False, True, False),
        (True, True, False),
    ],
)
def test_chunked_prefill_disabled_for_encoder_decoder(
    enable_chunked_prefill: bool, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate that chunked prefill is appropriately disabled for
    encoder-decoder models."""
    scheduler_config = SchedulerConfig(
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=is_encoder_decoder,
        # Must <= max_num_batched_tokens if chunked prefill is disabled
        max_model_len=SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )

    # `is_encoder_decoder` should only be used during construction
    # of the config, and otherwise stored in the model config.
    assert "is_encoder_decoder" not in vars(scheduler_config)
    assert "is_encoder_decoder" not in [
        f.name for f in dataclasses.fields(scheduler_config)
    ]
    _validate_chunked_prefill_settings_for_encoder_decoder(
        scheduler_config, is_encoder_decoder, expect_enabled
    )

    # Ensure it is retained in IrwkvConfig, even after its post-init.
    irwkv_config = IrwkvConfig(scheduler_config=scheduler_config)
    _validate_chunked_prefill_settings_for_encoder_decoder(
        irwkv_config.scheduler_config, is_encoder_decoder, expect_enabled
    )


def _validate_chunked_prefill_settings_for_encoder_decoder(
    scheduler_config: SchedulerConfig, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate chunked prefill settings in the scheduler config for
    encoder-decoder models."""
    assert scheduler_config.enable_chunked_prefill is expect_enabled
    if is_encoder_decoder:
        # Encoder-decoder models should automatically disable chunked multimodal
        # inputs as well
        assert scheduler_config.disable_chunked_mm_input is not expect_enabled
    if is_encoder_decoder and not expect_enabled:
        assert scheduler_config.long_prefill_token_threshold == 0


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests start
# NOTE: In E->P->D disagg case, both KV and EC Connector works in P instance
# Unless specify, the existence of KV Connector should not affect any test results
# ==============================================================================


def _assert_right_encoder_cache_allocated(
    scheduler: Scheduler,
    hashes_to_check: list[str] | None = None,
    requests: list[Request] | None = None,
    expected_total_allocated: int | None = None,
):
    """Check whether encoder cache is allocated correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager

    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, "Encoder cache manager should exist"

    # Verify number of cache
    if expected_total_allocated is not None:
        assert len(encoder_cache_manager.cached) == expected_total_allocated
        if expected_total_allocated == 0:
            return

    # Verify each request with MM data is in cache
    cached_hashes = set(encoder_cache_manager.cached.keys())

    if hashes_to_check:
        missed_hashes = set(hashes_to_check) - cached_hashes
        assert not missed_hashes, (
            f"Miss hashes: {missed_hashes} "
            f"Existing encoder cache: {encoder_cache_manager.cached}"
        )

    for req in requests if requests is not None else []:
        if req.mm_features:
            mm_hashes = [f.identifier for f in req.mm_features]
            req_hashes = set(mm_hashes)  # unique hashes set
            missed_hashes = req_hashes - cached_hashes
            assert not missed_hashes, (
                f"Miss hashes in cache for request {req.request_id}: {missed_hashes} "
                f"Existing encoder cache: {encoder_cache_manager.cached}"
            )


def _assert_right_ec_connector_metadata(
    output: SchedulerOutput,
    mm_features_list: list[MultiModalFeatureSpec],
):
    """Verify that ECConnector metadata EXACTLY matches the input MM data."""
    # Get the connector metadata
    metadata = output.ec_connector_metadata

    # Create lookup dictionaries for efficient access
    metadata_dict = {mm_data.mm_hash: mm_data for mm_data in metadata.mm_datas}

    # Check all required identifiers exist in metadata; and no extra
    # In ECExampleConnector format
    # NOTE: even having same identifier, the mm_features can be different
    # since their mm_position can be in different offsets, etc
    identifiers_dict = {f.identifier for f in mm_features_list}
    assert set(metadata_dict.keys()) == identifiers_dict

    # Verify the info matches
    for i, mm_feature in enumerate(mm_features_list):
        identifier = mm_feature.identifier
        assert metadata_dict[identifier].mm_hash == identifier
        assert metadata_dict[identifier].num_token == mm_feature.mm_position.length


def _assert_right_encoder_inputs(
    output: SchedulerOutput,
    check_exist: bool | None = True,
    requests: list[Request] | None = None,
    expected_encoder_inputs: list[list[int]] | None = None,
    expected_total_reqs: int | None = None,
):
    """Verify that requests/mm_hashes should (not) in scheduled encoder input
    If check_exist is False, this function returns True
    if requests are NOT in encoder inputs"""
    # Get the scheduled encoder inputs
    # NOTE: scheduled_encoder_inputs is a dictionary with request id as key
    scheduled_encoder_inputs = output.scheduled_encoder_inputs

    # Check if scheduled_encoder_inputs is empty as expected
    if expected_total_reqs is not None:
        assert len(scheduled_encoder_inputs) == expected_total_reqs
        if expected_total_reqs == 0:
            return

    # Number of expected encoder inputs should match number of requests
    if expected_encoder_inputs:
        assert check_exist and requests is not None  # only support expect input exist
        assert len(requests) == len(expected_encoder_inputs)

    # Check request (not) exist as expected
    for i, request in enumerate(requests if requests is not None else []):
        assert (request.request_id in scheduled_encoder_inputs) is check_exist, (
            f"Request {request.id} presence mismatch: expected {check_exist}, "
            f"got {request.id in scheduled_encoder_inputs}"
        )
        if expected_encoder_inputs:
            scheduled_encoder_input = scheduled_encoder_inputs[request.request_id]
            assert scheduled_encoder_input == expected_encoder_inputs[i]


def test_scheduler_no_ec_connector_by_default():
    """Test scheduler doesn't have EC connector by default."""
    scheduler = create_scheduler()
    assert scheduler.ec_connector is None


def test_mamba_align_encoder_cache_cap_makes_progress():
    """Two individually cacheable images must not deadlock Mamba alignment."""
    block_size = 768
    encoder_cache_size = 600
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        block_size=block_size,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.max_num_encoder_input_tokens = encoder_cache_size
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

    first_image_end = 510
    request = create_requests(
        num_requests=1,
        num_tokens=1010,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=500),
                PlaceholderRange(offset=first_image_end, length=500),
            ]
        ],
        max_tokens=1,
        block_size=block_size,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == first_image_end
    assert output.scheduled_encoder_inputs[request.request_id] == [0]

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )

    output = scheduler.schedule()
    next_block_boundary = block_size
    assert output.num_scheduled_tokens[request.request_id] == (
        next_block_boundary - first_image_end
    )
    assert output.scheduled_encoder_inputs[request.request_id] == [1]

    # Then MODEL_RUNNER will execute the encoder and cache the result


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests end
# ==============================================================================


def test_remote_kv_promotion_drains_before_grammar_prefix():
    scheduler = create_scheduler(max_num_seqs=1)
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)

    requests = create_requests(num_requests=4)
    req_grammar_1, req_grammar_2, req_remote, req_tail = requests

    # simulate two structured-output grammar requests at the waiting head
    # that become ready now.
    req_grammar_1.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_1.structured_output_request = Mock(grammar=object())
    req_grammar_2.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_2.structured_output_request = Mock(grammar=object())

    # simulate a remote-KV request that is ready to be promoted now. Parked
    # loads hold KV blocks, so they drain ahead of the grammar requests.
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    for request in requests:
        scheduler.add_request(request)
    assert list(scheduler.kv_holding_waiting) == [req_remote]
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    scheduler._update_waiting_for_remote_kv = Mock()

    output = scheduler.schedule()

    assert output.scheduled_new_reqs
    assert output.scheduled_new_reqs[0].req_id == req_remote.request_id
    waiting_req_ids = [req.request_id for req in scheduler.waiting]
    assert waiting_req_ids == [
        req_grammar_1.request_id,
        req_grammar_2.request_id,
        req_tail.request_id,
    ]


def test_fcfs_mixed_blocked_types_remote_load_drains_first():
    scheduler = create_scheduler(max_num_batched_tokens=20)
    scheduler._update_waiting_for_remote_kv = Mock()

    mk_req = lambda req_id, num_tokens=1: create_requests(  # noqa: E731
        num_requests=1, num_tokens=num_tokens, req_ids=[req_id]
    )[0]
    req_grammar, req_remote, req_stream = (
        mk_req("grammar"),
        mk_req("remote"),
        mk_req("stream"),
    )
    req_regular, req_tail = mk_req("regular", 20), mk_req("tail")
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar.structured_output_request = Mock(grammar=None)
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req_stream.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    for req in (req_grammar, req_remote, req_stream, req_regular, req_tail):
        scheduler.add_request(req)
    scheduler.schedule()
    assert list(scheduler.waiting) == [req_grammar, req_stream, req_tail]
    assert list(scheduler.kv_holding_waiting) == [req_remote]

    scheduler.finish_requests(req_regular.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.running

    req_grammar.structured_output_request = Mock(grammar=object())
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    req_stream.status = RequestStatus.WAITING

    second_output = scheduler.schedule()
    expected_order = [
        req_remote.request_id,
        req_grammar.request_id,
        req_stream.request_id,
        req_tail.request_id,
    ]
    assert [req.req_id for req in second_output.scheduled_new_reqs] == expected_order
    assert [req.request_id for req in scheduler.running] == expected_order
    scheduler._update_waiting_for_remote_kv.assert_called_once_with(req_remote)


def test_scheduler_kv_connector_stats():
    """Test worker-side, scheduler-side, and combined KV connector stats."""

    class GenericKVConnectorStats(KVConnectorStats):
        def reset(self):
            self.data = {}

        def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
            self.data.update(other.data)
            return self

        def reduce(self) -> dict[str, int | float]:
            return {}

        def is_empty(self) -> bool:
            return not self.data

    test_cases = (
        ({"worker": 1}, None, {"worker": 1}),
        (None, {"scheduler": 2}, {"scheduler": 2}),
        ({"worker": 1}, {"scheduler": 2}, {"worker": 1, "scheduler": 2}),
    )

    for worker_data, scheduler_data, expected_data in test_cases:
        scheduler = create_scheduler()
        worker_stats = (
            GenericKVConnectorStats(data=worker_data) if worker_data else None
        )
        scheduler_stats = (
            GenericKVConnectorStats(data=scheduler_data) if scheduler_data else None
        )
        scheduler.connector = Mock()
        scheduler.connector.get_kv_connector_stats.return_value = (
            scheduler_stats if worker_stats is None else None
        )
        scheduler.connector.take_events.return_value = []

        def update_connector_output(
            kv_connector_output: KVConnectorOutput,
            scheduler=scheduler,
            scheduler_stats=scheduler_stats,
        ):
            scheduler.connector.get_kv_connector_stats.return_value = scheduler_stats

        scheduler.connector.update_connector_output.side_effect = (
            update_connector_output
        )

        model_output = ModelRunnerOutput(
            req_ids=["req_0"],
            req_id_to_index={"req_0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            kv_connector_output=KVConnectorOutput(kv_connector_stats=worker_stats)
            if worker_stats
            else None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req_0": 1},
            total_num_scheduled_tokens=1,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        engine_core_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )

        final_stats = next(
            iter(engine_core_outputs.values())
        ).scheduler_stats.kv_connector_stats
        assert final_stats == expected_data


def test_scheduler_ec_connector_stats():
    """Test worker-side, scheduler-side, and combined EC connector stats."""

    class GenericECConnectorStats(ECConnectorStats):
        def reset(self):
            self.data = {}

        def aggregate(self, other: ECConnectorStats) -> ECConnectorStats:
            self.data.update(other.data)
            return self

        def reduce(self) -> dict[str, int | float]:
            return {}

        def is_empty(self) -> bool:
            return not self.data

    test_cases = (
        ({"worker": 1}, None, {"worker": 1}),
        (None, {"scheduler": 2}, {"scheduler": 2}),
        ({"worker": 1}, {"scheduler": 2}, {"worker": 1, "scheduler": 2}),
    )

    for worker_data, scheduler_data, expected_data in test_cases:
        scheduler = create_scheduler()
        worker_stats = (
            GenericECConnectorStats(data=worker_data) if worker_data else None
        )
        scheduler_stats = (
            GenericECConnectorStats(data=scheduler_data) if scheduler_data else None
        )
        scheduler.ec_connector = Mock()
        scheduler.ec_connector.take_unavailable_requests.return_value = set()
        scheduler.ec_connector.get_ec_connector_stats.return_value = (
            scheduler_stats if worker_stats is None else None
        )

        def update_connector_output(
            ec_connector_output: ECConnectorOutput,
            scheduler=scheduler,
            scheduler_stats=scheduler_stats,
        ):
            scheduler.ec_connector.get_ec_connector_stats.return_value = scheduler_stats

        scheduler.ec_connector.update_connector_output.side_effect = (
            update_connector_output
        )

        model_output = ModelRunnerOutput(
            req_ids=["req_0"],
            req_id_to_index={"req_0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            ec_connector_output=ECConnectorOutput(ec_connector_stats=worker_stats)
            if worker_stats
            else None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req_0": 1},
            total_num_scheduled_tokens=1,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        engine_core_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )

        final_stats = next(
            iter(engine_core_outputs.values())
        ).scheduler_stats.ec_connector_stats
        assert final_stats == expected_data


# ==============================================================================
# Variable-length encoder cross-attention block allocation tests
# ==============================================================================


def _get_num_cross_attn_blocks(scheduler: Scheduler, request_id: str) -> int:
    """Get the number of cross-attention blocks allocated for a request."""
    from irwkv.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    coordinator = scheduler.kv_cache_manager.coordinator
    for manager in coordinator.single_type_managers:
        if isinstance(manager, CrossAttentionManager):
            blocks = manager.req_to_blocks.get(request_id, [])
            return len(blocks)
    raise AssertionError("No CrossAttentionManager found in coordinator")


def test_ec_connector_pending_prefetch_only_checks_future_mm_features():
    """Test that future mm feature filtering only yields features beyond
    the computed token frontier.

    Features already within num_computed_tokens (past/boundary) must be
    filtered out; only features that extend beyond the frontier (future) should
    be yielded so that connector implementations know which items to prefetch.

    Filter cases:
      "past":     end = 0  + 16 = 16 <  32 → filtered OUT
      "boundary": end = 16 + 16 = 32 == 32 → filtered OUT (condition is >, not >=)
      "future":   end = 48 + 32 = 80 >  32 → yielded
    """
    BLOCK_SIZE = 16
    NUM_COMPUTED_TOKENS = BLOCK_SIZE * 2  # 32
    NUM_TOKENS = BLOCK_SIZE * 8  # 128

    HASH_PAST = "hash_past"
    HASH_BOUNDARY = "hash_boundary"
    HASH_FUTURE = "hash_future"

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[[HASH_PAST, HASH_BOUNDARY, HASH_FUTURE]],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=BLOCK_SIZE),  # end=16 (past)
                PlaceholderRange(offset=16, length=BLOCK_SIZE),  # end=32 (boundary)
                PlaceholderRange(offset=48, length=BLOCK_SIZE * 2),  # end=80 (future)
            ]
        ],
        block_size=BLOCK_SIZE,
    )[0]

    future_hashes = [
        f.identifier
        for f in request.mm_features
        if f.mm_position.offset + f.mm_position.length > NUM_COMPUTED_TOKENS
    ]

    assert future_hashes == [HASH_FUTURE], (
        f"Expected only {HASH_FUTURE!r} from future mm feature filtering, "
        f"got {future_hashes!r}. Past/boundary features must be filtered out."
    )


def _make_encoder_instance_request(scheduler, text_prefix=8, image_tokens=16):
    """One request whose image sits after some text, as an EPD proxy sends it.

    `create_scheduler` builds a text-only model, so `compute_mm_encoder_budget`
    leaves the encoder cache at zero capacity; give it room so the admitted path
    is reachable.
    """
    ecm = scheduler.encoder_cache_manager
    ecm.cache_size = ecm.num_free_slots = ecm.num_freeable_slots = 8 * image_tokens
    scheduler.max_num_encoder_input_tokens = 8 * image_tokens

    (request,) = create_requests(
        num_requests=1,
        num_tokens=text_prefix + image_tokens + 4,
        # Completion must hinge on the prompt being encoded, not on max_tokens.
        max_tokens=16,
        mm_hashes_list=[["img-hash-0"]],
        mm_positions=[[PlaceholderRange(offset=text_prefix, length=image_tokens)]],
    )
    scheduler.add_request(request)
    return request


class _RecordingGrammar:
    """Records what reaches ``validate_tokens`` and raises on a negative id.

    llguidance raises ``OverflowError`` on a negative token id, so this mirrors
    the real failure: the tests below fail on the unpatched scheduler (the -1
    padding reaches ``validate_tokens``) and pass once it is stripped first.
    """

    def __init__(self) -> None:
        self.seen: list[list[int]] = []

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        if any(t < 0 for t in tokens):
            raise OverflowError("llguidance rejects negative token ids")
        self.seen.append(list(tokens))
        return tokens


def _decode_ready_request(scheduler):
    """Add one request and advance it out of prefill into decode."""
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
        ),
    )
    return request
