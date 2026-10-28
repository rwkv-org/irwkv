# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from irwkv import SamplingParams
from irwkv.config import IrwkvConfig
from irwkv.engine.arg_utils import EngineArgs
from irwkv.platforms import current_platform
from irwkv.utils.torch_utils import set_default_torch_num_threads
from irwkv.v1.core.sched.interface import PauseState
from irwkv.v1.engine import EngineCoreRequest
from irwkv.v1.engine.core import EngineCore, EngineCoreProc
from irwkv.v1.executor.abstract import Executor
from irwkv.v1.executor.uniproc_executor import UniProcExecutor
from irwkv.v1.kv_cache_interface import KVCacheConfig
from irwkv.v1.outputs import ModelRunnerOutput

from ...utils import create_new_process_for_each_test, multi_gpu_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL_NAME = os.environ.get("IRWKV_RWKV_TEST_MODEL", "rwkv-rs/rwkv7-g1-st")
PROMPT_TOKENS = [326] * 12

_REQUEST_COUNTER = 0


def make_request() -> EngineCoreRequest:
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    request_id = f"request-{_REQUEST_COUNTER}"
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-{uuid.uuid4()}",
        prompt_token_ids=PROMPT_TOKENS,
        mm_features=None,
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
        cache_salt=None,
        data_parallel_rank=None,
    )


@create_new_process_for_each_test()
def test_engine_core():
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME, gpu_memory_utilization=0.5)
    irwkv_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(irwkv_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            irwkv_config=irwkv_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""

    # First request.
    engine_core.add_request(engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # Second request.
    engine_core.add_request(engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 1

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Add two requests in a row.
    engine_core.add_request(engine_core.preprocess_add_request(make_request()))
    engine_core.add_request(engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 4

    # Loop through until they are all done.
    while (outs := engine_core.step_fn()[0].get(0)) and outs.outputs:
        pass

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    """Test abort cycle."""

    # Basic abort.
    req = make_request()
    request_id = req.request_id

    engine_core.add_request(engine_core.preprocess_add_request(req))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    engine_core.abort_requests([request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    assert not engine_core.scheduler.has_unfinished_requests()
    assert engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert not engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    # Add, step, abort 1 of the 3.
    req0 = make_request()
    req1 = make_request()
    req2 = make_request()

    engine_core.add_request(engine_core.preprocess_add_request(req0))
    engine_core.add_request(engine_core.preprocess_add_request(req1))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    engine_core.add_request(engine_core.preprocess_add_request(req2))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 3

    # Abort just one.
    engine_core.abort_requests([req1.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Abort the other requests at the same time.
    engine_core.abort_requests([req2.request_id, req0.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0

    # Sending duplicate requests with same request_id
    req0 = make_request()
    req1 = make_request()
    req0.request_id = req1.request_id = "test"
    engine_core.add_request(engine_core.preprocess_add_request(req0))

    while engine_core.scheduler.has_requests():
        engine_core.step_fn()

    engine_core.add_request(engine_core.preprocess_add_request(req1))
    while engine_core.scheduler.has_requests():
        engine_core.step_fn()

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
def test_engine_core_advanced_sampling():
    """A basic end-to-end test to verify that the engine functions correctly
    when additional sampling parameters, such as top_p, min_tokens, and
    presence_penalty, are set.
    """
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME, gpu_memory_utilization=0.5)
    irwkv_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(irwkv_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            irwkv_config=irwkv_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""
    # First request.
    request: EngineCoreRequest = make_request()
    request.sampling_params = SamplingParams(
        min_tokens=4,
        presence_penalty=1.0,
        frequency_penalty=1.0,
        repetition_penalty=0.1,
        stop_token_ids=[1001, 1002],
    )
    engine_core.add_request(engine_core.preprocess_add_request(request))

    def _check_engine_state():
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        # Loop through until they are all done.
        while engine_core.scheduler.has_requests():
            engine_core.step_fn()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0

    _check_engine_state()

    # Second request.
    request2 = make_request()
    request2.sampling_params = SamplingParams(
        top_p=0.99,
        top_k=50,
    )
    engine_core.add_request(engine_core.preprocess_add_request(request2))
    _check_engine_state()


@create_new_process_for_each_test()
def test_engine_core_concurrent_batches():
    """Test that the engine can handle multiple concurrent batches."""

    def make_request_with_max_tokens(req_id: str, max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    class DummyExecutor(UniProcExecutor):
        def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
            super().initialize_from_config(kv_cache_configs)

            # Create a thread pool with a single worker
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        def execute_model(
            self,
            scheduler_output,
            non_block=False,
        ) -> Future[ModelRunnerOutput | None]:
            """Make execute_model non-blocking."""
            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("execute_model", args=(scheduler_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        def sample_tokens(
            self, grammar_output, non_block=False
        ) -> Future[ModelRunnerOutput]:
            """Make sample_tokens non-blocking."""
            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("sample_tokens", args=(grammar_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        def shutdown(self):
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

    engine_args = EngineArgs(
        model=MODEL_NAME,
        gpu_memory_utilization=0.5,
        # To test concurrent batches.
        max_num_seqs=2,
        # Avoid all requests being scheduled once.
        enable_prefix_caching=False,
        max_num_batched_tokens=10,
        # Reduce startup time.
        enforce_eager=True,
        # Test concurrent batch behaviour independently of async scheduling.
        async_scheduling=False,
    )
    irwkv_config = engine_args.create_engine_config()
    # Force two concurrent batches to exercise the batch queue independently
    # of async scheduling (which is disabled above).
    with (
        set_default_torch_num_threads(1),
        patch.object(
            IrwkvConfig,
            "max_concurrent_batches",
            new_callable=PropertyMock,
            return_value=2,
        ),
    ):
        engine_core = EngineCore(
            irwkv_config=irwkv_config, log_stats=False, executor_class=DummyExecutor
        )
    assert engine_core.batch_queue is not None

    # Add two requests in a row. Each request have 12 prompt tokens.
    req0 = make_request_with_max_tokens("0", 5)
    engine_core.add_request(engine_core.preprocess_add_request(req0))
    req1 = make_request_with_max_tokens("1", 5)
    engine_core.add_request(engine_core.preprocess_add_request(req1))

    # Schedule Batch 1: (10, req0)
    assert engine_core.step_with_batch_queue()[0] is None
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 10
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 10

    # Schedule Batch 2: (2, req0), (8, req1)
    assert engine_core.step_with_batch_queue()[0] == {}
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 2
    assert scheduler_output.num_scheduled_tokens["1"] == 8
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests["0"].num_computed_tokens == 12
    assert engine_core.scheduler.requests["1"].num_computed_tokens == 8

    assert engine_core.scheduler.get_num_unfinished_requests() == 2

    # Finish Batch 1 and schedule Batch 3: (4, req1).
    # Note that req0 cannot be scheduled
    # because it is in the decoding stage now.
    engine_core.step_with_batch_queue()
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 4

    # Finish Batch 2. Get first token of req0.
    # Schedule Batch 4: (1, req0).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 1

    # Finish Batch 3. Get first token of req1. Schedule Batch 5: (1, req1).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 1

    # Loop until req0 is finished.
    req_id = 0
    expected_num_tokens = [
        engine_core.scheduler.requests["0"].num_tokens + 1,
        engine_core.scheduler.requests["1"].num_tokens + 1,
    ]
    while engine_core.scheduler.get_num_unfinished_requests() == 2:
        output = engine_core.step_with_batch_queue()[0]
        # Every step consumes an output.
        assert output is not None
        assert len(output[0].outputs) == 1
        if req_id in engine_core.scheduler.requests:
            assert (
                engine_core.scheduler.requests[req_id].num_tokens
                == expected_num_tokens[req_id]
            )
        expected_num_tokens[req_id] += 1
        req_id = (req_id + 1) % 2


@multi_gpu_test(num_gpus=2)
def test_engine_core_tp():
    """Test engine can initialize worker in tp properly."""
    """Setup the EngineCore."""
    engine_args = EngineArgs(
        model=MODEL_NAME,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=2,
        # Reduce startup time.
        enforce_eager=True,
    )
    irwkv_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(irwkv_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            irwkv_config=irwkv_config, executor_class=executor_class, log_stats=True
        )

    def get_worker_cache_config_field(worker, key: str):
        return getattr(worker.cache_config, key)

    num_gpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_gpu_blocks",)
    )
    num_cpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_cpu_blocks",)
    )
    assert all(x is not None for x in num_gpu_blocks)
    assert all(x is not None for x in num_cpu_blocks)


@pytest.mark.parametrize("request_id", [uuid.uuid4(), 12345, None])
def test_engine_core_invalid_request_id_type(request_id):
    """Reject non-string IDs before admitting a request to the scheduler."""
    from irwkv.v1.request import Request

    core = object.__new__(EngineCore)
    core.scheduler = MagicMock()
    request = Request.from_engine_core_request(make_request(), None)
    request.request_id = request_id
    with pytest.raises(TypeError, match="request_id must be a string"):
        core.add_request(request)
    core.scheduler.add_request.assert_not_called()

    request.request_id = "valid"
    core.add_request(request)
    core.scheduler.add_request.assert_called_once_with(request)


def _pausable_engine_core_proc() -> EngineCoreProc:
    """A bare EngineCoreProc holding just the state pause_scheduler touches."""
    core = object.__new__(EngineCoreProc)
    core.model_executor = MagicMock()
    core.scheduler = MagicMock()
    core.scheduler.has_requests.return_value = False
    core.batch_queue = None
    core._idle_state_callbacks = []
    return core


@pytest.mark.parametrize(
    "pause_state,has_requests,has_batches",
    [
        pytest.param(PauseState.UNPAUSED, False, False, id="not-paused"),
        pytest.param(PauseState.PAUSED_ALL, True, False, id="pending-requests"),
        pytest.param(PauseState.PAUSED_ALL, False, True, id="pending-batches"),
    ],
)
def test_kv_cache_release_rejects_unsafe_state(pause_state, has_requests, has_batches):
    """Reject release before touching caches or memory if work can still use KV."""
    core = _pausable_engine_core_proc()
    core.scheduler.pause_state = pause_state
    core.scheduler.has_requests.return_value = has_requests
    core.batch_queue = [object()] if has_batches else None
    core._reset_caches = MagicMock()

    with pytest.raises(RuntimeError, match="requires a completed pause"):
        core.release_kv_cache_memory()

    core._reset_caches.assert_not_called()
    core.model_executor.discard.assert_not_called()


@pytest.mark.parametrize("deferred", [False, True])
def test_pause_synchronizes_device_before_cache_reset(deferred: bool):
    """A resolved pause promises an idle device: the barrier must run before
    caches are cleared and before the caller is unblocked."""
    core = _pausable_engine_core_proc()
    core.batch_queue = [object()] if deferred else None
    order: list[str] = []
    core.model_executor.collective_rpc.side_effect = lambda method: order.append(method)
    core._reset_caches = lambda: order.append("reset_caches")

    result = EngineCoreProc.pause_scheduler(core, mode="keep", clear_cache=True)
    if deferred:
        assert isinstance(result, Future)
        assert not result.done() and order == []
        core.batch_queue = None
        core._notify_idle_state_callbacks()
        assert result.result(timeout=0) is None
    else:
        assert result is None
    assert order == ["synchronize_device", "reset_caches"]
