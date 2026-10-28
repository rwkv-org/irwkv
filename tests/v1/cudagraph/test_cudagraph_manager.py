# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from irwkv.config import (
    CompilationConfig,
    CUDAGraphMode,
    IrwkvConfig,
    ParallelConfig,
    SchedulerConfig,
)
from irwkv.distributed.device_communicators import pynccl_allocator
from irwkv.v1.kv_cache_interface import KVCacheConfig
from irwkv.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from irwkv.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from irwkv.v1.worker.gpu.input_batch import InputBuffers
from irwkv.v1.worker.gpu.pcp_manager import PCPManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_irwkv_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    irwkv_config = MagicMock(spec=IrwkvConfig)
    irwkv_config.compilation_config = compilation_config
    irwkv_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    irwkv_config.parallel_config = ParallelConfig()
    irwkv_config.use_cumem_cudagraph_pool = False
    return irwkv_config


def test_full_capture_sets_graph_pool_id_before_cuda_graph(monkeypatch):
    """FULL capture must set graph_pool_id before entering torch.cuda.graph().

    NCCL symmetric memory checks this global during graph capture; without
    it, capture fails with:
    AssertionError: graph_pool_id is not set under graph capture
    """
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    manager = gpu_cudagraph_utils.CudaGraphManager(
        irwkv_config=_create_irwkv_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._capture_descs[CUDAGraphMode.FULL] = [desc]

    def create_forward_fn(desc, warmup):
        return lambda _mode: None

    @contextmanager
    def fake_graph_capture(*args, **kwargs):
        yield SimpleNamespace(stream=MagicMock())

    fake_offloader = MagicMock()

    def cuda_graph_enter(*args, **kwargs):
        assert pynccl_allocator._graph_pool_id is graph_pool

    mock_cuda_graph_ctx = MagicMock()
    mock_cuda_graph_ctx.__enter__ = cuda_graph_enter
    mock_cuda_graph_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(gpu_cudagraph_utils, "graph_capture", fake_graph_capture),
        patch.object(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader),
        patch.object(gpu_cudagraph_utils.torch.cuda, "CUDAGraph"),
        patch.object(
            gpu_cudagraph_utils.torch.cuda,
            "graph",
            return_value=mock_cuda_graph_ctx,
        ) as mock_cuda_graph,
    ):
        manager.capture(create_forward_fn)

    mock_cuda_graph.assert_called_once()


def test_query_sensitive_full_graphs_separate_decode_and_prefill(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        irwkv_config=_create_irwkv_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
        requires_max_query_len=True,
    )
    manager._graphs_captured = True

    decode = manager.dispatch(
        num_reqs=2,
        num_tokens=2,
        uniform_token_count=1,
        max_query_len=1,
    )
    prefill = manager.dispatch(
        num_reqs=1,
        num_tokens=2,
        uniform_token_count=None,
        max_query_len=2,
    )

    assert decode.max_query_len == 1
    assert decode.uniform_token_count == 1
    assert prefill.max_query_len == 4
    assert prefill.uniform_token_count is None


@pytest.mark.parametrize("num_reqs,num_tokens", [(1, 2), (2, 3)])
def test_rwkv_runner_preserves_prefill_query_bound(monkeypatch, num_reqs, num_tokens):
    from irwkv.v1.worker.gpu import model_runner

    runner = MagicMock(spec=model_runner.GPUModelRunner)
    runner.model_state = SimpleNamespace(requires_cudagraph_max_query_len=True)
    runner.gather_batch_req_state.return_value = (
        SimpleNamespace(num_tokens=num_tokens, has_prefill=True),
        None,
    )
    runner.pcp_manager = runner.ubatch_runner = None
    runner.cudagraph_manager = runner.parallel_config = MagicMock()
    runner.is_encoder_decoder = False
    runner.dp_size, runner.dp_rank, runner.decode_query_len = 1, 0, 1
    observed = []

    def dispatch(*args, **kwargs):
        observed.append(kwargs["max_query_len"])
        raise RuntimeError("stop after graph dispatch")

    monkeypatch.setattr(model_runner, "dispatch_cg_and_sync_dp", dispatch)
    scheduled = {str(i): 2 if i == 0 else 1 for i in range(num_reqs)}
    with pytest.raises(RuntimeError, match="stop after graph dispatch"):
        model_runner.GPUModelRunner.execute_model(
            runner,
            SimpleNamespace(
                num_scheduled_tokens=scheduled, total_num_scheduled_tokens=num_tokens
            ),
            dummy_run=True,
        )
    assert observed == [2]


def test_piecewise_capture_uses_pcp_dummy_slot_mappings():
    num_reqs = 32
    num_tokens = 56
    pcp_world_size = 2
    input_buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))

    pcp_block_tables = SimpleNamespace(
        num_kv_cache_groups=1,
        input_block_tables=(torch.zeros(num_reqs * 2, 1, dtype=torch.int32),),
    )
    pcp_manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=num_reqs,
        max_num_tokens=num_tokens,
        block_tables=pcp_block_tables,
    )

    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.zeros(
        1, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    model_state.prepare_attn.return_value = {}
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        input_buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=False,
        pcp_manager=pcp_manager,
    )

    slot_mappings = model_state.prepare_attn.call_args.args[3]
    assert slot_mappings.shape == (1, num_tokens * pcp_world_size)
    block_tables.get_dummy_slot_mappings.assert_not_called()


@pytest.mark.parametrize(
    "full_cudagraph,max_query_len,expected_max_query_len",
    [(False, None, None), (True, None, 8), (True, 1, 1), (True, 3, 3)],
)
def test_capture_query_length_covers_replay_batches(
    full_cudagraph, max_query_len, expected_max_query_len
):
    """Mixed FULL graphs must capture prefill kernels as well as decode kernels."""
    num_tokens = num_reqs = 8
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.empty(
        0, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=full_cudagraph,
        max_query_len=max_query_len,
    )

    batch = model_state.prepare_attn.call_args.args[0]
    assert batch.num_scheduled_tokens.max() == 1
    assert batch.max_query_len == expected_max_query_len
