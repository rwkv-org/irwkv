# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

import irwkv.envs as envs
from irwkv.config import CUDAGraphMode, IrwkvConfig, ParallelConfig
from irwkv.logger import init_logger
from irwkv.platforms import current_platform
from irwkv.v1.attention.backend import AttentionMetadata
from irwkv.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)

track_batchsize: bool = envs.IRWKV_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.IRWKV_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass(frozen=True)
class BatchDescriptor:
    """Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    pcp_size: int = 1,
    use_ep: bool = True,
) -> list[int]:
    if pcp_size > 1:
        num_tokens_across_dp_cpu = (
            num_tokens_across_dp_cpu.repeat_interleave(pcp_size)
            if use_ep
            else num_tokens_across_dp_cpu * pcp_size
        )
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()


@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor

    # NOTE: local_sizes should only be set by the chunked_sizes context manager
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        return DPMetadata(num_tokens_across_dp_cpu)

    @contextmanager
    def sp_local_sizes(
        self, sequence_parallel_size: int, pcp_size: int = 1, use_ep: bool = False
    ):
        """Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size, pcp_size, use_ep
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.
    # When sp_size==1, this is just the cumulative num tokens across DP.
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:
    # copy from irwkv_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one
    for each microbatch.
    Set dynamically for each forward pass
    """
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None

    # Boolean mask over the token axis: True for padding rows that are not real
    # tokens. Consumers can use it to skip work for padded tokens. None when
    # the producer does not set it.
    is_padding: torch.Tensor | None = None

    # If True, bypass the compiled model call, e.g. by using .forward() directly
    skip_compiled: bool = False

    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.cudagraph_runtime_mode.is_valid_runtime_mode(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def in_piecewise_cudagraph() -> bool:
    """Whether the current forward runs in piecewise cudagraph mode (graph
    segments separated by eager breaks), at capture or replay time."""
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def create_forward_context(
    attn_metadata: Any,
    irwkv_config: IrwkvConfig,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    return ForwardContext(
        no_compile_layers=irwkv_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
        skip_compiled=skip_compiled,
        additional_kwargs=additional_kwargs or {},
        is_padding=is_padding,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    irwkv_config: IrwkvConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
    is_padding: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata = None
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    additional_kwargs = current_platform.set_additional_forward_context(
        attn_metadata=attn_metadata,
        irwkv_config=irwkv_config,
        dp_metadata=dp_metadata,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )
    forward_context = create_forward_context(
        attn_metadata,
        irwkv_config,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
        slot_mapping,
        additional_kwargs,
        skip_compiled,
        is_padding=is_padding,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
