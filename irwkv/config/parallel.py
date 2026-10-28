# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import regex as re
import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp, Store

import irwkv.envs as envs
from irwkv.config.utils import config
from irwkv.logger import init_logger
from irwkv.platforms import current_platform
from irwkv.utils.network_utils import get_open_ports_list

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from irwkv.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)
_NUMACTL_CPUSET_PATTERN = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")

DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
DCPCommBackend = Literal["ag_rs", "a2a"]


@config
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = Field(default=1, ge=1)
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = Field(default=1, ge=1)
    """Number of tensor parallel groups."""
    prefill_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that split prefill sequence computation. PCP expands
    the process world size but does not increase the KV-cache shard count."""
    data_parallel_size: int = Field(default=1, ge=1)
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor, prefill-context, and data parallel sizes."""
    data_parallel_size_local: int = Field(default=1, ge=0)
    """Number of local data parallel groups. A value of 0 is a sentinel used by
    the engine-args layer to signal that data parallelism was specified
    externally (see `ParallelConfig.__post_init__`)."""
    data_parallel_rank: int = Field(default=0, ge=0)
    """Rank of the data parallel group. The runtime check at
    ``__post_init__`` further bounds this by ``data_parallel_size``."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group, set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = Field(default=29550, ge=1, le=65535)
    """Fixed port for data parallel messaging, shared by all nodes."""
    dp_sync_interval: int = Field(default=16, ge=1)
    """Steps between DP finish-sync all-reduces; must match across DP ranks."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Supported only for MoE deployments; non-MoE
    models should use independent irwkv instances without --data-parallel-*
    arguments. Set implicitly when --data-parallel-rank is provided explicitly
    to irwkv serve."""
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where irwkv load balances
    between local data parallel ranks, but an external LB balances
    between irwkv nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank."""
    enable_batch_sharded_sampling: bool | None = None
    """Use sharded sampling across tensor parallel ranks. Each rank samples
    a slice of the batch instead of every rank sampling all of it. Currently
    defaults to False if not set. Enabling it explicitly raises when the config
    cannot support it (`tensor_parallel_size` must be > 1, `max_num_seqs` at
    least `tensor_parallel_size`, and `max_logprobs` non-negative). Models opt in
    by implementing `compute_logits_local`."""

    max_parallel_loading_workers: int | None = Field(default=None, ge=1)
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    enable_shm_tensor_arena: bool = False
    """Route large CPU tensors (e.g. multimodal ``pixel_values``) in the
    engine→worker broadcast through a zero-copy shared-memory arena instead of
    sending a copy to every reader. **Experimental**: may be reconciled with
    irwkv's existing multimodal shm tensor caching in a future release. Opt-in;
    most beneficial for multimodal serving with large images and tensor
    parallelism. Active only when all queue readers are node-local; reserves
    slots in ``/dev/shm``."""

    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    ubatch_size: int = Field(default=0, ge=0)
    """Number of ubatch size."""

    dbo_decode_token_threshold: int = Field(default=32, ge=0)
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch."""
    dbo_prefill_token_threshold: int = Field(default=512, ge=0)  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch."""

    disable_nccl_for_dp_synchronization: bool | None = None
    """Forces the dp synchronization logic in irwkv/v1/worker/dp_utils.py
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    """

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""

    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """
    Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    Note:
        [TPU](https://docs.vllm.ai/projects/tpu/en/latest/) platform only supports Ray
        for distributed inference.
    """

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed
    inference when distributed_executor_backend is mp."""
    master_port: int = 29501
    """distributed master port for multi-node distributed
    inference when distributed_executor_backend is mp."""
    node_rank: int = Field(default=0, ge=0)
    """distributed node rank for multi-node distributed
    inference when distributed_executor_backend is mp."""
    nnodes: int = Field(default=1, ge=1)
    """num of nodes for multi-node distributed
    inference when distributed_executor_backend is mp."""
    numa_bind: bool = False
    """Enable NUMA binding for GPU worker subprocesses.

    By default, workers are pinned to their GPU's NUMA-local CPUs and
    memory; on PCT-capable Xeons they also auto-bind to the SKU's
    PCT priority cores.
    """
    numa_bind_nodes: list[int] | None = None
    """NUMA node to bind each GPU worker to.

    Specify one NUMA node per visible GPU, for example `[0, 0, 1, 1]`
    for a 4-GPU system with GPUs 0-1 on NUMA node 0 and GPUs 2-3 on
    NUMA node 1. If unset and `numa_bind=True`, irwkv auto-detects the
    GPU-to-NUMA topology. The values are passed to `numactl --membind`
    and `--cpunodebind`, so they must be valid `numactl` NUMA node indices.
    """
    numa_bind_cpus: list[str] | None = None
    """Optional CPU lists to bind each GPU worker to.

    Specify one CPU list per visible GPU, for example
    `["0-3", "4-7", "8-11", "12-15"]`. When set, irwkv uses
    `numactl --physcpubind` instead of `--cpunodebind`. This is useful
    for custom policies such as binding to PCT or other high-frequency cores.
    Each entry must use `numactl --physcpubind` CPU-list syntax, for example
    `"0-3"` or `"0,2,4-7"`.
    """
    assigned_physical_gpu_ids: list[int] | None = None
    """Mapping from irwkv-local logical GPU IDs to physical GPU IDs.

    For example, ``[2, 3]`` means logical GPU 0 maps to physical GPU 2,
    and logical GPU 1 maps to physical GPU 3. Physical IDs are used only
    at platform/topology boundaries such as NVML, NIC affinity, P2P
    checks, and final CUDA device selection when needed. When None,
    logical IDs map to visible device IDs in order."""

    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group).
    If set, this value is passed to torch.distributed.init_process_group as the
    timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL).
    Increase this for multi-node setups where model downloads may be slow."""

    cpu_distributed_timeout_seconds: int | None = None
    """Timeout (in seconds) for cpu communication groups. If None, PyTorch's
    default timeout is used (1800s for gloo)."""

    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    _coord_store_port: int = 0
    """Port of the coordination TCPStore. Can be set by the API server; workers
    connect as clients to exchange self-picked group ports at runtime."""

    decode_context_parallel_size: int = Field(default=1, ge=1)
    """Number of ranks that shard the decode KV cache. DCP does not expand
    the process world size. Without PCP, DCP reuses TP ranks. With PCP, DCP
    either spans the PCP axis or the full TP x PCP block."""

    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.

    """
    dcp_comm_backend: DCPCommBackend | None = None
    """Communication backend for Decode Context Parallel (DCP).
    - "ag_rs": AllGather + ReduceScatter (existing behavior)
    - "a2a": All-to-All exchange of partial outputs + LSE, then
      combine with Triton kernel. Reduces NCCL calls from 3 to 2
      per layer for MLA models.

    `None` selects the model default, which is "ag_rs" unless the model
    overrides it via [`set_dcp_defaults`][irwkv.config.ParallelConfig.set_dcp_defaults].
    """

    dcp_q_replicate: bool | None = None
    """Replicate the MLA query projection within each DCP group so decode can skip the
    query all-gather.

    With DCP the KV cache is sharded across the group, so the standard MLA decode path
    all-gathers the query every step. Replicating the (small) query projection at load
    time lets each rank materialize the full group-local head set and skip that
    collective, at the cost of computing the projection redundantly on every rank
    in the group.
    """

    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP.
    Store interleave_size tokens on dcp_rank i, then store next
    interleave_size tokens on dcp_rank i+1.
    Interleave_size=1: token-level alignment, where token `i` is stored on
        dcp_rank `i % dcp_world_size`.
    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks. Tokens are then stored
        in (rank i+1, block j) only after (rank i, block j) is fully occupied.
    Block_size should be greater than or equal to cp_kv_cache_interleave_size.
    Block_size should be divisible by cp_kv_cache_interleave_size.

    When --cp-kv-cache-interleave-size is omitted (None), the interleave size
    is resolved automatically based on NIXL transfer requirements.
    Explicit settings take priority.
    """

    _allow_auto_resolve_cp_interleave_size: bool = True
    """Whether NIXL may select the interleave size automatically."""

    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""

    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    @field_validator("numa_bind_nodes")
    @classmethod
    def _validate_numa_bind_nodes(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_nodes must not be empty.")
        if any(node < 0 for node in value):
            raise ValueError("numa_bind_nodes must contain non-negative integers.")
        return value

    @field_validator("numa_bind_cpus")
    @classmethod
    def _validate_numa_bind_cpus(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("numa_bind_cpus must not be empty.")

        for cpuset in value:
            if not cpuset:
                raise ValueError("numa_bind_cpus entries must not be empty.")
            if not _NUMACTL_CPUSET_PATTERN.fullmatch(cpuset):
                raise ValueError(
                    "numa_bind_cpus entries must use numactl CPU list syntax, "
                    "for example '0-3' or '0,2,4-7'."
                )
            for part in cpuset.split(","):
                if "-" not in part:
                    continue
                start_str, end_str = part.split("-", 1)
                if int(start_str) > int(end_str):
                    raise ValueError(
                        f"numa_bind_cpus ranges must be ascending, but got '{cpuset}'."
                    )
        return value

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if not self.numa_bind and (
            self.numa_bind_nodes is not None or self.numa_bind_cpus is not None
        ):
            raise ValueError(
                "numa_bind_nodes and numa_bind_cpus require numa_bind=True."
            )

        tp = self.tensor_parallel_size
        pcp = self.prefill_context_parallel_size
        dcp = self.decode_context_parallel_size
        if pcp == 1:
            # DCP reuses the TP ranks when PCP is disabled.
            if tp % dcp != 0:
                raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")
        elif dcp not in (1, pcp, tp * pcp):
            raise ValueError(
                "When PCP is enabled, DCP must be disabled, span the PCP "
                "axis, or span the full TP x PCP axis. "
                f"Got TP={tp}, PCP={pcp}, DCP={dcp}; valid DCP sizes are "
                f"{sorted({1, pcp, tp * pcp})}."
            )

        return self

    def set_dcp_defaults(
        self,
        comm_backend: DCPCommBackend = "ag_rs",
        q_replicate: bool = False,
    ) -> None:
        """Fill in the DCP options the user left unset.

        Models can set their preferred DCP settings by calling this from their
        `verify_and_update_config` hook.
        """
        if self.dcp_comm_backend is None:
            self.dcp_comm_backend = comm_backend
        if self.dcp_q_replicate is None:
            self.dcp_q_replicate = q_replicate

    @property
    def world_size_across_dp(self) -> int:
        """Process world size across TP, PCP, PP, and DP."""
        return self.world_size * self.data_parallel_size

    @property
    def pcp_shard_decode_requests(self) -> bool:
        """Whether PCP can shard decode requests across its ranks.

        PCP-only execution replicates the KV cache, so each decode request can
        have a single PCP owner. DCP shards the KV cache and therefore requires
        every decode request to run on every participating DCP rank.
        """
        return (
            self.prefill_context_parallel_size > 1
            and self.decode_context_parallel_size == 1
        )

    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else self.ubatch_size

    @property
    def local_engines_only(self) -> bool:
        """Client manages local+remote EngineCores in pure internal LB case.
        Client manages local EngineCores in hybrid and external LB case.
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    @property
    def cpu_distributed_timeout(self) -> timedelta | None:
        seconds = self.cpu_distributed_timeout_seconds
        return timedelta(seconds=seconds) if seconds is not None else None

    def get_next_dp_init_port(self) -> int:
        """We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def _pick_stateless_dp_port(self) -> tuple[int, socket.socket | None]:
        """Return ``(port, listen_socket)`` for DP group init.

        With a coord store, rank 0 binds a socket and publishes the port;
        others read it.  Without one, pops a pre-allocated port and
        returns ``listen_socket=None``.
        """
        if not self._coord_store_port:
            return self.get_next_dp_init_port(), None

        from irwkv.distributed.utils import get_cached_tcp_store_client

        store = get_cached_tcp_store_client(
            self.data_parallel_master_ip, self._coord_store_port
        )

        key = "dp_master_port"
        if self.data_parallel_rank == 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.data_parallel_master_ip, 0))
            s.listen()
            port = s.getsockname()[1]
            store.set(key, str(port).encode())
            return port, s
        else:
            return int(store.get(key).decode()), None

    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[False] = ...
    ) -> ProcessGroup: ...
    @overload
    def stateless_init_dp_group(
        self, return_store: Literal[True] = ...
    ) -> tuple[ProcessGroup, Store]: ...
    def stateless_init_dp_group(
        self, return_store: bool = False
    ) -> ProcessGroup | tuple[ProcessGroup, Store]:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from irwkv.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                port, listen_socket = self._pick_stateless_dp_port()
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    port,
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                    listen_socket=listen_socket,
                    timeout=self.cpu_distributed_timeout,
                )
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism, replicated tokens results in useless
    # duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #

    @property
    def node_rank_within_dp(self) -> int:
        return self.node_rank % self.nnodes_within_dp

    @property
    def nnodes_within_dp(self) -> int:
        """Number of nodes one DP replica spans.

        External LB pins ``data_parallel_size_local`` to 1, so the ratio
        rounds down to 0 once DP replicas outnumber nodes. A replica that
        does not span nodes still occupies exactly one.
        """
        if self.nnodes == 1:
            return 1
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        nnodes_within_dp = self.nnodes // data_parallel_node_size
        if self.data_parallel_external_lb:
            return max(nnodes_within_dp, 1)
        if self.nnodes % data_parallel_node_size != 0:
            raise ValueError(
                "Invalid data parallel configuration: "
                f"nnodes ({self.nnodes}) must be divisible by the number of "
                "data parallel node groups "
                f"({data_parallel_node_size} = data_parallel_size "
                f"{self.data_parallel_size} / data_parallel_size_local "
                f"{self.data_parallel_size_local})"
            )
        return nnodes_within_dp

    @property
    def local_world_size(self) -> int:
        return self.world_size // self.nnodes_within_dp

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_dp_state(
        dp_group: ProcessGroup, has_unfinished: bool, pending_pause: bool
    ) -> tuple[bool, bool]:
        """Combined all-reduce for DP state synchronization.

        Uses a single SUM all-reduce on a 2-element tensor:
          [0] = 1 if this rank has unfinished work, else 0.
                SUM > 0 ≡ logical OR across ranks → any rank has work.
          [1] = 1 if this rank has a pending pause request, else 0.
                SUM == dp_size ≡ all ranks reached pause consensus.

        has_unfinished_global is true if any rank has unfinished work,
        or if some ranks are waiting for a pause consensus.

        Returns:
            (has_unfinished_global, pause_consensus)

        """
        tensor = torch.tensor(
            [int(has_unfinished), int(pending_pause)], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=dp_group)
        dp_size = dp_group.size()
        pause_count = tensor[1].item()
        has_unfinished_global = tensor[0].item() > 0 or pause_count % dp_size != 0
        return has_unfinished_global, pause_count == dp_size

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "_coord_store_port",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "enable_shm_tensor_arena",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
            # NUMA binding is per-rank host-side memory locality; it does
            # not affect collective-communication semantics. When numa_bind
            # is enabled with auto-detection, each DP rank stores its own
            # NUMA node in numa_bind_nodes (see irwkv/utils/numa_utils.py
            # `_get_numa_node`), which would otherwise diverge the DP hash.
            "numa_bind",
            "numa_bind_nodes",
            "numa_bind_cpus",
            "assigned_physical_gpu_ids",
        }

        from irwkv.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None:
        # Continue with the rest of the initialization
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self._data_parallel_master_port_list:
                self._data_parallel_master_port_list = get_open_ports_list(5)
            self.data_parallel_master_port = self._data_parallel_master_port_list.pop()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.IRWKV_DP_SIZE
            self.data_parallel_rank = envs.IRWKV_DP_RANK
            self.data_parallel_rank_local = envs.IRWKV_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.IRWKV_DP_MASTER_IP
            self.data_parallel_master_port = envs.IRWKV_DP_MASTER_PORT

        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            os.environ["IRWKV_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from irwkv.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.IRWKV_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and current_platform.device_count() < self.world_size
            ):
                gpu_count = current_platform.device_count()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from irwkv.v1.executor import Executor

        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            from irwkv.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        # A batch below one token per microbatch cannot be split, so the
        # thresholds have to keep it out rather than the split having to cope.
        if self.use_ubatching and (
            min(self.dbo_decode_token_threshold, self.dbo_prefill_token_threshold)
            < self.num_ubatches
        ):
            raise ValueError(
                "dbo_decode_token_threshold and dbo_prefill_token_threshold must "
                f"be at least the number of microbatches ({self.num_ubatches})."
            )

        return self

    def reconfigure_for_independent_dp_rank(self) -> None:
        """Reconfigure for a single independent non-MoE DP rank."""
        # Capture these before changing DP fields.
        nnodes = self.nnodes_within_dp
        node_rank = self.node_rank_within_dp
        self.data_parallel_size = 1
        self.data_parallel_size_local = 1
        self.data_parallel_rank = 0
        self.nnodes = nnodes
        self.node_rank = node_rank
