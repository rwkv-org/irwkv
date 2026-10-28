# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path

import torch

import irwkv.envs as envs
from irwkv.config import (
    CacheConfig,
    ECTransferConfig,
    IrwkvConfig,
    KVTransferConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
)
from irwkv.config.model import RunnerOption
from irwkv.config.scheduler import SchedulerPolicy
from irwkv.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from irwkv.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from irwkv.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from irwkv.sampling_params import SamplingParams
from irwkv.utils.hashing import sha256
from irwkv.v1.core.kv_cache_manager import KVCacheBlocks
from irwkv.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from irwkv.v1.core.sched.async_scheduler import AsyncScheduler
from irwkv.v1.core.sched.scheduler import Scheduler, SchedulerOutput
from irwkv.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from irwkv.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)
from irwkv.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from irwkv.v1.request import Request
from irwkv.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


@lru_cache(maxsize=1)
def _scheduler_model_dir() -> tempfile.TemporaryDirectory:
    model_dir = tempfile.TemporaryDirectory()
    Path(model_dir.name, "config.json").write_text(
        json.dumps(
            {
                "architectures": ["RwkvForCausalLM"],
                "model_type": "rwkv",
                "architecture_version": "rwkv7",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "head_size": 64,
                "intermediate_size": 512,
                "vocab_size": 65536,
                "context_length": 16384,
                "eos_token_id": 0,
            }
        )
    )
    return model_dir


@dataclass(frozen=True)
class MockKVConfig:
    matched_tokens: int = 0
    is_async: bool = False
    num_defers_before_matching: int = 0
    supports_divergent_local_hybrid_hits: bool = False


class MockKVConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Scheduler tests check metadata.requests
        self.requests: list = []


class MockKVConnector(KVConnectorBase_V1):
    """Mock KV connector for scheduler tests, supporting both sync and async mode."""

    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(irwkv_config, role, kv_cache_config)
        extra_config = self._kv_transfer_config.kv_connector_extra_config
        self.config = MockKVConfig(
            matched_tokens=extra_config["matched_tokens"],
            is_async=extra_config["is_async"],
            num_defers_before_matching=extra_config.get(
                "num_defers_before_matching", 0
            ),
            supports_divergent_local_hybrid_hits=extra_config.get(
                "supports_divergent_local_hybrid_hits", False
            ),
        )
        self._defers_left: defaultdict[str, int] = defaultdict(
            lambda: self.config.num_defers_before_matching
        )

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return self.config.supports_divergent_local_hybrid_hits

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._defers_left[request.request_id] > 0:
            self._defers_left[request.request_id] -= 1
            return (None, False)
        return (self.config.matched_tokens, self.config.is_async)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        pass

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = MockKVConnectorMetadata()
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for req_id in chain(
            (req.req_id for req in scheduler_output.scheduled_new_reqs),
            (
                req_id
                for req_id in cached_reqs.req_ids
                if req_id in cached_reqs.resumed_req_ids
            ),
        ):
            metadata.requests.append({"req_id": req_id})
        return metadata

    def start_load_kv(self, kv_caches, finished_req_ids):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


KVConnectorFactory.register_connector("MockKVConnector", __name__, "MockKVConnector")


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    use_eos: bool = False,
    token_id: int = 0,
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""
    # Make request data.
    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = EOS_TOKEN_ID if use_eos else token_id
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    kv_connector_output = (
        None
        if (
            finished_sending is None
            and finished_recving is None
            and invalid_block_ids is None
            and kv_connector_worker_meta is None
        )
        else KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            kv_connector_worker_meta=kv_connector_worker_meta,
        )
    )

    # Make output data structure.
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        kv_connector_output=kv_connector_output,
    )


def mock_kv(matched_tokens: int, is_async: bool, num_defers_before_matching: int = 0):
    return MockKVConfig(
        matched_tokens=matched_tokens,
        is_async=is_async,
        num_defers_before_matching=num_defers_before_matching,
    )


def create_scheduler(
    model: str | None = None,
    max_num_seqs: int = 16,
    max_num_active_seqs: int | None = None,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    long_prefill_token_threshold_adaptive: bool = False,
    disable_chunked_mm_input: bool = False,
    mm_encoder_only: bool = False,
    use_kv_connector: None | bool | str | MockKVConfig = None,
    kv_role: str = "kv_both",
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    skip_tokenizer_init: bool = False,
    async_scheduling: bool = False,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    use_ec_connector: bool = False,
    ec_role: str | None = None,
    use_v2_model_runner: bool | None = None,
    kv_cache_spec: KVCacheSpec | None = None,
    runner: RunnerOption = "auto",
    scheduling_policy: SchedulerPolicy = "fcfs",
    scheduler_cls: type[Scheduler] | None = None,
) -> Scheduler | AsyncScheduler:
    """Create scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)
      long_prefill_token_threshold: cap on prefill chunk size
      long_prefill_token_threshold_adaptive: floor the cap at a
                             fair share of the token budget

    Returns:
      {class}`Scheduler` instance

    """
    model_config = ModelConfig(
        model=model or _scheduler_model_dir().name,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
        skip_tokenizer_init=skip_tokenizer_init or model is None,
        max_model_len=max_model_len,
        runner=runner,
    )
    if mm_encoder_only:
        model_config.multimodal_config.mm_encoder_only = True
    if use_ec_connector and ec_role == "ec_producer":
        model_config.multimodal_config = MultiModalConfig()
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_active_seqs=max_num_active_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        long_prefill_token_threshold_adaptive=(long_prefill_token_threshold_adaptive),
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=model_config.is_encoder_decoder,
        # Ensure admission/preemption mechanics are deterministic
        watermark=0.0,
        policy=scheduling_policy,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = None
    if isinstance(use_kv_connector, MockKVConfig):
        kv_transfer_config = KVTransferConfig(
            kv_connector="MockKVConnector",
            kv_role=kv_role,
            kv_connector_extra_config={
                "matched_tokens": use_kv_connector.matched_tokens,
                "is_async": use_kv_connector.is_async,
                "num_defers_before_matching": (
                    use_kv_connector.num_defers_before_matching
                ),
            },
        )
    elif isinstance(use_kv_connector, str):
        kv_transfer_config = KVTransferConfig(
            kv_connector=use_kv_connector,
            kv_role=kv_role,
        )
    elif use_kv_connector:
        kv_transfer_config = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role=kv_role,
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )

    ec_transfer_config = (
        ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role=ec_role,
            ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test"},
        )
        if use_ec_connector
        else None
    )

    irwkv_config = IrwkvConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
        ),
        kv_transfer_config=kv_transfer_config,
        ec_transfer_config=ec_transfer_config,
        observability_config=ObservabilityConfig(),
    )
    if kv_cache_spec is None:
        kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(irwkv_config)
    if scheduler_cls is None:
        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    scheduler = scheduler_cls(
        irwkv_config=irwkv_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(irwkv_config),
    )
    if use_v2_model_runner is None:
        use_v2_model_runner = bool(envs.IRWKV_USE_V2_MODEL_RUNNER)
    scheduler.use_v2_model_runner = use_v2_model_runner
    return scheduler


_none_hash_initialized = False


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    ignore_eos: bool = False,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
) -> list[Request]:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=ignore_eos,
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
        req_ids = [f"{i}" for i in range(num_requests)]

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

        prompt_token_ids = [0] * num_tokens if same_prompt else [i] * num_tokens
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            mm_features=mm_features if mm_features else None,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests
