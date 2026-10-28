# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import is_dataclass
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import torch
from pydantic import ConfigDict, Field, model_validator

import irwkv.envs as envs
from irwkv.logger import enable_trace_function_call, init_logger
from irwkv.transformers_utils.runai_utils import is_runai_obj_uri
from irwkv.triton_utils import HAS_TRITON
from irwkv.utils import random_uuid
from irwkv.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .ec_manager_config import EncoderCacheManagerConfig
from .ec_transfer import ECTransferConfig
from .kernel import KernelConfig
from .kv_events import KVEventsConfig
from .kv_transfer import KVTransferConfig
from .load import LoadConfig
from .logging import LoggingConfig
from .mamba import MambaConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .offload import OffloadConfig
from .parallel import ParallelConfig
from .profiler import ProfilerConfig
from .reasoning import ReasoningConfig
from .scheduler import SchedulerConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config, replace
from .watermarking import WatermarkConfig
from .weight_transfer import WeightTransferConfig

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from irwkv.sampling_params import BeamSearchParams, SamplingParams
    from irwkv.v1.kv_cache_interface import KVCacheConfig
else:
    PreTrainedConfig = Any

    KVCacheConfig = Any

logger = init_logger(__name__)


class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2s."""


PerformanceMode = Literal["balanced", "interactivity", "throughput"]


OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {},
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {},
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {},
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {},
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


@config(config=ConfigDict(arbitrary_types_allowed=True))
class IrwkvConfig:
    """Dataclass which contains all irwkv-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = None  # type: ignore[assignment]
    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig.default_factory,
    )
    """Scheduler configuration."""
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    """Load configuration."""
    offload_config: OffloadConfig = Field(default_factory=OffloadConfig)
    """Model weight offloading configuration."""
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    """Attention configuration."""
    mamba_config: MambaConfig = Field(default_factory=MambaConfig)
    """Mamba configuration."""
    kernel_config: KernelConfig = Field(default_factory=KernelConfig)
    """Kernel configuration."""
    watermark_config: WatermarkConfig | None = None
    """Text watermarking configuration."""
    structured_outputs_config: StructuredOutputsConfig = Field(
        default_factory=StructuredOutputsConfig
    )
    """Structured outputs configuration."""
    observability_config: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    """Observability configuration."""
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    """Logging configuration."""
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, one can append compilation arguments via
    -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

    You can specify the full compilation config like so:
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig)
    """Profiling configuration."""
    kv_transfer_config: KVTransferConfig | None = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: KVEventsConfig | None = None
    """The configurations for event publishing."""
    ec_transfer_config: ECTransferConfig | None = None
    """The configurations for distributed EC cache transfer."""
    ec_manager_config: EncoderCacheManagerConfig = Field(
        default_factory=EncoderCacheManagerConfig
    )
    """The configurations for custom encoder cache manager."""
    reasoning_config: ReasoningConfig | None = None
    """The configurations for reasoning model."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the irwkv instance."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -O2 is used by default. See OptimizationLevel for full
    description."""

    performance_mode: PerformanceMode = "balanced"
    """Performance mode for runtime behavior, 'balanced' is the default.
    'interactivity' favors low end-to-end per-request latency at small batch
    sizes (fine-grained CUDA graphs, latency-oriented kernels).
    'throughput' favors aggregate tokens/sec at high concurrency (larger CUDA
    graphs, more aggressive batching, throughput-oriented kernels)."""

    weight_transfer_config: WeightTransferConfig | None = None
    """The configurations for weight transfer during RL training."""

    shutdown_timeout: int = Field(default=0, ge=0)
    """Shutdown grace period for in-flight requests. Shutdown will be delayed for
    up to this amount of time to allow already-running requests to complete. Any
    remaining requests are aborted once the timeout is reached.
    """

    def compute_hash(self, include_version: bool = True) -> str:
        """WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        Args:
            include_version: Include the irwkv version in the hash.

        """
        factors: list[Any] = []

        # summarize irwkv config
        irwkv_factors: list[Any] = []
        if include_version:
            from irwkv import __version__

            irwkv_factors.append(__version__)
        if self.model_config:
            irwkv_factors.append(self.model_config.compute_hash())
            if (
                self.compilation_config
                and getattr(self.compilation_config, "compile_mm_encoder", False)
                and self.model_config.multimodal_config
            ):
                irwkv_factors.append(self.model_config.multimodal_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.cache_config:
            irwkv_factors.append(self.cache_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.parallel_config:
            irwkv_factors.append(self.parallel_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.scheduler_config:
            irwkv_factors.append(self.scheduler_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.device_config:
            irwkv_factors.append(self.device_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.load_config:
            irwkv_factors.append(self.load_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.offload_config:
            irwkv_factors.append(self.offload_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.attention_config:
            irwkv_factors.append(self.attention_config.compute_hash())
        else:
            irwkv_factors.append("None")
        irwkv_factors.append("None")
        irwkv_factors.append("None")
        if self.structured_outputs_config:
            irwkv_factors.append(self.structured_outputs_config.compute_hash())
        if self.profiler_config:
            irwkv_factors.append(self.profiler_config.compute_hash())
        else:
            irwkv_factors.append("None")
        irwkv_factors.append(self.observability_config.compute_hash())
        if self.compilation_config:
            irwkv_factors.append(self.compilation_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.kernel_config:
            irwkv_factors.append(self.kernel_config.compute_hash())
        else:
            irwkv_factors.append(None)
        if self.kv_transfer_config:
            irwkv_factors.append(self.kv_transfer_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.ec_transfer_config:
            irwkv_factors.append(self.ec_transfer_config.compute_hash())
        else:
            irwkv_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            irwkv_factors.append(additional_config_hash)
        else:
            irwkv_factors.append("None")
        factors.append(irwkv_factors)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    @property
    def is_mm_encoder_only(self) -> bool:
        mm_config = (
            self.model_config.multimodal_config
            if self.model_config is not None
            else None
        )
        return bool(mm_config and mm_config.mm_encoder_only)

    @property
    def max_concurrent_batches(self) -> int:
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        if self.scheduler_config.async_scheduling:
            return pp_size + 1
            # V1 Model Runner does not fully support async scheduling with PP.
        return pp_size

    @property
    def max_in_flight_tokens(self) -> int:
        # Upper bound on tokens that are scheduled but not yet settled (freed):
        # every concurrent batch may hold up to a full `max_num_batched_tokens`.
        # Recycling-aware KV cache specs (sliding-window, chunked-local) reserve
        # for this because out-of-window blocks are freed on the processed-token
        # basis, so in-flight steps transiently keep their blocks.
        return (
            self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens
        )

    @property
    def uniform_decode_query_len(self) -> int:
        """One query per request in a uniform decode batch."""
        return 1

    @property
    def use_cumem_cudagraph_pool(self) -> bool:
        """Whether CUDA graphs go to the cuMem pool that sleep offloads."""
        from irwkv.platforms import current_platform

        model_config = self.model_config
        return (
            model_config is not None
            and model_config.sleep_mode_offload_cudagraph
            and model_config.enable_sleep_mode
            and model_config.sleep_mode_backend == "cumem"
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and current_platform.is_cuda_alike()
        )

    @property
    def use_v2_model_runner(self) -> bool:
        return True

    def _maybe_enable_breakable_cudagraph(self) -> bool:
        from irwkv.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        enabled = is_breakable_cudagraph_enabled()
        if enabled:
            self.compilation_config.mode = CompilationMode.NONE
        return enabled

    @property
    def needs_dp_coordinator(self) -> bool:
        """Collect queue statistics for internal data-parallel load balancing."""
        return self.parallel_config.data_parallel_size > 1 and (
            (self.model_config is None)
            or (not self.parallel_config.data_parallel_external_lb)
        )

    def enable_trace_function_call_for_thread(self) -> None:
        """Set up function tracing for the current thread,
        if enabled via the `IRWKV_TRACE_FUNCTION` environment variable.
        """
        if envs.IRWKV_TRACE_FUNCTION:
            tmp_dir = tempfile.gettempdir()
            # add username to tmp_dir to avoid permission issues
            tmp_dir = os.path.join(tmp_dir, getpass.getuser())
            filename = (
                f"IRWKV_TRACE_FUNCTION_for_process_{os.getpid()}"
                f"_thread_{threading.get_ident()}_at_{datetime.now()}.log"
            ).replace(" ", "_")
            log_path = os.path.join(
                tmp_dir,
                "irwkv",
                f"irwkv-instance-{self.instance_id}",
                filename,
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            enable_trace_function_call(log_path)

    def with_hf_config(
        self,
        hf_config: PreTrainedConfig,
        architectures: list[str] | None = None,
    ) -> "IrwkvConfig":
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures
        elif hf_config.architectures is None:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )

            if hf_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                hf_config = copy.deepcopy(hf_config)
                hf_config.architectures = [
                    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[hf_config.model_type]
                ]

        model_config = copy.deepcopy(self.model_config)

        # In Transformers v5, tie_word_embeddings belongs to the config of the class
        # that can see both layers to be tied. For example:
        #
        # SomeVLModel:
        #   self.language_model = SomeLanguageModel(SomeVLTextConfig)
        #   self.vision_model = SomeVisionModel(SomeVLVisionConfig)
        #
        # SomeVLModelForMultimodalLM:
        #   self.model = SomeVLModel(SomeVLConfig)
        #   self.lm_head = nn.Linear()
        #
        # Therefore, tie_word_embeddings is defined in SomeVLConfig and is not present
        # in SomeVLTextConfig*. In irwkv, the lm_head belongs to the language_model, so
        # we must ensure that tie_word_embeddings is set in the language_model's config.
        #
        # *For some models, SomeVLTextConfig may also have a tie_word_embeddings field.
        # This is only the case if SomeVLTextConfig is also used for a text only version
        # of the same model. For example:
        #
        # SomeVLModelForCausalLM:
        #   self.model = SomeLanguageModel(SomeVLTextConfig)
        #   self.lm_head = nn.Linear()
        #
        # Therefore, the presence of tie_word_embeddings in SomeVLTextConfig cannot
        # be used as a signal for whether tie_word_embeddings should be copied from
        # hf_config to the language_model config.

        model_config.hf_config = hf_config
        model_config.model_arch_config = model_config.get_model_arch_config()
        model_config.is_submodel_config = True

        return replace(self, model_config=model_config)

    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user.

        Args:
            config_obj: Configuration object to update.
            key: Attribute name.
            value: Default value (static or callable).

        """
        if getattr(config_obj, key) is None:
            # Some config values are known before initialization and are
            # hard coded.
            # Other values depend on the user given configuration, so they are
            # implemented with lambda functions and decided at run time.
            setattr(config_obj, key, value(self) if callable(value) else value)

    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten.

        If the user configuration does not specify a value for a default field
        and if the default field is still None after all user selections are
        applied, then default values will be applied to the field. User specified
        fields will not be overridden by the default.

        Args:
            defaults: Dictionary of default values to apply.

        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            """Recursively apply defaults to config_obj, using self as root."""
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    def _normalize_piecewise_cudagraph_mode(
        self, *, breakable_cudagraph_enabled: bool
    ) -> None:
        compilation_config = self.compilation_config
        if (
            compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and compilation_config.mode != CompilationMode.IRWKV_COMPILE
            and not breakable_cudagraph_enabled
        ):
            fallback_mode = compilation_config.cudagraph_mode.without_piecewise()
            logger.info_once(
                "Cudagraph mode %s is not compatible with compilation mode %s. "
                "Overriding to %s.",
                compilation_config.cudagraph_mode,
                compilation_config.mode,
                fallback_mode,
            )
            compilation_config.cudagraph_mode = fallback_mode
            if fallback_mode == CUDAGraphMode.NONE:
                compilation_config.max_cudagraph_capture_size = 0
                compilation_config.cudagraph_capture_sizes = []

    def _post_init_kv_transfer_config(self) -> None:
        """Update KVTransferConfig based on top-level configs in IrwkvConfig.

        Right now, this function reads the offloading settings from
        CacheConfig and configures the KVTransferConfig accordingly.
        """
        # KV offloading is only activated when kv_offloading_size is set.
        if (kv_offloading_size := self.cache_config.kv_offloading_size) is None:
            return

        kv_offloading_backend = self.cache_config.kv_offloading_backend

        # If no KVTransferConfig is provided, create a default one.
        if self.kv_transfer_config is None:
            self.kv_transfer_config = KVTransferConfig()

        if kv_offloading_backend == "native":
            if envs.IRWKV_USE_SIMPLE_KV_OFFLOAD:
                config_connector = "SimpleCPUOffloadConnector"
            else:
                config_connector = "OffloadingConnector"
            self.kv_transfer_config.kv_connector = config_connector
            self.kv_transfer_config.kv_connector_extra_config.update(
                {"cpu_bytes_to_use": kv_offloading_size * (1073741824)}
            )
        elif kv_offloading_backend == "lmcache":
            # Default to LMCache multi-process (MP) mode. The actual KV
            # storage capacity is managed by the standalone LMCache server
            # process, so ``kv_offloading_size`` is not propagated here.
            # ``LMCacheMPConnector`` falls back to ``tcp://localhost:5555``
            # when host/port are not provided via extra_config.
            self.kv_transfer_config.kv_connector = "LMCacheMPConnector"

        # This is the same for all backends
        self.kv_transfer_config.kv_role = "kv_both"

    def _verify_kv_transfer_compat(self) -> None:
        """Reject configurations that silently corrupt KV transfers."""
        if (
            self.kv_transfer_config is None
            or self.kv_transfer_config.kv_connector is None
        ):
            return

        # PyTorch's expandable_segments allocator uses CUDA VMM, which can
        # remap a virtual address range to different physical pages over the
        # engine's lifetime. KV connectors that pin KV cache memory (e.g.
        # NixlConnector via ibv_reg_mr, MooncakeConnector) end up with their
        # registrations pointing at stale physical pages after any remap,
        # producing RDMA failures like IBV_WC_REM_ACCESS_ERR /
        # NIXL_ERR_REMOTE_DISCONNECT at the first inter-node KV transfer.
        # We can't enumerate every in-tree and out-of-tree connector that
        # pins memory, so we conservatively reject the combination whenever
        # any KV connector is configured.
        #
        # CuMem allocator is exempt: CuMemAllocator.use_memory_pool toggles
        # expandable_segments off around its pool (see #40812), so the KV
        # cache allocated within that context lands on stable physical pages
        # even when the env var is set.
        if "expandable_segments:True" not in os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", ""
        ):
            return
        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "unless enable_cumem_allocator is also enabled. PyTorch's CUDA VMM "
            "allocator can remap KV cache virtual addresses to different "
            "physical pages, invalidating any pinned/registered KV memory "
            "(e.g. IB memory regions registered by NIXL or Mooncake). Either "
            "unset expandable_segments:True or enable the cumem allocator "
            "(sleep mode does this automatically and also "
            "routes KV allocations through CuMemAllocator's pool, where "
            "expandable_segments is automatically disabled)."
        )

    def _verify_sampling_replay_config(self) -> None:
        model_config = self.model_config
        if model_config is None or not model_config.return_sampling_mask:
            return
        if model_config.logits_processors:
            raise ValueError(
                "sampling distribution replay does not support custom logits processors"
            )
        if model_config.logprobs_mode != "processed_logprobs":
            raise ValueError(
                "sampling distribution replay requires "
                "logprobs_mode='processed_logprobs' so that returned logprobs "
                "are normalized over the same nucleus as the sampling mask"
            )

    def _check_supports_watermarking(
        self,
        config: "SamplingParams | BeamSearchParams | None" = None,
        *,
        custom_sampler: bool = False,
    ) -> bool:
        watermark_config = getattr(self, "watermark_config", None)
        if watermark_config is None:
            if config is not None and config.watermarking is True:
                logger.warning_once(
                    "Watermarking is enabled for this request, but the engine has no "
                    "watermark configuration. This and subsequent requests will run "
                    "without watermarking.",
                    scope="global",
                )
            return False
        if custom_sampler:
            raise ValueError(
                "Model-specific custom samplers are not supported with watermarking."
            )
        if config is None:
            return True
        if config.watermarking is False:
            return False

        from irwkv.sampling_params import BeamSearchParams, SamplingParams

        if isinstance(config, BeamSearchParams):
            logger.warning_once(
                "Watermarking is enabled, but beam search cannot be watermarked. "
                "This and subsequent beam search requests will run without "
                "watermarking.",
                scope="global",
            )
            return False
        if not isinstance(config, SamplingParams):
            raise TypeError(f"Unsupported watermarking config: {type(config).__name__}")
        if config.trace_decode_token_ids is not None:
            logger.warning_once(
                "Watermarking is enabled, but trace replay cannot be watermarked. "
                "This and subsequent trace replay requests will run without "
                "watermarking.",
                scope="global",
            )
            return False
        if config.temperature == 0:
            logger.warning_once(
                "Watermarking is enabled, but greedy decoding "
                "(temperature=0) cannot be watermarked. This and subsequent "
                "greedy requests will use ordinary greedy sampling.",
                scope="global",
            )
            return False
        return True

    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""
        # To give each torch profile run a unique instance name.
        self.instance_id = f"{time.time_ns()}"

        if self.model_config is not None and self.model_config.is_submodel_config:
            # with_hf_config() view: the parent config was already validated,
            # and this view's empty architecture list makes the model-dependent checks
            # below unsafe (e.g. use_mla resolves the architecture registry).
            return

        self._resolve_mm_encoder_only()

        if self.performance_mode != "balanced":
            logger.info_once("Performance mode set to '%s'.", self.performance_mode)

        self.try_verify_and_update_config()

        self._check_supports_watermarking()
        # Models may have supplied their own DCP defaults above; anything still
        # unset falls back to the stock ones.
        self.parallel_config.set_dcp_defaults()

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)

        if (
            self.model_config is not None
            and self.model_config.multimodal_config is not None
            and self.model_config.multimodal_config.language_model_only
            and self.compilation_config.cudagraph_mm_encoder
        ):
            raise ValueError(
                "--language-model-only is incompatible with "
                "cudagraph_mm_encoder=True, since it disables all multimodal "
                "inputs and the multimodal encoder is never run. Please "
                "disable one of them."
            )

        self._verify_sampling_replay_config()

        # A NIXL side is either fully replicated or fully DCP-sharded; MLA only.
        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.has_connector("NixlConnector")
        ):
            dcp_size = self.parallel_config.decode_context_parallel_size
            transfer_tp_size = max(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.prefill_context_parallel_size,
            )
            assert dcp_size in (1, transfer_tp_size), (
                f"decode_context_parallel_size={dcp_size} must be 1 or equal "
                f"to the NIXL transfer parallel size={transfer_tp_size}."
            )
            if self.model_config is not None:
                assert self.model_config.use_mla or dcp_size == 1, (
                    "PD with decode_context_parallel_size > 1 is only "
                    "supported for MLA models."
                )

        if (
            self.mamba_config.enable_stochastic_rounding
            and self.cache_config.mamba_ssm_cache_dtype != "float16"
        ):
            raise ValueError(
                "Stochastic rounding for Mamba cache requires "
                "the SSM cache to be float16. Please set it explicitly, "
                "by specifying `--mamba-ssm-cache-dtype float16`, or disable "
                "stochastic rounding by not specifying "
                "`--enable-mamba-cache-stochastic-rounding`."
            )

        # "dummy" reads no weights at all, and the sharded formats read a irwkv
        # state dict, which stores tied word embeddings under the lm_head only.
        # Neither can tell us what the original checkpoint contained.
        if self.model_config is not None and self.load_config.load_format not in (
            "dummy",
            "sharded_state",
            "runai_streamer_sharded",
        ):
            self.model_config.maybe_untie_word_embeddings()

        from irwkv.platforms import current_platform
        from irwkv.v1.executor.abstract import Executor

        executor_backend = self.parallel_config.distributed_executor_backend
        executor_class = Executor.get_class(self)
        executor_supports_async_sched = executor_class.supports_async_scheduling()
        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            if not executor_supports_async_sched:
                raise ValueError(
                    f"`{executor_backend}` does not support async scheduling yet."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend. ",
                    executor_backend,
                )
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            if self.scheduler_config.async_scheduling:
                if self.parallel_config.data_parallel_size > 1 and (
                    self.model_config is None
                ):
                    logger.info_once(
                        "Disabling NCCL for DP synchronization "
                        "when using async scheduling.",
                    )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        if (
            self.model_config is not None
            and self.model_config.multimodal_config is not None
            and self.model_config.multimodal_config.mm_tensor_ipc == "torch_shm"
            and os.environ.get("IRWKV_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            raise ValueError(
                "torch_shm is known to fail without "
                "IRWKV_WORKER_MULTIPROC_METHOD set to spawn"
            )

        if (
            self.model_config is not None
            and self.scheduler_config.enable_chunked_prefill
            and self.model_config.dtype == torch.float32
            and current_platform.get_device_capability() == (7, 5)
        ):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, irwkv will set 'ieee' input "
                "precision for chunked prefill triton kernels."
            )

        if self.model_config is not None and self.model_config.enforce_eager:
            self.compilation_config.mode = CompilationMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning_once(
                "Enforce eager set, disabling torch.compile, CUDAGraphs, and "
                "JIT kernel warmup. This is equivalent to setting "
                "-cc.mode=none -cc.cudagraph_mode=none and "
                "--kernel_config.enable_jit_warmup=False"
            )
            self.kernel_config.enable_jit_warmup = False

        if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
            logger.warning_once(
                "TORCH_COMPILE_DISABLE is set, disabling torch.compile. "
                "This is equivalent to setting -cc.mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE

        breakable_cudagraph_enabled = self._maybe_enable_breakable_cudagraph()

        if not breakable_cudagraph_enabled and (
            self.compilation_config.backend == "eager"
            or (
                self.compilation_config.mode is not None
                and self.compilation_config.mode != CompilationMode.IRWKV_COMPILE
            )
        ):
            logger.warning_once(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094

        current_platform.apply_config_platform_defaults(self)

        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.IRWKV_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # By default, enable torch wrapping only when using custom Inductor lowering
        if self.compilation_config.ir_enable_torch_wrap is None:
            self.compilation_config.ir_enable_torch_wrap = (
                self.compilation_config.mode == CompilationMode.IRWKV_COMPILE
                and self.compilation_config.backend == "inductor"
            )

        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        # This populates IR op priorities,
        # must happen after compilation mode and backend are decided,
        # but before fusion defaults are applied as those may depend on op priority.
        self.kernel_config.set_platform_defaults(self)

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after applying "
                "optimization level defaults."
            )

        self._normalize_piecewise_cudagraph_mode(
            breakable_cudagraph_enabled=breakable_cudagraph_enabled
        )

        self._set_max_num_scheduled_tokens()

        if current_platform.support_static_graph_mode():
            # if cudagraph_mode has full cudagraphs, we need to check support

            # Check if KV connector requires PIECEWISE mode for CUDA graphs
            if (
                self.kv_transfer_config is not None
                and self.kv_transfer_config.is_kv_transfer_instance
                and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            ):
                # Lazy import to avoid circular dependencies
                from irwkv.distributed.kv_transfer.kv_connector.factory import (
                    KVConnectorFactory,
                )

                connector_cls = KVConnectorFactory.get_connector_class(
                    self.kv_transfer_config
                )
                if connector_cls.requires_piecewise_for_cudagraph(
                    self.kv_transfer_config.kv_connector_extra_config
                ):
                    logger.warning_once(
                        "KV connector %s requires PIECEWISE CUDA graph mode "
                        "due to layerwise async operations that cannot be "
                        "captured in CUDA graphs. "
                        "Overriding cudagraph_mode from %s to PIECEWISE.",
                        connector_cls.__name__,
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info_once("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # override related settings when enforce eager
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                self.compilation_config.cudagraph_num_of_warmups = 1

            self._set_cudagraph_sizes()

        else:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if self.cache_config.kv_sharing_fast_prefill:
            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings."
            )

        if (
            self.model_config
            and self.model_config.architecture == "WhisperForConditionalGeneration"
            and os.environ.get("IRWKV_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            logger.warning_once(
                "Whisper is known to have issues with "
                "forked workers. If startup is hanging, "
                "try setting 'IRWKV_WORKER_MULTIPROC_METHOD' "
                "to 'spawn'."
            )

        if (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
            and not self.cache_config.enable_prefix_caching
        ):
            logger.warning_once(
                "KV cache events are on, but prefix caching is not enabled. "
                "Use --enable-prefix-caching to enable."
            )
        if (
            self.kv_events_config is not None
            and self.kv_events_config.publisher != "null"
            and not self.kv_events_config.enable_kv_cache_events
        ):
            logger.warning_once(
                "KV cache events are disabled, "
                "but the scheduler is configured to publish them. "
                "Modify KVEventsConfig.enable_kv_cache_events "
                "to True to enable."
            )
        current_platform.check_and_update_config(self)

        self._normalize_piecewise_cudagraph_mode(
            breakable_cudagraph_enabled=breakable_cudagraph_enabled
        )

        self._resolve_mm_embedding_inputs()
        self._resolve_mm_processor_device()
        self._resolve_mm_video_decode_device()
        self._validate_mm_processor_device()

        self._disable_cudagraphs_for_v2_stock_torch_compile()
        self._validate_v2_model_runner()

        self._validate_profiler_config()
        self._validate_batch_sharded_sampling()

        # Re-compute compile ranges after platform-specific config updates
        # (e.g., XPU may lower max_num_batched_tokens when MLA is enabled)
        self._set_compile_ranges()

        # Do this after all the updates to compilation_config.mode
        effective_dp_size = (
            self.parallel_config.data_parallel_size if self.model_config is None else 1
        )
        self.compilation_config.set_splitting_ops_for_v1(
            data_parallel_size=effective_dp_size,
        )

        # final check of cudagraph mode after all possible updates
        if current_platform.is_cuda_alike():
            if (
                self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and self.model_config is not None
                and not self.model_config.disable_cascade_attn
                and not self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()  # noqa: E501
            ):
                logger.warning_once(
                    "No piecewise cudagraph for executing cascade attention. "
                    "Will fall back to eager execution if a batch runs into "
                    "cascade attentions."
                )

            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert (
                    self.compilation_config.mode == CompilationMode.IRWKV_COMPILE
                    or envs.IRWKV_USE_BREAKABLE_CUDAGRAPH
                ), (
                    "Compilation mode should be CompilationMode.IRWKV_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )
        if (
            self.model_config
            and envs.IRWKV_BATCH_INVARIANT
            and not self.model_config.disable_cascade_attn
        ):
            self.model_config.disable_cascade_attn = True
            logger.warning_once(
                "Disabling cascade attention when IRWKV_BATCH_INVARIANT is enabled.",
            )

        if (
            self.parallel_config.use_ubatching
            and not self.model_config.disable_cascade_attn
        ):
            self.model_config.disable_cascade_attn = True
            logger.warning_once("Disabling cascade attention when DBO is enabled.")

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        if self.reasoning_config is not None and self.model_config is not None:
            self.reasoning_config.initialize_token_ids(self.model_config)
            if not self.reasoning_config.enabled:
                logger.warning_once(
                    "Auto-initialization of reasoning token IDs failed. "
                    "Please check whether your reasoning parser has implemented "
                    "the `reasoning_start_str` and `reasoning_end_str`."
                )

        # Resolve kv_offloading-derived connector name into kv_transfer_config
        # before the HMA check below, which inspects the connector class.
        self._post_init_kv_transfer_config()

        # Hybrid KV cache manager (HMA) runtime rules:
        # - Explicit enable (--no-disable-kv-cache-manager): error if runtime
        #   disables it
        # - No preference: auto-disable for unsupported features or connector configs
        # - Explicit disable (--disable-kv-cache-manager): always respect it
        need_disable_hybrid_kv_cache_manager = False
        # logger should only print warning message for hybrid models. As we
        # can't know whether the model is hybrid or not now, so we don't log
        # warning message here and will log it later.
        if not current_platform.support_hybrid_kv_cache():
            # Hybrid KV cache manager is not supported on non-GPU platforms.
            need_disable_hybrid_kv_cache_manager = True
        if (
            self.model_config is not None
            and self.model_config.attention_chunk_size is not None
        ) and not envs.IRWKV_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
            logger.warning(
                "There is a latency regression when using chunked local"
                " attention with the hybrid KV cache manager. Disabling"
                " it, by default. To enable it, set the environment "
                "IRWKV_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
            )
            # Hybrid KV cache manager is not yet supported with chunked
            # local attention.
            need_disable_hybrid_kv_cache_manager = True

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Auto-disable HMA only when the connector config does not support it.
            if self.kv_transfer_config is not None:
                from irwkv.distributed.kv_transfer.kv_connector.factory import (
                    KVConnectorFactory,
                )

                if not KVConnectorFactory.supports_hma_config(self.kv_transfer_config):
                    need_disable_hybrid_kv_cache_manager = True
                    logger.warning(
                        "Turning off hybrid kv cache manager because "
                        "`--kv-transfer-config` selects a KV connector that "
                        "does not support it. Impact: hybrid SSM models "
                        "(e.g. Jamba, Bamba) require HMA and will fail at "
                        "startup without it; models with sliding window "
                        "attention will run with reduced performance. "
                        "To add HMA support to a KV connector, subclass "
                        "`SupportsHMA` defined in kv_connector/v1/base.py "
                        "(for MultiConnector, all child connectors must "
                        "support HMA)."
                    )
            self.scheduler_config.disable_hybrid_kv_cache_manager = (
                need_disable_hybrid_kv_cache_manager
            )
        elif (
            self.scheduler_config.disable_hybrid_kv_cache_manager is False
            and need_disable_hybrid_kv_cache_manager
        ):
            raise ValueError(
                "Hybrid KV cache manager was explicitly enabled but is not "
                "supported in this configuration. Consider omitting the "
                "--no-disable-hybrid-kv-cache-manager flag to let irwkv decide"
                " automatically."
            )

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Default to enable HMA if not explicitly disabled by user or logic above.
            self.scheduler_config.disable_hybrid_kv_cache_manager = False

        if self.compilation_config.debug_dump_path:
            self.compilation_config.debug_dump_path = (
                self.compilation_config.debug_dump_path.absolute().expanduser()
            )
        if envs.IRWKV_DEBUG_DUMP_PATH is not None:
            env_path = Path(envs.IRWKV_DEBUG_DUMP_PATH).absolute().expanduser()
            if self.compilation_config.debug_dump_path:
                logger.warning(
                    "Config-specified debug dump path is overridden"
                    " by IRWKV_DEBUG_DUMP_PATH to %s",
                    env_path,
                )
            self.compilation_config.debug_dump_path = env_path

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094

        self._verify_kv_transfer_compat()
        if self.use_cumem_cudagraph_pool:
            # NCCL graph registration pins the offloaded pool; workers inherit this.
            value = os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")
            if value != "0":
                logger.warning(
                    "NCCL_GRAPH_REGISTER=%s pins the CUDA graph pool during sleep.",
                    value,
                )
        # Log the custom passes that are enabled

    def _set_max_num_scheduled_tokens(self):
        """In most cases, the scheduler may schedule a batch with as many tokens as the
        worker is configured to handle.
        """

    def _set_cudagraph_sizes(self):
        """IRWKV defines the default candidate list of batch sizes for CUDA graph
        capture as:

        ```python
        default_max_graph_size = 1024 if is_data_center_blackwell else 512
        decode_query_len = self.uniform_decode_query_len
        max_graph_size = min(
            max_num_seqs * decode_query_len * 2, default_max_graph_size
        )
        # 1, 2, 4, then multiples of 8 up to 256 and then multiples of 16
        # up to max_graph_size
        cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_graph_size + 1, 16))

        `max_num_batched_tokens` is also appended to the list if it fits
        within `max_cudagraph_capture_size`, so the max batch size is captured
        even when off-stride. Uniform decode sizes are appended when they fit
        within the platform's default capture ceiling, since they need not land
        on an 8- or 16-token stride.

        In the end, `irwkv_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in ascending order).

        These sizes are used to capture and reuse CUDA graphs for
        performance-critical paths (e.g., decoding). Capturing enables
        significantly faster kernel dispatch by avoiding Python overhead. The
        list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
        most GPUs), which controls the total allowed number of tokens in a
        batch. Since each sequence may have a variable number of tokens, the
        maximum usable batch size will depend on actual sequence lengths.

        Example:
            With `max_num_batched_tokens = 8192`, and typical sequences
            averaging ~32 tokens, most practical batch sizes fall below 256.
            However, the system will still allow capture sizes up to the
            platform default if shape and memory permit.

        Note:
            If users explicitly specify cudagraph capture sizes in the
            compilation config, those will override this default logic.
            At runtime:

            - If batch size <= one of the `cudagraph_capture_sizes`, the closest
            padded CUDA graph will be used.
            - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
            not be used.

        """
        if (
            self.model_config is not None
            and not self.model_config.enforce_eager
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            # Decode sizes to cover, in tokens. Populated only when the default
            # is computed here, so an explicit capture range is left exactly as
            # configured.
            uniform_decode_sizes: list[int] = []
            if max_cudagraph_capture_size is None:
                from irwkv.platforms import current_platform

                default_max_graph_size = (
                    1024 if current_platform.is_device_capability_family(100) else 512
                )
                decode_query_len = self.uniform_decode_query_len
                max_num_seqs = self.scheduler_config.max_num_seqs
                max_cudagraph_capture_size = min(
                    max_num_seqs * decode_query_len * 2, default_max_graph_size
                )
                if max_num_seqs <= max_cudagraph_capture_size:
                    uniform_decode_sizes = [max_num_seqs]
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )

            # determine the cudagraph_capture_sizes
            if self.compilation_config.cudagraph_capture_sizes is not None:
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                # de-duplicate the sizes provided by the config
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                # sort to make sure the sizes are in ascending order
                cudagraph_capture_sizes.sort()
            else:
                if self.performance_mode == "interactivity":
                    # Fine-grained CUDA graphs at small batch sizes
                    # for minimal padding overhead
                    interactivity_max = min(max_cudagraph_capture_size, 32)
                    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
                else:
                    cudagraph_capture_sizes = [
                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                    ]
                if max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
                # ensure max_num_tokens is captured if within max capture size
                if (
                    max_num_tokens <= max_cudagraph_capture_size
                    and max_num_tokens not in cudagraph_capture_sizes
                ):
                    cudagraph_capture_sizes.append(max_num_tokens)
                # Preserve the platform's default capture ceiling. Larger
                # uniform decode batches fall back to eager execution unless
                # users explicitly configure wider capture sizes.
                cudagraph_capture_sizes += [
                    size for size in uniform_decode_sizes if size <= max_num_tokens
                ]
                # de-duplicate and sort the sizes
                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))

            # user-specific compilation_config.max_cudagraph_capture_size get
            # truncated to valid_max_size when they are inconsistent.
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            if (
                self.compilation_config.max_cudagraph_capture_size is not None
                and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # raise error only when both two flags are user-specified
                # and they are inconsistent with each other
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )

                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d",
                    valid_max_size,
                )
            # always set the final max_cudagraph_capture_size
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                # If users have specified capture sizes, we only need to
                # compare the lens before and after modification since the modified
                # list is only the subset of the original list.
                logger.warning(
                    (
                        "cudagraph_capture_sizes specified in compilation_config"
                        " %s is overridden by config %s"
                    ),
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            # always write back the final sizes
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

        else:
            # no cudagraph in use
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # complete the remaining process.
        self.compilation_config.post_init_cudagraph_sizes()

    def _set_compile_ranges(self):
        """Set the compile ranges for the compilation config."""
        compilation_config = self.compilation_config
        computed_compile_ranges_endpoints = []

        # The upper bound of the compile ranges is the max_num_batched_tokens.
        compile_range_end = self.scheduler_config.max_num_batched_tokens
        if compile_range_end is not None:
            computed_compile_ranges_endpoints.append(compile_range_end)

        if compilation_config.compile_ranges_endpoints is not None:
            for x in compilation_config.compile_ranges_endpoints:
                assert isinstance(x, int)
                assert x > 0, f"Invalid compile range endpoint: {x}"
                if compile_range_end is not None and x < compile_range_end and x > 1:
                    computed_compile_ranges_endpoints.append(x)
        compilation_config.compile_ranges_endpoints = sorted(
            computed_compile_ranges_endpoints
        )

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from irwkv.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
            self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                "runai_streamer",
                "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from object storage (S3/GCS/Azure), "
                    f"'load_format' must be 'runai_streamer' or "
                    f"'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

    def compile_debug_dump_path(self) -> Path | None:
        """Returns a rank-aware path for dumping
        torch.compile debug information.
        """
        if self.compilation_config.debug_dump_path is None:
            return None
        tp_rank = self.parallel_config.rank
        dp_rank = self.parallel_config.data_parallel_index
        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
        path = self.compilation_config.debug_dump_path / append_path
        return path

    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"decode_context_parallel_size={self.parallel_config.decode_context_parallel_size}, "  # noqa
            f"dcp_comm_backend={self.parallel_config.dcp_comm_backend}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"structured_outputs_config={self.structured_outputs_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"enable_chunked_prefill={self.scheduler_config.enable_chunked_prefill}, "  # noqa
            f"compilation_config={self.compilation_config!r}, "
            f"kernel_config={self.kernel_config!r}"
        )

    def _resolve_mm_embedding_inputs(self) -> None:
        """Accept embedding inputs, tensor optional, on disaggregated consumers.

        An EC consumer loads embeddings from its connector. A KV consumer
        receives the prompt KV produced from those embeddings, so it does not
        need the tensors either. On every other deployment a missing tensor is
        a client error and must keep failing fast in the frontend.
        """
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return

        ec_config = self.ec_transfer_config
        kv_config = self.kv_transfer_config
        # Derived, so overwrite unconditionally rather than honouring a value
        # that was set by hand.
        mm_config.allow_missing_mm_embeddings = (
            ec_config is not None and ec_config.is_ec_consumer
        ) or (kv_config is not None and kv_config.is_kv_consumer)
        if not mm_config.allow_missing_mm_embeddings:
            return

        if not mm_config.enable_mm_embeds:
            # Allowing missing tensors still requires enabling embedding inputs
            # for the frontend to accept metadata-only requests.
            mm_config.enable_mm_embeds = True
            logger.info_once(
                "EC/KV consumer: accepting pre-computed-embedding inputs, "
                "which this role is sent by definition."
            )
        logger.info_once(
            "EC/KV consumer: pre-computed-embedding inputs may "
            "omit the embedding tensor."
        )

    def _resolve_mm_encoder_only(self) -> None:
        """Enable encoder-only mode for a dedicated EC producer."""
        ec_config = self.ec_transfer_config
        if ec_config is None or not ec_config.is_encode_only:
            return

        model_config = self.model_config
        mm_config = model_config.multimodal_config if model_config is not None else None
        if mm_config is None:
            raise ValueError(
                "An EC producer-only instance requires a multimodal model."
            )
        mm_config.mm_encoder_only = True

    def _resolve_mm_processor_device(self) -> None:
        """Settle `--mm-processor-device=auto` now that the EC role is known.

        "auto" means "the accelerator, but only where the processor has it to
        itself and its output can be handed over without a copy back to host":
        an encode-only instance whose tensor transport carries device tensors.
        Every other deployment keeps the processor on CPU.

        An explicit device -- from `--mm-processor-device` or straight from
        `mm_processor_kwargs` -- is already folded in by `MultiModalConfig`, so
        it is left alone here and validated by `_validate_mm_processor_device`.
        """
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return
        if mm_config.get_mm_processor_device_type() is not None:
            return

        from irwkv.platforms import current_platform

        device_type = current_platform.device_type
        if device_type in ("", "cpu"):
            return

        ec_config = self.ec_transfer_config
        # An EC producer that is not also a consumer runs no forward pass and
        # allocates no KV cache, so frontend accelerator work has the device to
        # itself.
        if ec_config is None or not ec_config.is_encode_only:
            return

        if mm_config.mm_tensor_ipc != "torch_shm":
            # Any other transport serializes host bytes, so the output would be
            # copied back, and that copy costs more than running the transform
            # on device saves.
            logger.info_once(
                "EPD encoder instance: keeping the multi-modal processor on CPU "
                "because mm_tensor_ipc=%s cannot carry device tensors. Add "
                "--mm-tensor-ipc=torch_shm to run it on the accelerator.",
                mm_config.mm_tensor_ipc,
            )
            return

        mm_config.mm_processor_kwargs = {
            **(mm_config.mm_processor_kwargs or {}),
            "device": device_type,
        }
        logger.info_once(
            "EPD encoder instance: running the multi-modal processor on %s. "
            "Override with --mm-processor-device=cpu.",
            device_type,
        )

    def _resolve_mm_video_decode_device(self) -> None:
        """Default video decoding to torchcodec GPU backend for EPD encoder-only
        instance if the mm processor runs on CUDA.

        The processor consumes the decoded frames on-device in that case, so
        keeping the frames on the GPU skips the host round-trip through the
        CPU media path. An explicit codec/backend choice in
        `--media-io-kwargs` is left alone, and the default is skipped where
        torchcodec (or its FFmpeg runtime) is unavailable.
        """
        if self.model_config is None or self.model_config.multimodal_config is None:
            return
        mm_config = self.model_config.multimodal_config

        ec_config = self.ec_transfer_config
        # An EC producer that is not also a consumer runs no forward pass and
        # allocates no KV cache, so frontend accelerator work has the device to
        # itself.
        if ec_config is None or not ec_config.is_encode_only:
            return

        from irwkv.platforms import current_platform

        device_type = current_platform.device_type
        if (
            device_type != "cuda"
            or mm_config.get_mm_processor_device_type() != device_type
        ):
            return

        # User set video backend or device explicitly
        video_kwargs = mm_config.media_io_kwargs.setdefault("video", {})
        if "backend" in video_kwargs or "device" in video_kwargs:
            return

        from irwkv.utils.import_utils import check_torchcodec_available

        try:
            check_torchcodec_available()
        except (ImportError, RuntimeError):
            # torchcodec is not installed, or is installed without a usable
            # FFmpeg runtime (it raises rather than returning False).
            logger.info_once(
                "EPD encoder instance: keeping CPU video decoding because "
                "torchcodec is not available (needs a CUDA build with FFmpeg)."
            )
            return

        video_kwargs["backend"] = "torchcodec"
        video_kwargs["device"] = device_type
        logger.info_once(
            "EPD encoder instance: decoding video with NVDEC (torchcodec device=%s).",
            device_type,
        )

    def _validate_mm_processor_device(self) -> None:
        """Hand the EC config to `MultiModalConfig`, which owns the rule."""
        model_config = self.model_config
        if model_config is None:
            return
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return

        mm_config.validate_mm_processor_device(self.ec_transfer_config)

    def _get_v2_model_runner_unsupported_features(self) -> list[str]:
        """Collect features not yet supported by the V2 model runner."""
        unsupported: list[str] = []

        # V2 does not implement the external_launcher (torchrun) PP-output
        # broadcast that V1 uses to keep all ranks in sync (broadcast_pp_output).
        if (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and self.parallel_config.pipeline_parallel_size > 1
        ):
            unsupported.append("pipeline parallelism with external_launcher")

        if self.parallel_config.use_ubatching:
            unsupported.extend(self._get_dbo_unsupported_features())

        return unsupported

    def _validate_batch_sharded_sampling(self) -> None:
        """Validate `enable_batch_sharded_sampling` against the rest of the config."""
        if not self.parallel_config.enable_batch_sharded_sampling:
            # Default to False if not set.
            self.parallel_config.enable_batch_sharded_sampling = False
            return

        blockers: list[str] = []
        tp_size = self.parallel_config.tensor_parallel_size

        if tp_size <= 1:
            blockers.append("tensor_parallel_size is 1, so there is nothing to shard")
        elif self.scheduler_config.max_num_seqs < tp_size:
            # Requests are assigned to ranks whole, so fewer slots than ranks
            # leaves some ranks without work in every step.
            blockers.append(
                f"max_num_seqs ({self.scheduler_config.max_num_seqs}) is below "
                f"tensor_parallel_size ({tp_size})"
            )

        if self.model_config is not None and self.model_config.max_logprobs < 0:
            # max_logprobs == -1 allows vocab-size logprob requests, which the
            # fixed-width logprobs gather cannot reasonably size for.
            blockers.append("max_logprobs is -1, allowing vocab-size logprob requests")

        if self.model_config is not None and self.model_config.return_sampling_mask:
            # gather_sampler_output() drops SamplingMaskTensors: masks come back None.
            blockers.append(
                "return_sampling_mask is set and the batch-sharded gather does "
                "not forward sampling masks"
            )

        if blockers:
            raise ValueError(
                "Batch-sharded sampling was explicitly enabled via "
                "the --enable-batch-sharded-sampling flag, but is not supported "
                "in this configuration for the following reason(s): "
                f"{'; '.join(blockers)}."
            )

    def _get_dbo_unsupported_features(self) -> list[str]:
        """Collect what the V2 model runner cannot combine with DBO.

        The V2 runner microbatches a plain decoder forward pass. Anything that
        slices or replays the batch differently (drafting, adapters, pipeline
        stages, context parallelism, encoders) is not handled yet.
        """
        # TODO: DBO with model runner V2 is under development.
        # It should be enabled with explicit IRWKV_USE_V2_MODEL_RUNNER environ.
        # Remove it when stable.
        if envs.IRWKV_USE_V2_MODEL_RUNNER is None:
            return ["dual batch overlap"]

        unsupported: list[str] = []
        model_config = self.model_config
        parallel_config = self.parallel_config

        if parallel_config.pipeline_parallel_size > 1:
            unsupported.append("dual batch overlap with pipeline parallelism")
        if (
            parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1
        ):
            unsupported.append("dual batch overlap with context parallelism")
        if model_config is not None and model_config.is_hybrid:
            unsupported.append("dual batch overlap with hybrid models")
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
            # DBO captures FULL graphs only.
            unsupported.append("dual batch overlap with PIECEWISE CUDA graphs")

        return unsupported

    def _validate_profiler_config(self) -> None:
        if self.profiler_config.profiler != "proton":
            return

        from irwkv.platforms import current_platform

        if not current_platform.is_cuda():
            raise ValueError("The Proton profiler currently supports NVIDIA CUDA only")

        has_cuda_graphs = (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            or (
                self.compilation_config.cudagraph_mm_encoder
                and not (self.model_config and self.model_config.enforce_eager)
            )
        )
        if not has_cuda_graphs:
            return
        mode = self.profiler_config.proton_mode
        if mode and mode.split(":", 1)[0].lower() == "pcsampling":
            raise ValueError(
                "Proton PC sampling requires CUDA graphs to be disabled. "
                "Use --enforce-eager."
            )
        if not self.profiler_config.proton_graph_attribution:
            raise ValueError(
                "Proton profiling with CUDA graphs requires "
                "proton_graph_attribution=True to capture replayed kernels. "
                "Enable attribution or use --enforce-eager."
            )

    def _disable_cudagraphs_for_v2_stock_torch_compile(self) -> None:
        """Run stock torch.compile without CUDA graphs in Model Runner V2.

        V1 never wraps a stock-compiled model in CUDAGraphWrapper, so it runs
        without CUDA graphs. V2's CUDA graph manager would otherwise capture
        FULL graphs around the stock-compiled model, so disable them to match.
        """
        compilation_config = self.compilation_config
        if (
            compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE
            or compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        ):
            return
        logger.info_once(
            "CUDA graphs are not supported with stock torch.compile in Model "
            "Runner V2. Overriding cudagraph_mode %s to NONE.",
            compilation_config.cudagraph_mode.name,
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.max_cudagraph_capture_size = 0
        compilation_config.cudagraph_capture_sizes = []

    def _validate_v2_model_runner(self) -> None:
        """Check for features not yet supported by the V2 model runner."""
        if not HAS_TRITON:
            raise ValueError("Model Runner V2 requires Triton.")

        unsupported = self._get_v2_model_runner_unsupported_features()
        if unsupported:
            raise ValueError(
                f"Model Runner V2 does not yet support: {', '.join(unsupported)}"
            )

    def adjust_dcp_kv_cache_interleave_size(
        self, kv_cache_config: "KVCacheConfig"
    ) -> None:
        """Normalize DCP interleave size against block_size for NIXL P/D.

        Called by each worker (via ensure_kv_transfer_initialized), once it knows its
        own final block_size via kv_cache_config.
        """
        dcp_size = self.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            return
        if self.parallel_config.dcp_kv_cache_interleave_size > 1 and (
            self.parallel_config.cp_kv_cache_interleave_size
            != self.parallel_config.dcp_kv_cache_interleave_size
        ):
            self.parallel_config.cp_kv_cache_interleave_size = (
                self.parallel_config.dcp_kv_cache_interleave_size
            )
            logger.warning_once(
                "cp_kv_cache_interleave_size is overridden by dcp_kv_cache"
                "_interleave_size. And dcp-kv-cache-interleave-size will be "
                "deprecated when PCP is fully supported."
            )

        if self.kv_transfer_config is None or not self.kv_transfer_config.has_connector(
            "NixlConnector"
        ):
            return
        if not self.parallel_config._allow_auto_resolve_cp_interleave_size:
            return

        # Get the kernel block_size, but don't use resolve_kv_cache_block_size to avoid
        # scaling by dcp_size (we need the local block_size here).
        local_block_size = min(
            g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
        )
        if self.parallel_config.cp_kv_cache_interleave_size != local_block_size:
            interleave = self.parallel_config.cp_kv_cache_interleave_size
            self.parallel_config.cp_kv_cache_interleave_size = local_block_size
            logger.info_once(
                "When using PD disaggregation with DCP "
                "(decode_context_parallel_size=%d), "
                "cp_kv_cache_interleave_size is automatically adjusted "
                "from %d to block_size %d for block-level alignment.",
                dcp_size,
                interleave,
                local_block_size,
            )

    def validate_block_size(self) -> None:
        """Validate block_size against DCP and mamba constraints.

        Called after Platform.update_block_size_for_backend() has
        finalised block_size.
        """
        block_size = self.cache_config.block_size

        # Skip DCP interleave-size compatibility for NIXL P/D: the interleave
        # size is pinned to block_size by each worker.
        nixl_pd_active = (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.has_connector("NixlConnector")
        )
        if self.parallel_config.decode_context_parallel_size > 1 and not nixl_pd_active:
            assert (
                self.parallel_config.cp_kv_cache_interleave_size <= block_size
                and block_size % self.parallel_config.cp_kv_cache_interleave_size == 0
            ), (
                f"Block_size({block_size}) should be greater "
                "than or equal to and divisible by cp_kv_cache_interleave_size "
                f"({self.parallel_config.cp_kv_cache_interleave_size})."
            )
        # Mamba cache align-mode constraints
        if self.cache_config.mamba_cache_mode == "align":
            assert not self.scheduler_config.disable_chunked_mm_input, (
                "Chunked MM input is required because we need the flexibility "
                "to schedule a multiple of block_size tokens even if they are "
                "in the middle of a mm input"
            )

    @model_validator(mode="after")
    def validate_nvfp4_kv_cache_with_mla(self) -> "IrwkvConfig":
        if self.model_config is None:
            return self
        # The ds_mla layouts are MLA-only by construction; the plain nvfp4
        # layout (head_size//2 + head_size//16) does not apply to MLA.
        if (
            self.cache_config.cache_dtype.startswith("nvfp4")
            and not self.cache_config.cache_dtype.endswith("_ds_mla")
            and self.model_config.use_mla
        ):
            raise ValueError(
                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                "Attention) backends. Please use a different --kv-cache-dtype "
                "(e.g., 'fp8', 'auto', or 'nvfp4_ds_mla' with a sparse MLA "
                "backend) for MLA models such as DeepSeek."
            )
        return self

    @model_validator(mode="after")
    def validate_mamba_block_size(self) -> "IrwkvConfig":
        if self.model_config is None:
            return self
        mamba_block_size_is_set = (
            self.cache_config.mamba_block_size is not None
            and self.cache_config.mamba_block_size != self.model_config.max_model_len
        )
        if mamba_block_size_is_set and not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "--mamba-block-size can only be set with --enable-prefix-caching"
            )
        return self


_current_irwkv_config: IrwkvConfig | None = None
_current_prefix: str | None = None


@contextmanager
def set_current_irwkv_config(
    irwkv_config: IrwkvConfig, check_compile=False, prefix: str | None = None
):
    """Temporarily set the current irwkv config.
    Used during model initialization.
    We save the current irwkv config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the irwkv config to determine how to dispatch.
    """
    global _current_irwkv_config, _current_prefix
    old_irwkv_config = _current_irwkv_config
    old_prefix = _current_prefix
    from irwkv.compilation.counter import compilation_counter

    num_models_seen = compilation_counter.num_models_seen
    try:
        # Clear the compilation config cache when context changes.
        # This is needed since the old config may have been accessed
        # and cached before the new config is set.
        get_cached_compilation_config.cache_clear()

        _current_irwkv_config = irwkv_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            irwkv_config.compilation_config.custom_op_log_check()

        if (
            check_compile
            and irwkv_config.compilation_config.mode == CompilationMode.IRWKV_COMPILE
            and compilation_counter.num_models_seen == num_models_seen
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                irwkv_config.model_config.model,
            )
    finally:
        _current_irwkv_config = old_irwkv_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_irwkv_config()"""
    return get_current_irwkv_config().compilation_config


def get_current_irwkv_config() -> IrwkvConfig:
    if _current_irwkv_config is None:
        raise AssertionError(
            "Current irwkv config is not set. This typically means "
            "get_current_irwkv_config() was called outside of a "
            "set_current_irwkv_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_irwkv_config' pytest fixture from tests/conftest.py."
        )
    return _current_irwkv_config


def get_current_irwkv_config_or_none() -> IrwkvConfig | None:
    return _current_irwkv_config


T = TypeVar("T")


def get_layers_from_irwkv_config(
    irwkv_config: IrwkvConfig,
    layer_type: type[T],
    layer_names: Iterable[str] | None = None,
) -> dict[str, T]:
    """Get layers from the irwkv config.

    Args:
        irwkv_config: The irwkv config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.

    """
    forward_context = irwkv_config.compilation_config.static_forward_context
    if layer_names is None:
        layer_names = forward_context.keys()

    return {
        layer_name: layer
        for layer_name in layer_names
        if isinstance(layer := forward_context.get(layer_name), layer_type)
    }
