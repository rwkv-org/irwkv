# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

import functools
import gc
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn as nn

import irwkv.envs as envs
from irwkv.compilation.counter import compilation_counter
from irwkv.compilation.cuda_graph import CUDAGraphStat
from irwkv.compilation.wrapper import compile_model_with_stock_torch
from irwkv.config import IrwkvConfig, update_config
from irwkv.config.compilation import CompilationMode, CUDAGraphMode
from irwkv.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
)
from irwkv.forward_context import BatchDescriptor, set_forward_context
from irwkv.logger import init_logger
from irwkv.model_executor.model_loader import get_model_loader
from irwkv.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from irwkv.model_executor.models.interfaces import requires_raw_input_tokens
from irwkv.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from irwkv.model_executor.warmup.jit_warmup import JitWarmupRegistry
from irwkv.multimodal import MULTIMODAL_REGISTRY
from irwkv.platforms import current_platform
from irwkv.sequence import IntermediateTensors
from irwkv.tasks import SupportedTask
from irwkv.utils.gc_utils import freeze_gc_for_cudagraph_capture
from irwkv.utils.mem_utils import DeviceMemoryProfiler, format_gib
from irwkv.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, async_tensor_h2d
from irwkv.v1.core.sched.output import GrammarOutput, SchedulerOutput
from irwkv.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from irwkv.v1.outputs import (
    ECConnectorOutput,
    ModelRunnerOutput,
)
from irwkv.v1.watermarking import create_watermarker
from irwkv.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from irwkv.v1.worker.block_table import get_block_table_width
from irwkv.v1.worker.cp_utils import check_attention_cp_compatibility
from irwkv.v1.worker.gpu import pcp_manager as pcp
from irwkv.v1.worker.gpu.async_utils import (
    AsyncOutput,
)
from irwkv.v1.worker.gpu.attn_utils import (
    FastPrefillHelper,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from irwkv.v1.worker.gpu.block_table import BlockTables
from irwkv.v1.worker.gpu.buffer_utils import set_default_max_concurrency
from irwkv.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from irwkv.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    has_compiled_submodule,
    make_cudagraph_stats,
)
from irwkv.v1.worker.gpu.cudagraph_utils import (
    profile_cudagraph_memory as _profile_cudagraph_memory,
)
from irwkv.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from irwkv.v1.worker.gpu.ec_connector import get_ec_connector
from irwkv.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
    prepare_sampled_tokens,
    set_dummy_context,
)
from irwkv.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from irwkv.v1.worker.gpu.model_states import init_model_state
from irwkv.v1.worker.gpu.pp_utils import PPHandler
from irwkv.v1.worker.gpu.sample.batch_shard import (
    BatchSharder,
    all_to_all_logits,
    gather_sampler_output,
)
from irwkv.v1.worker.gpu.sample.logits_processor import build_custom_logits_processors
from irwkv.v1.worker.gpu.sample.output import SamplerOutput
from irwkv.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from irwkv.v1.worker.gpu.sample.sampler import Sampler
from irwkv.v1.worker.gpu.shutdown import free_before_shutdown
from irwkv.v1.worker.gpu.states import RequestState
from irwkv.v1.worker.gpu.structured_outputs import (
    StructuredOutputsWorker,
)
from irwkv.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    maybe_build_ubatch_runner,
)
from irwkv.v1.worker.utils import (
    KVBlockZeroer,
    clear_layer_kv_caches,
    copy_kv_cache_blocks_inplace,
    get_uniform_decode_token_count,
)
from irwkv.v1.worker.workspace import lock_workspace

logger = init_logger(__name__)


def _get_parameter_for_reload(model: nn.Module, name: str) -> nn.Parameter:
    """Resolve checkpoint names without changing the model's module tree."""
    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name)
    return module.get_parameter(parameter_name)


class GPUModelRunner:
    def __init__(self, irwkv_config: IrwkvConfig, device: torch.device):
        self.irwkv_config = irwkv_config
        self.model_config = irwkv_config.model_config
        self.cache_config = irwkv_config.cache_config
        self.compilation_config = irwkv_config.compilation_config
        self.load_config = irwkv_config.load_config
        self.parallel_config = irwkv_config.parallel_config
        self.scheduler_config = irwkv_config.scheduler_config
        self.observability_config = irwkv_config.observability_config
        self.jit_warmup_registry = JitWarmupRegistry(irwkv_config)

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_decoder = False

        self.output_copy_stream = torch.cuda.Stream(self.device)

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed
        set_default_max_concurrency(irwkv_config.max_concurrent_batches)

        # PP broadcast/recv helper. Runs the collective on a side stream.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Dual batch overlap. Created in initialize_kv_cache(), once everything
        # it runs the microbatched forward with exists.
        self.ubatch_runner: UBatchRunner | None = None

        # Decode context parallelism.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = False
        self.uses_inputs_embeds = self.model_config.enable_prompt_embeds
        self.encoder_cache = None
        self.ec_connector = get_ec_connector(irwkv_config, self.encoder_cache)

        # General request states.
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.vocab_size,
            device=self.device,
        )

        self.use_aux_hidden_state_outputs = False

        self.pcp_manager: pcp.PCPManager | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        self.fast_prefill: FastPrefillHelper | None = None
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                device=self.device,
            )

        # Samplers and decode_query_len created in load_model() after
        # model_state exists (num_new_sampled_tokens_per_step from ModelState).
        self.sampler: Sampler | None = None
        self.batch_sharder: BatchSharder | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        set_offloader(create_offloader(self.irwkv_config.offload_config))

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        tasks.extend(self.model_state.get_supported_generation_tasks())
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.irwkv_config.load_config)
            logger.info_once("Loading model from scratch...")

            # Capture warmup providers selected while constructing the model.
            with self.jit_warmup_registry.activate():
                self.model = model_loader.load_model(
                    irwkv_config=self.irwkv_config,
                    model_config=self.irwkv_config.model_config,
                )

        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        # Initialize the components that require the model.
        self.model_state = init_model_state(
            self.irwkv_config, self.model, self.encoder_cache, self.device
        )

        self.decode_query_len = 1

        if self.parallel_config.enable_batch_sharded_sampling:
            if hasattr(self.model, "compute_logits_local"):
                self.batch_sharder = BatchSharder(
                    max_num_reqs=self.max_num_reqs,
                    max_num_logits_per_req=self.decode_query_len,
                    device=self.device,
                )
                logger.info("Batch-sharded sampling enabled.")
            else:
                logger.warning_once(
                    "Disabling batch-sharded sampling: %s does not implement "
                    "compute_logits_local",
                    type(self.model).__name__,
                )

        # Initialize samplers. Model states may override via custom_sampler().
        config_processors = self.model_config.logits_processors or ()
        custom_logits_processors = build_custom_logits_processors(
            self.irwkv_config, self.req_states, False, config_processors
        )
        if self.is_last_pp_rank:
            # V2 bypasses TopKTopPSampler, which registers native warmups.
            # ROCm only: on CUDA these warmups add ~2 min to every engine start.
            if current_platform.is_rocm():
                from irwkv.v1.sample.ops.topk_topp_sampler import (
                    register_top_k_top_p_warmups,
                )

                with self.jit_warmup_registry.activate():
                    register_top_k_top_p_warmups()

            sampler_kwargs: dict[str, Any] = {
                "irwkv_config": self.irwkv_config,
                "max_num_reqs": self.max_num_reqs,
                "vocab_size": self.vocab_size,
                "device": self.device,
                "req_states": self.req_states,
                "logprobs_mode": self.model_config.logprobs_mode,
                "use_fp64_gumbel": self.model_config.use_fp64_gumbel,
                "enable_trace_replay": self.model_config.enable_trace_replay,
                "return_sampling_mask": self.model_config.return_sampling_mask,
                "custom_logits_processors": custom_logits_processors,
            }
            if self.irwkv_config.watermark_config is None:
                self.sampler = Sampler(**sampler_kwargs)
            else:
                wm_config = self.irwkv_config.watermark_config
                watermarker = create_watermarker(wm_config)
                self.sampler = GPUWatermarkSampler(
                    watermarker,
                    deduplicate_contexts=wm_config.deduplicate_contexts,
                    deduplicate_contexts_max_history=(
                        wm_config.deduplicate_contexts_max_history
                    ),
                    **sampler_kwargs,
                )
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                self.irwkv_config._check_supports_watermarking(custom_sampler=True)
                self.sampler = custom
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                self.device,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
            )

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        get_offloader().post_init()

        if self.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            compile_model_with_stock_torch(self.model, self.irwkv_config)

    @property
    def dp_size(self) -> int:
        return self.parallel_config.data_parallel_size

    @property
    def dp_rank(self) -> int:
        return self.parallel_config.data_parallel_rank

    def get_model(self) -> nn.Module:
        return self.model

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Reload weights from a weights iterator or from disk

        Args:
            weights_iterator: weights to load into model
            weights_path: path to load weights from if weights_iterator is not
                provided. Use path of original model if neither is provided.
            is_checkpoint_format: set to False if weights have already been
                processed into kernel format (repacking, renaming, etc.)

        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                # The revision and any object-storage `model_weights` source
                # belong to the model we are reloading away from, so they must
                # not be carried over to the new path.
                self.model_config.model = weights_path
                self.model_config.model_weights = ""
                self.model_config.revision = None
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...")
        if is_checkpoint_format:
            # load weights from checkpoint/ original model format
            initialize_layerwise_reload(model)
            loaded_weights = model.load_weights(weights_iterator)
            finalize_layerwise_reload(model, self.model_config)

        else:
            # load weights from kernel format
            logger.warning_once(
                "Reloading with `is_checkpoint_format=False` requires that "
                "weights be in kernel format and already sharded",
            )
            loaded_weights = set()
            for name, loaded_weight in weights_iterator:
                param = _get_parameter_for_reload(model, name)  # TODO: buffers?
                param.copy_(loaded_weight)
                loaded_weights.add(name)

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
        )
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

        self.reset_encoder_cache()
        self.reset_mm_cache()

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            if config_name not in allowed_config_names:
                allowed = ", ".join(sorted(allowed_config_names))
                raise ValueError(
                    f"Config override '{config_name}' is not supported. "
                    f"Supported configs: {allowed}"
                )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

        self.irwkv_config.model_config = self.model_config
        self.irwkv_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        return torch.cuda.current_stream(self.device)

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        encoder_runner = getattr(self.model_state, "encoder_runner", None)
        return encoder_runner.get_encoder_timing_stats() if encoder_runner else {}

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.irwkv_config)

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # GPUWorker finalizes the PD interleave before KV cache initialization.
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        dcp_sharded = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
            dcp_sharded.append(spec.dcp_sharded)
            # Let each cache type account for CP. Attention KV is DCP-sharded,
            # while Mamba/GDN recurrent state is replicated across DCP ranks.
            max_num_blocks = spec.max_num_blocks_per_req(
                self.irwkv_config, block_table_max_model_len
            )
            # Preserve each cache type's alignment requirements after applying
            # its topology-aware block-table width.
            max_num_blocks = get_block_table_width(
                max_num_blocks,
                spec.block_size,
                token_alignment=spec.block_table_token_alignment,
            )
            max_num_blocks_per_group.append(max_num_blocks)

        # Metadata builders select attention kernels that need JIT warmup.
        with self.jit_warmup_registry.activate():
            (
                self.attn_groups,
                attn_cg_support,
                self.kernel_block_sizes,
            ) = init_attn_backend(
                self.kv_cache_config,
                self.irwkv_config,
                self.device,
            )
        additional_attn_cg_support = self.model_state.get_additional_cg_support()
        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            slot_mapping_enabled=slot_mapping_enabled,
            dcp_sharded=dcp_sharded,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.irwkv_config,
            self.device,
            False,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        self.ubatch_runner = maybe_build_ubatch_runner(
            self.irwkv_config,
            self.device,
            self.model_state,
            self.attn_groups,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        piecewise_capture_available = bool(
            envs.IRWKV_USE_BREAKABLE_CUDAGRAPH or has_compiled_submodule(self.model)
        )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
            piecewise_capture_available=piecewise_capture_available,
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.irwkv_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            ubatch_runner=self.ubatch_runner,
            requires_max_query_len=(self.model_state.requires_cudagraph_max_query_len),
        )
        if self.cache_config.kv_sharing_fast_prefill and self.pcp_manager is None:
            self.fast_prefill = FastPrefillHelper(
                self.cudagraph_manager, self.max_num_tokens
            )
        check_attention_cp_compatibility(self.irwkv_config, None)
        # Capture warmup providers that depend on allocated KV-cache strides.
        with self.jit_warmup_registry.activate():
            kv_caches_dict = init_kv_cache(
                self.compilation_config.static_forward_context,
                self.kv_cache_config,
                self.device,
                self.kernel_block_sizes,
                self.irwkv_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
                block_tables=self.block_tables,
                attn_groups=(g for groups in self.attn_groups for g in groups),
            )
        self.kv_caches = [
            cache for cache in kv_caches_dict.values() if cache.device == self.device
        ]
        self.model_state.initialize_kv_cache(self.kv_cache_config, self.block_tables)
        if is_profiling:
            self.kv_connector = NO_OP_KV_CONNECTOR
        else:
            self.kv_connector = get_kv_connector(self.irwkv_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        context_len: int = 0,
        is_profile: bool = False,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        # Distribute the remainder evenly so no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.

        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        self.execute_model(
            dummy_scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=True,
            skip_attn_for_dummy_run=skip_attn,
            is_profile=is_profile,
            context_len=context_len,
            valid_dummy_state_slots=valid_dummy_state_slots,
            randomize_inputs=randomize_inputs,
        )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        self.execute_model_state = None

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def profile_run(self, randomize_inputs: bool = False) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens,
            skip_attn=True,
            is_profile=True,
            randomize_inputs=randomize_inputs,
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            self._dummy_sampler_run(sample_hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        self.reset_encoder_cache()
        gc.collect()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        """Estimate the GPU memory required to capture CUDA graphs."""
        return _profile_cudagraph_memory(self)

    def needs_cudagraph_capture(self) -> bool:
        """Whether capture_model() has any CUDA graphs left to capture."""
        return (
            self.cudagraph_manager is not None
            and self.cudagraph_manager.needs_capture()
        )

    @torch.inference_mode()
    def capture_model(self, *, profile_only: bool = False) -> int:
        assert self.cudagraph_manager is not None
        capture_encoder = False
        capture_decoder = self.cudagraph_manager.needs_capture()
        if not capture_encoder and not capture_decoder:
            logger.warning(
                "Skipping encoder and decoder CUDA graph capture. To enable "
                "encoder capture, ensure `cudagraph_mm_encoder` is enabled; "
                "to enable decoder capture, ensure `cudagraph_mode` is not `NONE`."
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        with freeze_gc_for_cudagraph_capture():
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

            if capture_encoder:
                self.model_state.encoder_runner.capture()

            if capture_decoder:
                input_buffers = self.input_buffers
                if self.pcp_manager is not None:
                    input_buffers = self.pcp_manager.input_buffers
                self.cudagraph_manager.capture(
                    self.model,
                    self.model_state,
                    input_buffers,
                    self.intermediate_tensors,
                    self.block_tables,
                    self.attn_groups,
                    self.kv_cache_config,
                    pcp_manager=self.pcp_manager,
                    use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                )
                self.model_state.capture_inner_cudagraphs(
                    input_buffers,
                    self.block_tables,
                    self.attn_groups,
                    self.kv_cache_config,
                )
                self.kv_connector.reset_capture_state()

            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        if not profile_only:
            # Lock workspace to prevent resizing during execution. A resize after
            # capture frees the static cuda graph buffer.
            lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1073741824),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # Call model_state.remove_request *before* req_states.remove_request
        # so the model_state can still look up the slot index.
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            for req_id in sorted(preempted_req_ids):
                self.model_state.preempt_request(req_id)
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        # Sorted so every TP rank frees request slots in the same order.
        # Features like batch-sharded sampling derive rank request ownership
        # from the slot index.
        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input updates replace an existing request slot.
            if req_id in self.req_states.req_id_to_index:
                self._remove_request(req_id)

            prompt_len = new_req_data.prompt_len
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(req_index, new_req_data.sampling_params)
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            block_ids = scheduler_output.new_block_ids_to_zero
            if not self.model_state.reset_kv_cache_blocks(block_ids):
                assert self.kv_block_zeroer is not None
                self.kv_block_zeroer.zero_block_ids(block_ids)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            block_copies = scheduler_output.kv_cache_block_copies
            if not self.model_state.copy_kv_cache_blocks(block_copies):
                copy_kv_cache_blocks_inplace(
                    self.kv_caches,
                    self.kv_cache_config.num_blocks,
                    block_copies,
                )

    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ) -> tuple["BatchReqState | None", int | None]:
        """Gather CPU request state for the scheduled batch, in batch order.
        Returns (batch_state, uniform_decode_token_count)
        """
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())

        if dummy_run:
            # Dummy batches are uniform by construction.
            return None, get_uniform_decode_token_count(
                num_reqs, num_toks, max_query_len, decode_graph_eligible=True
            )

        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

        numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
        has_prefill = bool(is_prefilling_np.any())
        max_seq_len_np = None
        if self.use_pp and has_prefill:
            # consumed by the PP `compute_need_sampled_mask` for batches with prefill.
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        prefill_runs_as_decode_np = None
        decode_graph_eligible = not has_prefill
        if has_prefill and self.pcp_manager is None:
            num_new_tokens = num_scheduled_tokens
            one_token_ext = (num_new_tokens == 1) & (num_computed_prefill_tokens_np > 0)
            prefill_runs_as_decode_np = is_prefilling_np & one_token_ext
            decode_graph_eligible = bool(
                (prefill_runs_as_decode_np | ~is_prefilling_np).all()
            )

        batch_state = BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_toks,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            has_prefill=has_prefill,
            prefill_runs_as_decode_np=prefill_runs_as_decode_np,
            decode_graph_eligible=decode_graph_eligible,
            max_seq_len_np=max_seq_len_np,
        )
        return batch_state, get_uniform_decode_token_count(
            num_reqs, num_toks, max_query_len, decode_graph_eligible
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: "BatchReqState",
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        assert num_tokens > 0
        is_padding = self.input_buffers.is_padding[:num_tokens_after_padding]

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_tensor_h2d(
            idx_mapping_np, device=self.device, dtype=torch.int64
        )
        num_reqs = len(req_ids)

        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        cu_num_logits = torch.arange(
            num_reqs + 1, device=self.device, dtype=torch.int32
        )
        expanded_idx_mapping = idx_mapping
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )

        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = self.input_buffers.query_start_loc
        async_tensor_h2d(query_start_loc_np, out=query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]

        # Get prefill tokens if any.
        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        logits_indices = prepare_sampled_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
        )

        fast_prefill = None
        if self.fast_prefill is not None:
            fast_prefill = self.fast_prefill.prepare(
                logits_indices,
                num_reqs,
                cu_num_logits_np,
                batch_req_state.has_prefill,
                batch_desc,
            )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            prefill_runs_as_decode_np=batch_req_state.prefill_runs_as_decode_np,
            decode_graph_eligible=batch_req_state.decode_graph_eligible,
            max_seq_len_np=batch_req_state.max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=is_padding,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
            fast_prefill=fast_prefill,
            max_query_len=batch_desc.max_query_len,
        )
        if self.pcp_manager is not None:
            input_batch = self.pcp_manager.partition_batch(input_batch, batch_desc)
        return input_batch

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is not None:
            return self.pcp_manager.prepare_attn(input_batch)

        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping, num_reqs_padded=input_batch.num_reqs_after_padding
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch, valid_state_slots: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_table_provider = self.pcp_manager or self.block_tables
        block_tables = block_table_provider.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
            state_slots = torch.arange(
                1, input_batch.num_reqs + 1, dtype=torch.int32, device=self.device
            )
            for block_table in block_tables:
                block_table[:, 0].copy_(state_slots)
        slot_mappings = block_table_provider.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        shard_metadata = None
        global_input_batch = input_batch
        if self.batch_sharder is not None:
            # Shard the inputs along the batch dimension to sample in parallel
            # across TP ranks.
            input_batch, sorted_logits_indices, grammar_output, shard_metadata = (
                self.batch_sharder.shard_sampler_inputs(input_batch, grammar_output)
            )
            # The hidden states must be gathered in rank-owner-sorted order
            # before computing the partial-vocab logits, so that the all-to-all
            # produces full-vocab logits for just the locally-owned requests.
            sample_hidden_states = hidden_states[sorted_logits_indices]
            local_logits = self.model.compute_logits_local(sample_hidden_states)
            logits = all_to_all_logits(local_logits, shard_metadata)
            logits = logits[:, : self.vocab_size]
        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            custom_output = self.model_state.custom_sample(
                self.model,
                sample_hidden_states,
                input_batch,
                grammar_output,
            )
            if custom_output is not None:
                return (
                    custom_output,
                    custom_output.num_sampled,
                    custom_output.num_rejected,
                )
            logits = self.model.compute_logits(sample_hidden_states)

        if grammar_output is not None and logits.shape[0] > 0:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        sampler_output: SamplerOutput | None
        if input_batch.num_reqs == 0:
            # This rank owns no requests this step. It contributes an
            # all-padding block to the gather below.
            sampler_output = None
        else:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)

        if shard_metadata is not None:
            # Gather the sharded sampler outputs from the TP ranks into a single
            # sampler output.
            assert self.sampler is not None
            sampler_output = gather_sampler_output(
                sampler_output,
                shard_metadata,
                device=self.device,
                global_batch=global_input_batch,
                local_batch=input_batch,
                gather_num_nans=self.sampler.compute_nans,
                logprobs_dims=self.sampler.get_logprobs_dims(
                    global_input_batch.idx_mapping_np,
                ),
            )

        assert sampler_output is not None
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        output_bin_counts = None
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    def _merge_ec_connector_no_forward(
        self, scheduler_output: SchedulerOutput, output: ModelRunnerOutput
    ) -> ModelRunnerOutput:
        """Let the EC connector send/recv on a step with no work to run."""
        return ModelRunnerOutput.with_ec_conn_output(
            output,
            self.ec_connector.no_forward(scheduler_output).ec_connector_output,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return self._merge_ec_connector_no_forward(
                    scheduler_output, empty_output
                )

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len: int | None = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(
            scheduler_output, dummy_run
        )
        if batch_req_state is not None:
            num_toks = batch_req_state.num_tokens
            if (
                batch_req_state.has_prefill
                and not self.model_state.requires_cudagraph_max_query_len
            ):
                # Varlen decode graphs replay decodes only, and their bound
                # alone would admit a short prefill.
                max_query_len = None
            if self.pcp_manager is not None:
                num_toks = self.pcp_manager.get_num_tokens_for_dispatch(
                    batch_req_state.num_scheduled_tokens,
                    batch_req_state.is_prefilling_np,
                )

        skip_compiled = False

        batch_desc, dp_sync_state = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            max_query_len=max_query_len,
            need_eager=is_profile or skip_compiled,
            parallel_config=self.parallel_config,
            allow_ubatching=(
                self.ubatch_runner is not None and not skip_attn_for_dummy_run
            ),
            uniform_decode=uniform_tok_count == self.decode_query_len,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)

        cudagraph_stats = None
        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            if self.observability_config.cudagraph_metrics:
                cudagraph_stats = make_cudagraph_stats(batch_desc, num_toks)
            assert batch_req_state is not None
            input_batch = self.prepare_inputs(
                scheduler_output, batch_req_state, batch_desc
            )
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # Mamba "align" pre-copy: migrate recurrent state across block
            # boundaries before the forward. Runs only on real batches, and
            # before model_state.prepare_attn gathers num_accepted_tokens so the
            # boundary reset is visible to the attention metadata.
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            dummy_num_reqs = batch_desc.num_reqs or num_reqs
            input_batch = InputBatch.make_dummy(
                dummy_num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
                max_query_len=batch_desc.max_query_len,
                is_padding=not is_profile,
            )
            if randomize_inputs:
                input_batch.input_ids.random_(0, self.vocab_size)
            if self.pcp_manager is not None:
                input_batch = self.pcp_manager.prepare_inputs_to_capture(input_batch)
            if skip_attn_for_dummy_run:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables, slot_mappings = None, None
            else:
                block_tables, slot_mappings = self.prepare_dummy_attn(
                    input_batch, valid_dummy_state_slots
                )
                if context_len:
                    set_dummy_context(
                        input_batch,
                        self.block_tables,
                        context_len,
                        self.kv_cache_config.num_blocks,
                        self.max_model_len,
                        input_block_tables=block_tables,
                    )

        if self.dcp_size > 1 and not (dummy_run and skip_attn_for_dummy_run):
            input_buffers = self.input_buffers
            if self.pcp_manager is not None:
                input_buffers = self.pcp_manager.input_buffers
            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                input_batch.num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
                num_reqs_padded=input_batch.num_reqs_after_padding,
            )

        attn_metadata = None
        slot_mappings_by_layer = None
        ubatch_state: UBatchState | None = None
        if batch_desc.num_ubatches > 1:
            assert self.ubatch_runner is not None
            assert block_tables is not None and slot_mappings is not None
            ubatch_state = self.ubatch_runner.prepare(
                input_batch,
                block_tables,
                slot_mappings,
                cg_mode=batch_desc.cg_mode,
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )
        elif not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_groups = self.attn_groups
            if dummy_run and is_profile:
                # Mamba layers take a cheap warmup path with no metadata;
                # attention metadata is still built so those kernels tune.
                attn_groups = [
                    [g for g in groups if not isinstance(g.kv_cache_spec, MambaSpec)]
                    for groups in attn_groups
                ]
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                self.kv_cache_config,
                # FULL replay reads capture-time metadata buffers. Re-stage them
                # from the zeroed dummy block tables instead of retaining state
                # indices from the previous real batch.
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
        if self.uses_inputs_embeds and self.is_first_pp_rank:
            # Prepare inputs_embeds (MM encoder outputs and/or prompt_embeds
            # overlay). Only first PP rank prepares them.
            if dummy_run:
                # Obtain embeddings of correct shape for compiled model.
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    inputs_embeds = self.model_state.prepare_inputs_embeds(
                        scheduled_encoder_inputs, input_batch, self.req_states
                    )
            if inputs_embeds is not None and not requires_raw_input_tokens(self.model):
                input_ids = None

        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        ubatch_slices = ubatch_state.slices if ubatch_state is not None else None

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(
                scheduler_output, input_batch, attn_metadata=attn_metadata
            )
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
            )

            with set_forward_context(
                attn_metadata,
                self.irwkv_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=dp_sync_state
                and dp_sync_state.num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
            ):
                self.kv_connector.pre_forward(scheduler_output, input_batch)
                if ubatch_state is not None:
                    assert self.ubatch_runner is not None
                    model_output = self.ubatch_runner.run(
                        self.model, model_inputs, ubatch_state
                    )
                elif batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager (NONE): call the raw model directly.
                    model_output = self.model(**model_inputs)

        self.kv_connector.finish_forward()

        if self.is_last_pp_rank:
            assert isinstance(model_output, torch.Tensor)
            hidden_states = model_output
            aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            dp_sync_state=dp_sync_state,
            finished_req_ids=finished_req_ids,
            ec_connector_output=ec_connector_output,
            cudagraph_stats=cudagraph_stats,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            assert output_intermediate_tensors is not None
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        cudagraph_stats = self.execute_model_state.cudagraph_stats
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            if self.pcp_manager is not None:
                input_batch = self.pcp_manager.global_batch
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            # The first PP rank holds the encoder cache, so pass its EC output on.
            output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        # Last rank: sample tokens
        assert hidden_states is not None
        if self.pcp_manager is not None:
            hidden_states, aux_hidden_states, input_batch = (
                self.pcp_manager.restore_for_sampling(hidden_states, aux_hidden_states)
            )

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )
        prompt_token_id_logprobs_dict = (
            self.prompt_logprobs_worker.compute_prompt_token_id_logprobs(
                self.model.compute_logits,
                hidden_states,
                input_batch,
                self.req_states.prompt_len.np,
            )
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            prompt_token_id_logprobs_dict=prompt_token_id_logprobs_dict,
            cudagraph_stats=cudagraph_stats,
        )

        # Start the async output copy before updating request states.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        torch.accelerator.synchronize()
        set_offloader(None)
        self.cudagraph_manager = None
        self.fast_prefill = None
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        free_before_shutdown(self.irwkv_config)
        if hasattr(self, "model_state"):
            del self.model_state
        # Detach the layer-level KV/state cache tensors before dropping the
        # models; the model objects can outlive this runner.
        if hasattr(self, "model"):
            clear_layer_kv_caches(self.model.modules())
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    # Out-of-tree hardware runners can select a PCP manager class.
    @property
    def pcp_manager_cls(self) -> type[pcp.PCPManager]:
        return pcp.PCPManager


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    dp_sync_state: DPSyncState | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None


class BatchReqState(NamedTuple):
    """CPU request state for a scheduled batch, in batch (sorted) order."""

    req_ids: list[str]
    num_scheduled_tokens: np.ndarray  # [num_reqs]
    num_tokens: int
    idx_mapping_np: np.ndarray  # [num_reqs]
    prefill_len_np: np.ndarray  # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray  # [num_reqs]
    is_prefilling_np: np.ndarray  # [num_reqs]
    max_seq_len_np: np.ndarray | None  # [num_reqs], None in non-pp case
    has_prefill: bool
    prefill_runs_as_decode_np: np.ndarray | None  # [num_reqs]
    decode_graph_eligible: bool


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    decode_query_len: int,
) -> list[str]:
    # Order decode -> short_extend -> prefill;
    # split_decodes_and_prefills relies on decode-like requests leading.
    key = lambda r: (
        (num := num_tokens_per_req[r]) != decode_query_len,
        num,
    )
    return sorted(num_tokens_per_req, key=key)
