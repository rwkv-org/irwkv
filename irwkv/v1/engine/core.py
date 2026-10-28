# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast, get_args

import msgspec
import zmq

from irwkv.config import IrwkvConfig, ParallelConfig
from irwkv.distributed import (
    cleanup_dist_env_and_memory,
)
from irwkv.envs import enable_envs_cache
from irwkv.logger import configure_logging, init_logger
from irwkv.logging_utils.dump_input import dump_engine_exception
from irwkv.multimodal.cache import (
    MultiModalCacheMissError,
    engine_receiver_cache_from_config,
)
from irwkv.tasks import SupportedTask
from irwkv.tracing import instrument, maybe_init_worker_tracer
from irwkv.transformers_utils.config import maybe_register_config_serialize_by_value
from irwkv.utils import numa_utils
from irwkv.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from irwkv.utils.hashing import get_hash_fn_by_name
from irwkv.utils.network_utils import make_zmq_socket
from irwkv.utils.system_utils import decorate_logs, set_process_title
from irwkv.v1.attention.backends.utils import resolve_kv_cache_layout
from irwkv.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
    resolve_cache_hit_alignment_tokens,
    resolve_kv_cache_block_sizes,
    update_kv_cache_capacity,
)
from irwkv.v1.core.sched.interface import PauseState, SchedulerInterface
from irwkv.v1.core.sched.output import SchedulerOutput
from irwkv.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from irwkv.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    UtilityOutput,
    UtilityResult,
)
from irwkv.v1.engine.tensor_ipc import TensorIpcReceiver
from irwkv.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_physical_gpu_ids_for_local_dp_rank,
)
from irwkv.v1.executor import Executor
from irwkv.v1.kv_cache_interface import (
    KVCacheConfig,
    get_kv_cache_spec_kind,
    is_full_attention_spec,
)
from irwkv.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
from irwkv.v1.outputs import ModelRunnerOutput
from irwkv.v1.request import Request, RequestStatus
from irwkv.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from irwkv.v1.structured_output import StructuredOutputManager
from irwkv.v1.utils import compute_iteration_details
from irwkv.version import __version__ as IRWKV_VERSION

logger = init_logger(__name__)


HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of irwkv's Engine."""

    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from irwkv.plugins import load_general_plugins

        load_general_plugins()

        self.irwkv_config = irwkv_config
        if not irwkv_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                IRWKV_VERSION,
                irwkv_config,
            )

        self.log_stats = log_stats
        # Opaque weight version supplied by the caller.
        self._weight_version = "default"

        # Setup Model.
        self.model_executor = executor_class(irwkv_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(irwkv_config)
        self.structured_output_manager = StructuredOutputManager(irwkv_config)

        # Setup scheduler.
        Scheduler = irwkv_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if irwkv_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                irwkv_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, irwkv_config
        )

        self.scheduler: SchedulerInterface = Scheduler(
            irwkv_config=irwkv_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        self._initialize_effective_attention_block_size()
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
        if self.scheduler.ec_connector is not None:  # type: ignore
            self.model_executor.init_ec_output_aggregator()

        self.mm_receiver_cache = engine_receiver_cache_from_config(irwkv_config)

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {(pp_rank, tp_rank): metadata}
                # Merge all worker dicts into a single dict
                content: dict[tuple[int, int], Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata_pp_aware(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = irwkv_config.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_mm_encoder_only = False

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if irwkv_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                irwkv_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = irwkv_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, irwkv_config: IrwkvConfig) -> KVCacheConfig:
        start = time.time()

        # register all kvcache specs in enginecore process.
        register_all_kvcache_specs(irwkv_config)

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Some layers (e.g. Prefix LM attention) run non-causally and tag their
        # KV cache spec with ``non_causal=True``. The specs are collected here in
        # the engine-core process (the same process that builds the scheduler),
        # so this is the multiproc-safe place to translate that layer-level
        # signal into a scheduling policy: chunked prefill and prefix caching
        # both assume causal attention and would corrupt non-causal prefill.
        if any(
            getattr(spec, "non_causal", False)
            for worker_specs in kv_cache_specs
            for spec in worker_specs.values()
        ):
            if irwkv_config.scheduler_config.enable_chunked_prefill:
                logger.info(
                    "Disabling chunked prefill: model has non-causal attention layers."
                )
                irwkv_config.scheduler_config.enable_chunked_prefill = False
            if irwkv_config.cache_config.enable_prefix_caching:
                logger.info(
                    "Disabling prefix caching: model has non-causal attention layers."
                )
                irwkv_config.cache_config.enable_prefix_caching = False

        # Resolve the KV cache layout before memory profiling: workers that
        # capture full cudagraphs initialize a minimal KV cache during it.
        # Attention-free models resolve the default so layout reads never precede
        # resolution.
        layout = resolve_kv_cache_layout(
            irwkv_config,
            self.model_executor.get_supported_kv_cache_layouts(),
            [s for specs in kv_cache_specs for s in specs.values()],
        )
        self.model_executor.set_kv_cache_layout(layout.name)

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # Profiles the peak memory usage of the model to determine how much
            # memory can be allocated for KV cache.
            available_gpu_memory = self.model_executor.determine_available_memory()
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = irwkv_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            irwkv_config, kv_cache_specs, available_gpu_memory
        )
        for kv_cache_config in kv_cache_configs:
            kv_cache_config.kv_cache_layout = irwkv_config.cache_config.kv_cache_layout

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = irwkv_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        irwkv_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            # Exclude groups that opt out of prefix caching (e.g. a circular
            # buffer, a 1-block/req scratch buffer with a small ring block):
            # their small block_size would otherwise drag the global block_size
            # below the real allocator block size and desync it from mamba.
            participating = [
                g.kv_cache_spec.block_size
                for g in kv_cache_groups
                if g.kv_cache_spec.prefix_cacheable
            ]
            irwkv_config.cache_config.block_size = min(
                participating
                if participating
                else [g.kv_cache_spec.block_size for g in kv_cache_groups]
            )
            update_kv_cache_capacity(irwkv_config, scheduler_kv_cache_config)

        irwkv_config.validate_block_size()

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            scheduler_kv_cache_config, irwkv_config
        )
        cache_hit_alignment_tokens = resolve_cache_hit_alignment_tokens(
            scheduler_kv_cache_config,
            irwkv_config,
            scheduler_block_size,
            hash_block_size,
        )
        # The scheduler's config is a deep copy taken before these are resolved,
        # so set them there too: the KV connector's scheduler side reads the same
        # fields as its workers.
        for kv_cache_config in (*kv_cache_configs, scheduler_kv_cache_config):
            kv_cache_config.hash_block_size = hash_block_size
            kv_cache_config.cache_hit_alignment_tokens = cache_hit_alignment_tokens

        self.model_executor.initialize_from_config(kv_cache_configs)
        self.model_executor.compile_or_warm_up_model()

        elapsed = time.time() - start
        compile_time = irwkv_config.compilation_config.compilation_time
        encoder_compile_time = irwkv_config.compilation_config.encoder_compilation_time
        if encoder_compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,
                compile_time + encoder_compile_time,
                compile_time,
                encoder_compile_time,
            )
        elif compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,
                compile_time,
            )
        else:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,
            )
        return scheduler_kv_cache_config

    def _initialize_effective_attention_block_size(self) -> None:
        cache_config = self.irwkv_config.cache_config
        cache_config.effective_attention_block_size = None
        cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        if cache_manager is None:
            return
        block_sizes = {
            manager.block_size
            for manager in cache_manager.coordinator.single_type_managers
            if is_full_attention_spec(manager.kv_cache_spec)
        }
        cache_config.effective_attention_block_size = (
            block_sizes.pop() if len(block_sizes) == 1 else None
        )

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        """Return serializable KV cache metadata for external event consumers
        (e.g., Dynamo). This function has no call sites within irwkv but needs to
        be kept.
        """
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        kv_cache_manager = cast(Any, self.scheduler).kv_cache_manager
        managers = kv_cache_manager.coordinator.single_type_managers
        metadata: list[dict[str, int | str | None]] = []
        for group_idx, (group, manager) in enumerate(
            zip(kv_cache_config.kv_cache_groups, managers, strict=True)
        ):
            spec = group.kv_cache_spec
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": get_kv_cache_spec_kind(spec).value,
                    "block_size": manager.block_size,
                    "sliding_window": getattr(spec, "sliding_window", None),
                }
            )
        return metadata

    def add_request(self, request: Request):
        """Add request to the scheduler."""
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        if (
            request.ec_transfer_params is not None
            and self.scheduler.get_ec_connector() is None
        ):
            logger.warning(
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        self.scheduler.add_request(request)
        if request.abort_immediately:
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            self.abort_requests([request.request_id])

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.irwkv_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def capture_iteration_details(
        self, scheduler_output: SchedulerOutput | None
    ) -> Generator[SchedulerIterationDetails | None, None, None]:
        enable_details = (
            self.irwkv_config.observability_config.enable_logging_iteration_details
        )
        if not self.log_stats or not enable_details:
            yield None
            return
        # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
        if (
            scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens == 0
        ):
            yield None
            return

        iteration_index = getattr(self, "_iteration_index", 0)
        # scheduler_output=None marks a DP dummy iteration.
        if scheduler_output is None:
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=0,
                num_ctx_tokens=0,
                num_generation_requests=0,
                num_generation_tokens=0,
                elapsed_ms=0.0,
                is_dummy=True,
            )
        else:
            details = compute_iteration_details(scheduler_output)
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=details.num_ctx_requests,
                num_ctx_tokens=details.num_ctx_tokens,
                num_generation_requests=details.num_generation_requests,
                num_generation_tokens=details.num_generation_tokens,
                elapsed_ms=0.0,
                num_encoder_inputs=details.num_encoder_inputs,
                num_encoder_output_tokens=details.num_encoder_output_tokens,
            )

        start_time = time.monotonic()
        yield iteration_details
        iteration_details.elapsed_ms = (time.monotonic() - start_time) * 1000
        self._iteration_index = iteration_index + 1

    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        stats = self.scheduler.make_stats() or SchedulerStats()
        stats.iteration_details = iteration_details
        return stats

    def _attach_iteration_details(
        self,
        outputs: dict[int, EngineCoreOutputs],
        iteration_details: SchedulerIterationDetails | None,
    ) -> None:
        if iteration_details is None:
            return

        if (eco := next(iter(outputs.values()), None)) is None:
            outputs[0] = eco = EngineCoreOutputs()
        if eco.scheduler_stats is None:
            eco.scheduler_stats = self._make_iteration_details_stats(iteration_details)
        else:
            eco.scheduler_stats.iteration_details = iteration_details

    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        gc.unfreeze()
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        cleanup_dist_env_and_memory()
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ):
        self.model_executor.profile(
            is_start,
            profile_prefix,
            delay_iterations=delay_iterations,
            max_iterations=max_iterations,
        )

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(
        self,
        reset_running_requests: bool = True,
        reset_connector: bool = True,
    ) -> None:
        # reset_connector=True so external connectors clear alongside
        # local caches, matching the pause_generation(clear_cache=True)
        # contract. No-op when no connector is configured.
        if not self.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        ):
            raise RuntimeError("Failed to reset the KV connector cache.")
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def _finish_pause(self, clear_cache: bool) -> None:
        # A completed pause promises an idle device: nothing else waits on
        # the last dummy batch an idle DP rank launches.
        self.model_executor.collective_rpc("synchronize_device")
        if clear_cache:
            self._reset_caches()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in get_args(PauseMode):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        self._finish_pause(clear_cache)

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.

        """
        # Pause scheduler before sleeping.
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        if pause_future is None:
            model_executor.sleep(level)
            return None

        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.

        Returns:
            Whether all executor memory is resident again (fully awake).

        """
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Partial wakes intentionally keep the remaining allocations asleep.
        # Resume scheduling only once all executor memory is resident again.
        fully_awake = not self.model_executor.is_sleeping
        if fully_awake:
            self.resume_scheduler()
        return fully_awake

    def release_kv_cache_memory(self) -> None:
        """Discard KV cache physical memory. Requires a completed pause
        and all executor memory to be resident. Kept requests are recomputed
        after wake-up.
        """
        if not (
            self.is_scheduler_paused()
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            raise RuntimeError(
                "release_kv_cache_memory() requires a completed pause first"
            )
        if self.model_executor.is_sleeping:
            raise RuntimeError(
                "release_kv_cache_memory() requires all executor memory to be resident"
            )
        self._reset_caches()
        self.model_executor.discard(("kv_cache",))

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def compute_weight_checksums(self) -> list[dict[str, str]]:
        return self.collective_rpc("compute_weight_checksums")

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def set_weight_version(self, weight_version: str) -> None:
        self._weight_version = weight_version

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self._weight_version

    def preprocess_add_request(self, request: EngineCoreRequest) -> Request:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            irwkv_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not irwkv_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing
            self.last_counts = (0, 0)

            self.addresses = addresses
            self.process_input_queue_block = True

            super().__init__(
                irwkv_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        irwkv_config: IrwkvConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            irwkv_config,
        )
        if client_handshake_address is None:
            # We only need to handshake with one party.
            with handshake as addresses:
                yield addresses
        else:
            # We need to handshake with rank 0 front-end and our colocated frontend.
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, irwkv_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        irwkv_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        irwkv_config: IrwkvConfig,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(handshake_socket, local_client, headless)
            yield addresses

            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # Include config hash for DP configuration validation
            if irwkv_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    irwkv_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""
        irwkv_config: IrwkvConfig = kwargs["irwkv_config"]
        if logging_config := getattr(irwkv_config, "logging_config", None):
            configure_logging(logging_config)

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        clean_shutdown = False
        try:
            parallel_config: ParallelConfig = irwkv_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("irwkv.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            if data_parallel and irwkv_config.kv_transfer_config is not None:
                # modify the engine_id and append the dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                irwkv_config.kv_transfer_config.engine_id = (
                    f"{irwkv_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    irwkv_config.kv_transfer_config.engine_id,
                )

            # declare engine core as None and register signal handler before
            # engine_core initialization in case engine core process is terminated
            # in the middle of initialization. For example, frontend process exits
            # unexpectedly, if signal handler is not registered, engine core process
            # will use default signal handler which will terminate the process
            # silently. If engine core model executor has been initialized, it will
            # not be terminated asap. Register signal handler early and only log a
            # message will make engine core process alive. engine core process and
            # its subprocesses(model executor processes) will be force killed in
            # CoreEngineProcManager.shutdown.
            engine_core = None
            # A termination signal can arrive while engine_core is still being
            # constructed. At that point the request cannot be recorded on
            # engine_core, so keep it in its own flag and apply it later.
            shutdown_requested = False

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                assert engine_core is not None
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            def signal_handler(signum, frame):
                nonlocal shutdown_requested
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                # Record the request first so it survives the construction window.
                shutdown_requested = True
                if engine_core is not None:
                    engine_core.shutdown_state = EngineShutdownState.REQUESTED
                # signal_callback is armed only after engine_core exists. Don't
                # trigger it earlier: it is one-shot, and a wakeup cannot be
                # delivered while there is no input_queue to wake.
                # For more info: https://github.com/vllm-project/vllm/pull/52299#pullrequestreview-5370255271
                if signal_callback is not None:
                    signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            parallel_config.data_parallel_index = dp_rank
            parallel_config.reconfigure_for_independent_dp_rank()
            engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            # Apply a signal received during construction before entering the
            # loop. run_busy_loop calls _handle_shutdown first, so the loop exits
            # right away instead of blocking on an idle input_queue.
            if shutdown_requested:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED

            # Arm the wakeup callback only now that input_queue exists, so a
            # later signal can still wake an idle busy loop.
            signal_callback = SignalCallback(wakeup_engine)

            engine_core.run_busy_loop()

        except SystemExit as e:
            logger.info_once("[shutdown] EngineCore: exiting busy loop")
            clean_shutdown = (
                e.code in (None, 0)
                and engine_core is not None
                and engine_core.shutdown_state == EngineShutdownState.SHUTTING_DOWN
                and not engine_core.has_work()
                and engine_core.irwkv_config.shutdown_timeout == 0
            )
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()
            if clean_shutdown:
                from irwkv.platforms import current_platform

                if current_platform.is_rocm():
                    # Cleanup above already unfreezes and collects the heap.
                    # Freeze the surviving graph to skip another slow cyclic-GC
                    # scan during finalization; process exit reclaims it.
                    gc.freeze()

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return self.scheduler.has_requests() or bool(self.batch_queue)

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            self._maybe_publish_request_counts()

        raise SystemExit

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, kv_cache_usage=self.scheduler.get_kv_cache_usage()
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.irwkv_config.shutdown_timeout
            mode = "abort" if shutdown_timeout == 0 else "drain"

            logger.info(
                "[shutdown] EngineCore: start mode=%s timeout=%ds",
                mode,
                shutdown_timeout,
            )

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: aborting in-flight requests count=%d",
                        num_requests,
                    )
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: draining in-flight requests "
                        "count=%d timeout=%ds",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False

        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""
        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.debug(
            "[shutdown] EngineCore: rejecting new request request_id=%s",
            request.request_id,
        )
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning(
            "[shutdown] EngineCore: rejecting utility call method=%s",
            method_name,
        )
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""
        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "irwkv shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def _make_ready_response(self) -> EngineCoreReadyResponse:
        parallel_config = self.irwkv_config.parallel_config
        scheduler_config = self.irwkv_config.scheduler_config
        return EngineCoreReadyResponse(
            max_model_len=self.irwkv_config.model_config.max_model_len,
            num_gpu_blocks=self.irwkv_config.cache_config.num_gpu_blocks or 0,
            block_size=self.irwkv_config.cache_config.block_size,
            mamba_block_size=self.irwkv_config.cache_config.mamba_block_size,
            effective_attention_block_size=(
                self.irwkv_config.cache_config.effective_attention_block_size
            ),
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.irwkv_config.model_config.dtype).removeprefix("torch."),
            irwkv_version=IRWKV_VERSION,
            world_size=self.irwkv_config.parallel_config.world_size,
            data_parallel_size=parallel_config.data_parallel_size,
            kv_cache_size_tokens=self.irwkv_config.cache_config.kv_cache_size_tokens,
            kv_cache_max_concurrency=(
                self.irwkv_config.cache_config.kv_cache_max_concurrency
            ),
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            data_parallel_rank=self.engine_index,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            instance_id=self.irwkv_config.instance_id,
            kv_events_config=self.scheduler.get_kv_event_publisher_config(),
            weight_transfer_backend=(
                self.irwkv_config.weight_transfer_config.backend
                if self.irwkv_config.weight_transfer_config is not None
                else None
            ),
            enable_sleep_mode=self.irwkv_config.model_config.enable_sleep_mode,
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""
        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = self._make_ready_response()
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except MultiModalCacheMissError as e:
                            # P0/P1 shadow drift -- return a retryable signal (P0
                            # drops the stale entry, client resends with data).
                            self._handle_mm_cache_miss(req, e)
                            continue
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""
        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Payload buffers that can't be reused yet because zmq may still be
        # sending them.
        # Buffers of the zero-copy tensor/ndarray frames don't need tracking
        # here: zmq itself holds a reference to each until it's done with it.
        pending = deque[tuple[zmq.MessageTracker, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reclaimed = pending.pop()[1]
                    if len(reuse_buffers) < max_reuse_bufs:
                        reuse_buffers.append(reclaimed)

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = self._send_msg_tracking_payload(
                    sockets[client_index], buffers
                )
                if not tracker.done:
                    pending.appendleft((tracker, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_mm_cache_miss(
        self, request: EngineCoreRequest, err: MultiModalCacheMissError
    ) -> None:
        """Return a retryable response for a P0/P1 cache-drift miss.

        Surfaces every drifted hash via ``EngineCoreOutput.mm_cache_miss_hashes`` so
        the frontend drops them from its sender cache and the client resends with
        data (see ``MultiModalCacheMissError``). Logged at warning, not exception,
        because it is expected and self-healing.
        """
        logger.warning(
            "Multi-modal cache miss for request %s (mm_hashes=%s): P0/P1 cache "
            "drift; returning a retryable response so the items are resent with data.",
            request.request_id,
            err.mm_hashes,
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                            mm_cache_miss_hashes=err.mm_hashes,
                        )
                    ],
                ),
            )
        )

    @staticmethod
    def _send_msg_tracking_payload(
        socket: zmq.Socket, buffers: Sequence[bytestr]
    ) -> zmq.MessageTracker:
        """Send `buffers` as a zero-copy multipart message, returning a tracker
        for the *first* frame.

        Used instead of `Socket.send_multipart()` because we reuse the buffer
        passed to `MsgpackEncoder.encode_into()`: `send_multipart()` returns a
        tracker for the last frame only.
        """
        more_flag = zmq.SNDMORE if len(buffers) > 1 else 0
        tracker = socket.send(buffers[0], more_flag, copy=False, track=True)
        if more_flag:
            socket.send_multipart(buffers[1:], copy=False)
        return tracker

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in get_args(PauseMode):
            raise ValueError(f"Invalid pause mode: {mode}")

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            try:
                engine._finish_pause(clear_cache)
            except Exception as e:
                future.set_exception(e)
            else:
                future.set_result(None)

        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)

        if self._pause_complete():
            self._finish_pause(clear_cache)
            return None

        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        return not self.has_work()

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[Request]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for request in aborted_reqs:
                by_client[request.client_index].add(request.request_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class EngineCoreActorMixin:
    """Ray actor for running EngineCore in a data parallel context."""

    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        if logging_config := getattr(irwkv_config, "logging_config", None):
            configure_logging(logging_config)

        # Initialize tracer for distributed tracing if configured.
        maybe_init_worker_tracer(
            instrumenting_module_name="irwkv.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        self.addresses = addresses
        irwkv_config.parallel_config.data_parallel_index = dp_rank
        irwkv_config.parallel_config.data_parallel_rank_local = local_dp_rank

        self._set_nixl_side_channel_host()

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but irwkv workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the irwkv worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(irwkv_config, local_dp_rank)

    @staticmethod
    def _set_nixl_side_channel_host():
        import ray

        # The driver-side value is excluded from Ray actor env propagation.
        # Fill in an actor-local default while preserving explicit overrides.
        os.environ.setdefault(
            "IRWKV_NIXL_SIDE_CHANNEL_HOST", ray.util.get_node_ip_address()
        )

    def _set_visible_devices(self, irwkv_config: IrwkvConfig, local_dp_rank: int):
        from irwkv.platforms import current_platform

        device_control_env_var = current_platform.device_control_env_var
        self._set_assigned_physical_gpu_ids(
            irwkv_config, local_dp_rank, device_control_env_var
        )

    def _set_assigned_physical_gpu_ids(
        self,
        irwkv_config: IrwkvConfig,
        local_dp_rank: int,
        device_control_env_var: str,
    ):
        world_size = irwkv_config.parallel_config.world_size
        try:
            physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                device_control_env_var,
                local_dp_rank,
                world_size,
                user_assigned_gpu_ids=(
                    irwkv_config.parallel_config.assigned_physical_gpu_ids
                ),
            )
            irwkv_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
        except IndexError as e:
            raise Exception(
                f"Error computing assigned_physical_gpu_ids: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        irwkv_config: IrwkvConfig,
        client_handshake_address: str | None,
    ):
        """For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """

    def run(self):
        """Run the engine core busy loop."""
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        irwkv_config.parallel_config.reconfigure_for_independent_dp_rank()
        EngineCoreActorMixin.__init__(
            self, irwkv_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            irwkv_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
