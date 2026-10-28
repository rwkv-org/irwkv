# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import irwkv.envs as envs
from irwkv.logger import init_logger
from irwkv.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from irwkv.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
)
from irwkv.model_executor.warmup.watermark_sample_warmup import (
    watermark_sample_warmup,
)
from irwkv.platforms import current_platform
from irwkv.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from irwkv.v1.worker.gpu.model_runner import GPUModelRunner
    from irwkv.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    pass

    enable_jit_warmup = worker.irwkv_config.kernel_config.enable_jit_warmup
    if enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            registry = (
                worker.model_runner.jit_warmup_registry  # type: ignore[attr-defined]
            )
            registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    # Run next so input-prep kernels JIT against pristine runner state.
    if enable_jit_warmup:
        watermark_sample_warmup(worker)

    if worker.irwkv_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    # Allocate the exact decode-sized workspace, autotune cache misses, and
    # resolve every CUDA Graph bucket before capture begins.
    # Lazy import: flashinfer_pcie_ipc_all_reduce imports flashinfer.comm,
    # which initializes CUDA at import time and must not run at module scope.
    if envs.IRWKV_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
        from irwkv.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce,
        )

        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)

    enable_flashinfer_autotune = (
        worker.irwkv_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info_once("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _run_flashinfer_autotune_dummy_runs(
    runner: "GPUModelRunner", *, skip_attn: bool = False
) -> None:
    import irwkv.utils.flashinfer as fi_utils

    dummy_run_kwargs = {"skip_attn": True} if skip_attn else {}
    num_tokens = runner.scheduler_config.max_num_batched_tokens
    max_tuning_tokens = num_tokens
    tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
        max_tuning_tokens
    )
    logger.info(
        "Running FlashInfer autotune with %d tokens and token buckets %s.",
        num_tokens,
        tuning_buckets,
    )
    # Round M up to match serving-time bucket selection for non-bucket M.
    # See https://github.com/flashinfer-ai/flashinfer/issues/5450
    with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):
        runner._dummy_run(
            num_tokens=num_tokens,
            is_profile=True,
            randomize_inputs=True,
            **dummy_run_kwargs,
        )


def _autotune_cache_fingerprint(path: Path) -> tuple[str, int] | None:
    """Identify a saved autotune file by its FlashInfer metadata and size."""
    try:
        configs = json.loads(path.read_text())
        metadata = configs.pop("_metadata", None)
    except (OSError, ValueError, AttributeError):
        return None
    return json.dumps(metadata, sort_keys=True), len(configs)


def _all_ranks_have_matching_cache(path: Path, group) -> bool:
    """True iff every rank in ``group`` has its own, mutually consistent file."""
    fingerprint = _autotune_cache_fingerprint(path)
    if group.world_size == 1:
        return fingerprint is not None
    gathered: list[tuple[str, int] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, fingerprint, group=group.cpu_group)
    return fingerprint is not None and all(f == fingerprint for f in gathered)


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    With PP > 1, stages run different layers and may profile different ops,
    so each stage's TP group tunes separately with its own cache file;
    otherwise the world group tunes together. Per-tactic timings are
    averaged over the tuning group so all its ranks select the same tactic.

    Results are persisted per rank: FlashInfer keys MoE entries by tp/ep rank
    (``MoERunner.get_cache_key_extras``), so one rank's file only hits on that
    rank. A rank with a cache hit skips the per-tactic reduce the others block
    in, so ranks keep loaded configs only if every rank in the tuning group
    has a matching file and successfully loads it.
    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import irwkv.utils.flashinfer as fi_utils
    from irwkv.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
        get_world_group,
    )

    world = get_world_group()
    pp_size = get_pp_group().world_size
    tune_group = get_tp_group() if pp_size > 1 else world
    is_leader = tune_group.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = set(envs.IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS or ())
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    cache_path = resolve_flashinfer_autotune_file(runner)
    # The world group only spans DP ranks when irwkv folds DP into it.
    dp_rank = runner.irwkv_config.parallel_config.data_parallel_rank
    cache_path = cache_path.with_name(
        f"{cache_path.stem}_dp{dp_rank}_rank{world.rank_in_group}{cache_path.suffix}"
    )
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)

    if _all_ranks_have_matching_cache(cache_path, tune_group):
        loaded = tuner.load_configs(str(cache_path))
        if tune_group.world_size > 1:
            loaded_by_rank: list[bool | None] = [None] * tune_group.world_size
            torch.distributed.all_gather_object(
                loaded_by_rank, loaded, group=tune_group.cpu_group
            )
            loaded = all(loaded_by_rank)
        if not loaded:
            tuner.clear_cache()

    group = tune_group.cpu_group if tune_group.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with (
            torch.inference_mode(),
            fi_utils.autotune(tune_mode=True, **autotune_kwargs),
        ):
            _run_flashinfer_autotune_dummy_runs(runner)
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    # Skip the rewrite when nothing was tuned this start (every entry came from
    # the file). FlashInfer gates its own autotune(cache=...) save the same way.
    if tuner._dirty:
        tuner.save_configs(str(cache_path))
