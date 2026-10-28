# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import multiprocessing as mp
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import irwkv.envs as envs
from irwkv.platforms import current_platform
from irwkv.utils.network_utils import get_open_port

from ..utils import multi_gpu_test

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-only quick-reduce tests",
)

MB = 1024 * 1024
WORLD_SIZE = 2
QUANT_LEVELS = ["FP", "INT8", "INT6", "INT4"]


def _log(message: str) -> None:
    print(f"[rocm_quick_reduce] {message}", flush=True)


def _reload_envs():
    return importlib.reload(envs)


def _make_quick_allreduce(
    *,
    disabled: bool = False,
    world_size: int = 2,
    quant_level: str = "FP",
    use_fp16_kernels: bool = False,
    qr_max_size: int = 64 * MB,
):
    from irwkv.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
        QuickReduceRegime,
    )

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = disabled
    qar.world_size = world_size
    qar.use_fp16_kernels = use_fp16_kernels
    qar.qr_quant_level = QuickReduceRegime[quant_level]
    qar.qr_max_size = qr_max_size
    qar.qr_min_size = None
    qar.qr_quantization_min_size = None
    return qar


def _quick_allreduce_worker(
    rank: int,
    port: int,
    quant_level: str,
    dtype_name: str,
    cast_bf16: bool,
):
    os.environ["IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_level
    os.environ["IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1" if cast_bf16 else "0"
    _log(
        f"worker start: rank={rank} quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )

    qar = None
    try:
        from irwkv.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qar = QuickAllReduce(group=dist.GroupMember.WORLD, device=rank)
        assert not qar.disabled

        num_elements = 8 * MB if dtype_name == "float16" else 4 * MB

        dtype = getattr(torch, dtype_name)
        inp = torch.ones(num_elements, dtype=dtype, device=device)

        assert qar.should_quick_allreduce(inp)
        if cast_bf16:
            assert qar.use_fp16_kernels

        out = qar.quick_all_reduce(inp)
        assert torch.allclose(out, inp * WORLD_SIZE, atol=2.5, rtol=0.1)
        _log(
            f"worker complete: rank={rank} quant={quant_level} "
            f"dtype={dtype_name} num_elements={num_elements} "
            f"use_fp16_kernels={qar.use_fp16_kernels}"
        )
    finally:
        if qar is not None:
            qar.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_gpu_quick_allreduce_test(
    *,
    quant_level: str,
    dtype_name: str,
    cast_bf16: bool,
):
    _log(
        f"launch 2-GPU case: quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )
    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = []

    for rank in range(WORLD_SIZE):
        proc = ctx.Process(
            target=_quick_allreduce_worker,
            args=(rank, port, quant_level, dtype_name, cast_bf16),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker exited with code {proc.exitcode}"
    _log(
        f"finished 2-GPU case: quant={quant_level} "
        f"dtype={dtype_name} cast_bf16={cast_bf16}"
    )


CUDAGRAPH_WORLD_SIZE = 2
CUDAGRAPH_ROUNDS = 10
CUDAGRAPH_NUM_ELEMENTS = 1 << 21  # 2M fp16 = 4 MB, above the quick-reduce thresholds


def _quick_allreduce_cudagraph_worker(
    rank: int,
    world_size: int,
    port: int,
    quant_level: str,
):
    # FP keeps the all-reduce bit-exact for small fp16 integers, so every
    # replayed round can be checked exactly.
    os.environ["IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION"] = quant_level
    os.environ["IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "0"
    _log(f"cudagraph worker start: rank={rank} quant={quant_level}")

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    qar = None
    try:
        from irwkv.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qar = QuickAllReduce(group=dist.GroupMember.WORLD, device=rank)
        assert not qar.disabled

        N = CUDAGRAPH_NUM_ELEMENTS
        inp = torch.empty(N, dtype=torch.float16, device=device)
        out = torch.empty(N, dtype=torch.float16, device=device)
        assert qar.should_quick_allreduce(inp)

        # Every rank contributes the same value v in a round, so the true
        # cross-rank all-reduce sum is simply world_size * v.
        def expected(v):
            return float(world_size * v)

        if rank == 0:
            print(
                f"[repro] world_size={world_size} elems={N} regime={quant_level} fp16",
                flush=True,
            )

        # Warmup, then capture a graph with EXACTLY ONE quick-reduce (isolated
        # qr, so it is the sole writer of its flag slot -- the condition that
        # triggers the stale-flag bug).
        inp.fill_(1.0)
        qar.quick_all_reduce(inp, out=out)
        torch.accelerator.synchronize()
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            qar.quick_all_reduce(inp, out=out)
        torch.accelerator.synchronize()
        dist.barrier()

        for v in range(CUDAGRAPH_ROUNDS):
            inp.fill_(float(v))  # in-place: same value on every rank
            dist.barrier()
            g.replay()
            torch.accelerator.synchronize()
            dist.barrier()
            got = out.float()
            expect = expected(v)
            if rank == 0:
                print(f"round {v}: got={got[:10]}, expected={expect}", flush=True)
            mismatch = ~torch.isclose(got, torch.full_like(got, expect))
            num_mismatch = int(mismatch.sum().item())
            assert num_mismatch == 0, (
                f"rank={rank} round={v} expected={expect} "
                f"mismatched {num_mismatch}/{got.numel()} elements; "
                f"unique wrong values={torch.unique(got[mismatch])[:8].tolist()}"
            )
        _log(f"cudagraph worker complete: rank={rank} rounds={CUDAGRAPH_ROUNDS}")
    finally:
        if qar is not None:
            qar.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_cudagraph_replay_test(*, world_size: int, quant_level: str):
    _log(f"launch {world_size}-GPU cudagraph replay case: quant={quant_level}")
    ctx = mp.get_context("spawn")
    port = get_open_port()
    procs = []

    for rank in range(world_size):
        proc = ctx.Process(
            target=_quick_allreduce_cudagraph_worker,
            args=(rank, world_size, port, quant_level),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0, f"worker exited with code {proc.exitcode}"
    _log(f"finished {world_size}-GPU cudagraph replay case: quant={quant_level}")


def test_quick_reduce_regime_values():
    from irwkv.distributed.device_communicators.quick_all_reduce import (
        QuickReduceRegime,
    )

    assert QuickReduceRegime.FP.value == 0
    assert QuickReduceRegime.INT8.value == 1
    assert QuickReduceRegime.INT6.value == 2
    assert QuickReduceRegime.INT4.value == 3
    assert QuickReduceRegime.INT3.value == 4
    assert QuickReduceRegime.NONE.value == 5


def test_quick_reduce_regime_names():
    from irwkv.distributed.device_communicators.quick_all_reduce import (
        QuickReduceRegime,
    )

    assert set(QuickReduceRegime.__members__) == {
        "FP",
        "INT8",
        "INT6",
        "INT4",
        "INT3",
        "NONE",
    }


@pytest.mark.parametrize("quant_level", QUANT_LEVELS + ["NONE"])
def test_quick_reduce_quantization_env_var(monkeypatch, quant_level):
    monkeypatch.setenv("IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION", quant_level)

    reloaded_envs = _reload_envs()
    assert quant_level == reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION


def test_quick_reduce_quantization_default(monkeypatch):
    monkeypatch.delenv("IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION == "NONE"


@pytest.mark.parametrize("cast_bf16", [True, False])
def test_quick_reduce_cast_bf16_to_fp16_env_var(monkeypatch, cast_bf16):
    monkeypatch.setenv(
        "IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1" if cast_bf16 else "0"
    )

    reloaded_envs = _reload_envs()
    assert reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 is cast_bf16


def test_quick_reduce_cast_bf16_to_fp16_default(monkeypatch):
    monkeypatch.delenv("IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16 is True


@pytest.mark.parametrize("max_mb", [128, 512, 2048, None])
def test_quick_reduce_max_size_env_var(monkeypatch, max_mb):
    if max_mb is None:
        monkeypatch.delenv("IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)
    else:
        monkeypatch.setenv("IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", str(max_mb))

    reloaded_envs = _reload_envs()
    assert max_mb == reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB


def test_quick_reduce_max_size_default(monkeypatch):
    monkeypatch.delenv("IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", raising=False)

    reloaded_envs = _reload_envs()
    assert reloaded_envs.IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is None


@pytest.mark.parametrize(
    ("gcn_arch_name", "expected"),
    [
        ("gfx942", True),
        ("gfx950", True),
        ("gfx90a", False),
        ("", False),
    ],
)
def test_quick_allreduce_rocm_arch_available(gcn_arch_name, expected):
    from irwkv.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = True

    with (
        patch(
            "irwkv.distributed.device_communicators.quick_all_reduce.current_platform."
            "is_rocm",
            return_value=True,
        ),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(gcnArchName=gcn_arch_name),
        ),
    ):
        assert qar._rocm_arch_available() is expected


def test_quick_allreduce_rocm_arch_available_handles_probe_failure():
    from irwkv.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    qar = QuickAllReduce.__new__(QuickAllReduce)
    qar.disabled = True

    with (
        patch(
            "irwkv.distributed.device_communicators.quick_all_reduce.current_platform."
            "is_rocm",
            return_value=True,
        ),
        patch("torch.cuda.get_device_properties", side_effect=RuntimeError),
    ):
        assert qar._rocm_arch_available() is False


def test_quick_allreduce_rejects_disabled():
    qar = _make_quick_allreduce(disabled=True)

    inp = torch.zeros(1024, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_unsupported_dtype():
    qar = _make_quick_allreduce()

    inp = torch.zeros(1024 * 1024, dtype=torch.float32)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_non_aligned_input():
    qar = _make_quick_allreduce()

    inp = torch.zeros(5, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_non_contiguous_input():
    qar = _make_quick_allreduce()

    inp = torch.zeros((1024, 1024), dtype=torch.float16)[:, ::2]
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_rejects_input_smaller_than_threshold():
    qar = _make_quick_allreduce()

    inp = torch.zeros((MB // 2) - 8, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_accepts_input_at_threshold():
    qar = _make_quick_allreduce()

    inp = torch.zeros(MB // 2, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is True


def test_quick_allreduce_rejects_input_larger_than_max_size():
    qar = _make_quick_allreduce(qr_max_size=1 * MB)

    inp = torch.zeros(MB, dtype=torch.float16)
    assert qar.should_quick_allreduce(inp) is False


def test_quick_allreduce_bf16_uses_fp16_threshold_when_cast_enabled():
    inp = torch.zeros(MB // 2, dtype=torch.bfloat16)

    without_cast = _make_quick_allreduce(use_fp16_kernels=False)
    with_cast = _make_quick_allreduce(use_fp16_kernels=True)

    assert without_cast.should_quick_allreduce(inp) is False
    assert with_cast.should_quick_allreduce(inp) is True


def test_quick_allreduce_supported_world_sizes():
    from irwkv.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert QuickAllReduce._SUPPORTED_WORLD_SIZES == [2, 4, 8]


def test_quick_allreduce_supported_dtypes():
    from irwkv.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    assert [torch.float16, torch.bfloat16] == QuickAllReduce._SUPPORTED_DTYPES


def test_quick_allreduce_min_size_table():
    from irwkv.distributed.device_communicators.quick_all_reduce import QuickAllReduce

    for dtype in [torch.float16, torch.bfloat16]:
        for world_size in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            min_sizes = QuickAllReduce._QR_MIN_SIZE[(dtype, world_size)]
            assert len(min_sizes) == 5
            assert all(size > 0 for size in min_sizes)


def test_qr_max_size():
    from irwkv import _custom_ops as ops

    max_size = ops.qr_max_size()
    assert isinstance(max_size, int)
    assert max_size > 0


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
@pytest.mark.parametrize("quant_level", QUANT_LEVELS)
def test_quick_allreduce_two_gpu_correctness(quant_level):
    _log(f"two-GPU correctness case: quant={quant_level}")
    _run_two_gpu_quick_allreduce_test(
        quant_level=quant_level,
        dtype_name="float16",
        cast_bf16=False,
    )


@multi_gpu_test(num_gpus=CUDAGRAPH_WORLD_SIZE)
def test_quick_allreduce_cudagraph_replay():
    # Regression test for the stale flag_color bug: a quick-reduce captured in a
    # CUDA graph must return correct results on every replay, not the data from
    # a previous round.
    _log("cudagraph replay case")
    _run_cudagraph_replay_test(
        world_size=CUDAGRAPH_WORLD_SIZE,
        quant_level="FP",
    )


@pytest.mark.skipif(
    current_platform.device_count() < WORLD_SIZE,
    reason="requires 2 ROCm GPUs",
)
def test_quick_allreduce_bf16_cast_mode():
    _log("BF16 cast case")
    _run_two_gpu_quick_allreduce_test(
        quant_level="FP",
        dtype_name="bfloat16",
        cast_bf16=True,
    )
