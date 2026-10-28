# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy wrappers for FlashInfer linear algebra, sampling, and communication."""

import contextlib
import functools
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
from collections.abc import Callable
from typing import Any, NoReturn

import torch
from packaging.version import Version

import irwkv.envs as envs
from irwkv.logger import init_logger
from irwkv.platforms import current_platform

logger = init_logger(__name__)


def flashinfer_bf16_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    backend: str,
) -> torch.Tensor:
    from flashinfer import mm_bf16

    return mm_bf16(
        a,
        b,
        bias=bias,
        pdl=pdl,
        out_dtype=torch.bfloat16,
        backend=backend,
    )


_DEFAULT_CUDA_HOME = "/usr/local/cuda"


def _flashinfer_nvcc_path() -> str | None:
    """Return the nvcc FlashInfer's JIT would run, or None if it is missing.

    Mirrors ``flashinfer.jit.cpp_ext.get_cuda_path()`` without importing
    FlashInfer, whose import initializes CUDA.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = (
            os.path.dirname(os.path.dirname(nvcc)) if nvcc else _DEFAULT_CUDA_HOME
        )
    return shutil.which(os.path.join(cuda_home, "bin", "nvcc"))


@functools.cache
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.IRWKV_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # FlashInfer's JIT runs nvcc and `ninja` (from PATH).
    if not has_flashinfer_cubin() and (
        _flashinfer_nvcc_path() is None or shutil.which("ninja") is None
    ):
        logger.warning_once(
            "FlashInfer kernels are disabled: flashinfer-cubin is not installed "
            "and nvcc (CUDA_HOME, CUDA_PATH, PATH or /usr/local/cuda) or ninja "
            "(PATH) is missing. Set CUDA_HOME to a CUDA toolkit and put ninja on "
            "PATH, or run `irwkv download-kernels`."
        )
        return False
    return True


def _installed_version(distribution: str) -> str | None:
    try:
        return Version(importlib.metadata.version(distribution)).public
    except importlib.metadata.PackageNotFoundError:
        return None


def has_flashinfer_jit_cache_wheels() -> bool:
    """Return whether FlashInfer publishes flashinfer-jit-cache for the installed
    PyTorch's CUDA version, which it does from CUDA 12.9."""
    cuda_version = torch.version.cuda
    return cuda_version is not None and Version(cuda_version) >= Version("12.9")


def warn_if_flashinfer_kernels_missing() -> None:
    """Warn on Hopper and newer GPUs when FlashInfer's precompiled kernels are
    missing or were installed for another FlashInfer version."""
    if not (
        current_platform.is_cuda()
        and current_platform.has_device_capability(90)
        and has_flashinfer()
    ):
        return
    flashinfer_version = _installed_version("flashinfer-python")
    packages = []
    if not envs.IRWKV_HAS_FLASHINFER_CUBIN:
        packages.append("flashinfer-cubin")
    if has_flashinfer_jit_cache_wheels():
        packages.append("flashinfer-jit-cache")
    installed = {name: _installed_version(name) for name in packages}
    if flashinfer_version is None or all(
        version == flashinfer_version for version in installed.values()
    ):
        return
    logger.warning_once(
        "FlashInfer's precompiled kernels are missing or do not match "
        "flashinfer-python %s (%s). FlashInfer then downloads and compiles "
        "kernels at startup, which can take several minutes, or fails to import "
        "mismatched ones. Run `irwkv download-kernels` in this Python environment "
        "to install them.",
        flashinfer_version,
        ", ".join(
            f"{name} {version or 'not installed'}"
            for name, version in installed.items()
        ),
    )


@functools.cache
def has_flashinfer_bf16_gemm() -> bool:
    """Return whether FlashInfer exposes the BF16 dense GEMM API."""
    if not has_flashinfer():
        return False
    mod = _get_submodule("flashinfer")
    return mod is not None and callable(getattr(mod, "mm_bf16", None))


@functools.cache
def is_flashinfer_bf16_gemm_supported(
    backend: str,
    compute_capability: int | None = None,
) -> bool:
    """Return whether an exact FlashInfer BF16 backend is available."""
    if not current_platform.is_cuda() or not has_flashinfer_bf16_gemm():
        return False

    mod = _get_submodule("flashinfer")
    mm_bf16 = getattr(mod, "mm_bf16", None) if mod is not None else None
    backend_supported = getattr(mm_bf16, "is_backend_supported", None)
    if not callable(backend_supported):
        return False

    if compute_capability is None:
        device_capability = current_platform.get_device_capability()
        if device_capability is None:
            return False
        compute_capability = device_capability.to_int()

    try:
        return bool(backend_supported(backend, compute_capability))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


@functools.cache
def is_flashinfer_cutedsl_bf16_gemm_supported() -> bool:
    """Return whether the CuTeDSL BF16 dense GEMM backend is available."""
    if not is_flashinfer_bf16_gemm_supported("cute-dsl"):
        return False
    try:
        from flashinfer.cute_dsl.utils import is_cute_dsl_available
        from flashinfer.utils import is_sm100a_supported
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        return is_cute_dsl_available() and is_sm100a_supported(torch.device("cuda"))
    except (RuntimeError, TypeError, ValueError):
        return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


flashinfer_get_hybrid_num_tokens_buckets = _lazy_import_wrapper(
    "flashinfer.fused_moe.utils", "get_hybrid_num_tokens_buckets"
)


autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_nvlink_two_sided() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True
