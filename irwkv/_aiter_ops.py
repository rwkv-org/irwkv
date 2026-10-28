# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import functools
import os
from collections.abc import Callable

import torch

import irwkv.envs as envs
from irwkv.logger import init_logger
from irwkv.platforms import current_platform

logger = init_logger(__name__)


def is_aiter_found() -> bool:
    from importlib.util import find_spec

    return find_spec("aiter") is not None


# `find_spec` is not torch.compile compatible.
# In cases where aiter availability might have
# been checked in forward passes that are torch compiled.
# we keep this global outside to not cause torch compile breaks.
IS_AITER_FOUND = is_aiter_found()


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_HIP_RUNTIME_HANDLE: ctypes.CDLL | None = None


def _find_torch_hip_runtime_path() -> str | None:
    """Find the HIP runtime already selected by PyTorch's extension graph."""
    try:
        torch_extension = ctypes.CDLL(torch._C.__file__)
        hip_symbol = torch_extension.hipGetDeviceCount
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        dladdr.restype = ctypes.c_int
        info = _DlInfo()
        if not dladdr(ctypes.cast(hip_symbol, ctypes.c_void_p), ctypes.byref(info)):
            return None
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
        return None
    if info.dli_fname is None:
        return None

    runtime_path = os.fsdecode(info.dli_fname)
    if "libamdhip64.so" not in os.path.basename(runtime_path):
        return None
    return runtime_path


def _promote_torch_hip_runtime() -> None:
    """Make PyTorch's HIP runtime symbols visible to AITER JIT extensions.

    PyTorch loads its native extension with ``RTLD_LOCAL``. Some AITER JIT
    extensions rely on HIP registration and launch symbols being available in
    the process-global lookup scope. Reopening the exact runtime that provides
    PyTorch's ``hipGetDeviceCount`` symbol promotes the existing DSO instead of
    selecting or initializing a second HIP runtime.
    """
    global _HIP_RUNTIME_HANDLE
    if _HIP_RUNTIME_HANDLE is not None:
        return

    runtime_path = _find_torch_hip_runtime_path()
    if runtime_path is None:
        logger.warning_once(
            "Could not locate PyTorch's HIP runtime; AITER JIT extensions may "
            "fail to resolve HIP symbols."
        )
        return

    try:
        # Retain the handle for the process lifetime. Besides preventing a
        # future dlclose, this documents that the RTLD_GLOBAL promotion is
        # intentional and must outlive every lazily imported AITER extension.
        _HIP_RUNTIME_HANDLE = ctypes.CDLL(runtime_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        logger.warning_once(
            "Could not promote PyTorch's HIP runtime at %s; AITER JIT "
            "extensions may fail to resolve HIP symbols: %s",
            runtime_path,
            err,
        )


def _maybe_promote_torch_hip_runtime() -> None:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        _promote_torch_hip_runtime()


_maybe_promote_torch_hip_runtime()


def is_aiter_found_and_supported() -> bool:
    """Check that AITER is installed on a supported ROCm architecture."""
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from irwkv.platforms.rocm import get_cdna_version

        return get_cdna_version() > 2
    return False


def if_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if
    ROCm AITER package is supported and enabled on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_aiter_found_and_supported():
            return func(*args, **kwargs)

        return None

    return wrapper


class rocm_aiter_ops:
    """Availability checks for ROCm AITER sampling and communication."""

    _AITER_ENABLED = envs.IRWKV_ROCM_USE_AITER
    _CUSTOM_ALL_REDUCE_ENABLED = envs.IRWKV_ROCM_USE_AITER_CUSTOM_AR

    @classmethod
    def refresh_env_variables(cls):
        cls._AITER_ENABLED = envs.IRWKV_ROCM_USE_AITER
        cls._CUSTOM_ALL_REDUCE_ENABLED = envs.IRWKV_ROCM_USE_AITER_CUSTOM_AR

    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED

    @classmethod
    @if_aiter_supported
    def is_custom_all_reduce_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._CUSTOM_ALL_REDUCE_ENABLED
