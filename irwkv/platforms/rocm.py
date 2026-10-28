# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import os
import platform
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING

import regex as re
import torch

import irwkv.envs as envs
from irwkv.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

_KV_CACHE_DTYPE_REASON = "kv_cache_dtype not supported"

try:
    from amdsmi import (
        AmdSmiMemoryType,
        amdsmi_get_gpu_asic_info,
        amdsmi_get_gpu_memory_total,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
    )
except ImportError as e:
    logger.warning("Failed to import from amdsmi with %r", e)

_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",  # MI300X VF
    "0x74a2": "AMD_Instinct_MI308X",
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",  # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",
    "0x74bd": "AMD_Instinct_MI300X_HF",
    "0x744c": "AMD_Radeon_RX7900XTX",
    # RDNA 3.5 APUs (Strix Point / Strix Halo)
    "0x150e": "AMD_Radeon_890M",  # gfx1150, Strix Point
    "0x1586": "AMD_Radeon_8060S",  # gfx1151, Strix Halo
    # RDNA 4 discrete (Navi 48)
    "0x7550": "AMD_Radeon_RX9070XT",  # gfx1201
    "0x7551": "AMD_Radeon_R9700",  # gfx1201
}


@lru_cache(maxsize=8)
def _rocm_device_count_stateless(cuda_visible_devices: str | None = None) -> int:
    """Get number of ROCm devices, caching based on the value of CUDA_VISIBLE_DEVICES
    at the time of call.

    This should be used instead of torch.accelerator.device_count() unless
    CUDA_VISIBLE_DEVICES has already been set to the desired value.

    # This can be removed and simply replaced with torch.cuda.get_device_count
    # after https://github.com/pytorch/pytorch/pull/122815 is released."""
    # Note: cuda_visible_devices is not used, but we keep it as an argument for
    # LRU Cache purposes.

    # Code below is based on
    # https://github.com/pytorch/pytorch/blob/
    # c1cd946818442aca8c7f812b16d187ce1586c3bc/
    # torch/cuda/__init__.py#L831C1-L831C17
    import torch.cuda

    if not torch.cuda._is_compiled():
        return 0
    # ROCm uses amdsmi instead of nvml for stateless device count
    # This requires a sufficiently modern version of Torch 2.4.0
    raw_count = (
        torch.cuda._device_count_amdsmi()
        if (hasattr(torch.cuda, "_device_count_amdsmi"))
        else -1
    )
    r = torch._C._cuda_getDeviceCount() if raw_count < 0 else raw_count
    return r


@cache
def _get_wsl_kernel_version() -> tuple[int, ...] | None:
    try:
        release = platform.uname().release
        parts = release.split("-")[0].split(".")
        return tuple(int(part) for part in parts[:3])
    except (TypeError, ValueError):
        return None


def _sync_hip_cuda_env_vars():
    """Ensure HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are consistent.
    Treats empty string as unset. Raises on genuine conflicts."""
    hip_val = os.environ.get("HIP_VISIBLE_DEVICES") or None
    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES") or None

    if hip_val is not None and cuda_val is not None:
        if hip_val != cuda_val:
            raise ValueError(
                f"Inconsistent GPU visibility env vars: "
                f"HIP_VISIBLE_DEVICES='{hip_val}' vs "
                f"CUDA_VISIBLE_DEVICES='{cuda_val}'. "
                f"Please set only one, or ensure they match."
            )
    elif hip_val is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_val


# Sync at import time - catches misconfigurations from process start.
_sync_hip_cuda_env_vars()


# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


@with_amdsmi_context
def _query_gcn_arch_from_amdsmi() -> str:
    """Query GCN arch from amdsmi. Raises if not available."""
    handles = amdsmi_get_processor_handles()
    if handles:
        asic_info = amdsmi_get_gpu_asic_info(handles[0])
        # Use target_graphics_version which contains the gfx name
        # e.g., 'gfx942' for MI300X/MI325X
        target_gfx = asic_info.get("target_graphics_version", "")
        if target_gfx:
            return target_gfx
    raise RuntimeError("amdsmi did not return valid GCN arch")


@with_amdsmi_context
def _query_total_memory_from_amdsmi(physical_device_id: int) -> int:
    """Query total VRAM (bytes) from amdsmi. Raises if not available."""
    handles = amdsmi_get_processor_handles()
    handle = handles[physical_device_id]
    return amdsmi_get_gpu_memory_total(handle, AmdSmiMemoryType.VRAM)


def _get_gcn_arch() -> str:
    """Get GCN arch via amdsmi (no CUDA init), fallback to torch.cuda.
    Called once at module level; result stored in _GCN_ARCH.
    """
    try:
        return _query_gcn_arch_from_amdsmi()
    except Exception as e:
        logger.debug("Failed to get GCN arch via amdsmi: %s", e)
    # Ultimate fallback: use torch.cuda (will initialize CUDA)
    return torch.cuda.get_device_properties("cuda").gcnArchName


# Resolve once at module load. Uses amdsmi (no CUDA init) so Ray workers
# can still set CUDA_VISIBLE_DEVICES after import.
# These are plain Python bools — fully torch.compile/Dynamo safe.
_GCN_ARCH = _get_gcn_arch()

_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
_ON_GFX11 = "gfx11" in _GCN_ARCH
_ON_GFX1100 = "gfx1100" in _GCN_ARCH
_ON_GFX1151 = "gfx1151" in _GCN_ARCH
_ON_GFX12X = any(arch in _GCN_ARCH for arch in ["gfx12"])
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_GFX90A = "gfx90a" in _GCN_ARCH
_ON_GFX942 = "gfx942" in _GCN_ARCH
_ON_GFX950 = "gfx950" in _GCN_ARCH
_ON_GFX1250 = "gfx1250" in _GCN_ARCH

_ON_CDNA = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"])
# RDNA = gfx11/gfx12 minus the CDNA-classified gfx1250.
_ON_RDNA = _ON_GFX1X and not _ON_CDNA
_ON_RDNA4 = any(arch in _GCN_ARCH for arch in ["gfx1200", "gfx1201"])


def _capability_from_gcn_arch(gcn_arch: str) -> tuple[int, int] | None:
    """Parse (major, minor) from a GCN arch string, mirroring how
    HIP derives hipDeviceProp_t.major / .minor.

    Format: gfx<MAJOR><MINOR><STEPPING>
      - 1-digit major  (gfx9xx):  "gfx" + M + m + stepping
      - 2-digit major  (gfx1xxx): "gfx" + MM + m + stepping

    Examples:
      gfx90a  -> (9, 0)    gfx942  -> (9, 4)    gfx950 -> (9, 5)
      gfx1100 -> (11, 0)   gfx1101 -> (11, 0)   gfx1200 -> (12, 0)

    Returns None only when the string is not gfx-prefixed at all
    (i.e. not a ROCm arch string). Raises on any string that looks
    like a GCN arch but does not match a known layout.

    """
    m = re.match(r"gfx(\d+)", gcn_arch)
    if not m:
        # Not a gfx string at all — caller should fall back to torch.cuda
        return None

    digits = m.group(1)
    n = len(digits)

    if n < 2:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has too few digits ({n}) after 'gfx' "
            f"to derive a (major, minor) capability. "
            f"Please file a irwkv issue with your GPU model."
        )

    if n in (2, 3):
        # 1-digit major: gfx9 family
        # len 2: major + minor          (e.g. gfx90 from gfx90a)
        # len 3: major + minor + step   (e.g. gfx942)
        major = int(digits[0])
        minor = int(digits[1])
    elif n == 4:
        # 2-digit major: gfx10xx, gfx11xx, gfx12xx
        # major(2) + minor(1) + stepping(1)
        major = int(digits[:2])
        minor = int(digits[2])
    elif n >= 5:
        raise ValueError(
            f"GCN arch '{gcn_arch}' has {n} digits after 'gfx', which "
            f"exceeds the known 4-digit layout (MMms). Cannot determine "
            f"major/minor split unambiguously. "
            f"Please file a irwkv issue with your GPU model."
        )

    if major < 9:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version < 9 is not expected for any supported AMD GPU. "
            f"Please file a irwkv issue with your GPU model."
        )

    if major > 12:
        raise ValueError(
            f"Parsed unknown ROCm architecture from GCN arch '{gcn_arch}': "
            f"major={major}, minor={minor}. "
            f"Major version > 12 is beyond currently known AMD generations. "
            f"Please file a irwkv issue with your GPU model so support "
            f"can be added."
        )

    return (major, minor)


def on_gfx1x() -> bool:
    return _ON_GFX1X and not _ON_CDNA


def on_gfx11() -> bool:
    return _ON_GFX11


def on_gfx1100() -> bool:
    return _ON_GFX1100


def on_gfx1151() -> bool:
    return _ON_GFX1151


def on_gfx12x() -> bool:
    return _ON_GFX12X and not _ON_CDNA


def on_gfx1250() -> bool:
    return _ON_GFX1250


def on_rdna4() -> bool:
    return _ON_RDNA4


def on_mi3xx() -> bool:
    return _ON_MI3XX


def on_gfx9() -> bool:
    return _ON_GFX9


def on_gfx90a() -> bool:
    return _ON_GFX90A


def on_gfx942() -> bool:
    return _ON_GFX942


def on_gfx950() -> bool:
    return _ON_GFX950


def on_cdna() -> bool:
    return _ON_CDNA


def on_rdna() -> bool:
    return _ON_RDNA


def get_cdna_version() -> int:
    if on_gfx90a():
        return 2
    if on_gfx942():
        return 3
    if on_gfx950():
        return 4
    if on_gfx1250():
        return 5
    return 0


@cache
def get_rocm_version() -> tuple[int, ...] | None:
    """Return the installed ROCm release as (major, minor, patch), or None."""
    # ROCm 10+ ships as the `rocm` pip SDK; older releases install to /opt/rocm.
    try:
        version = importlib.metadata.version("rocm")
    except importlib.metadata.PackageNotFoundError:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        try:
            with open(os.path.join(rocm_path, ".info", "version")) as f:
                version = f.read()
        except OSError:
            return None
    match = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


# Enable HIP online tuning early, before hipBLASLt initializes.
# Turn on hipBLASLt online tuning if use AITER hipBLASLt GEMM.
if (
    envs.IRWKV_ROCM_USE_AITER
    and envs.IRWKV_ROCM_USE_AITER_LINEAR
    and envs.IRWKV_ROCM_USE_AITER_LINEAR_HIPBMM
    and get_cdna_version() > 2
):
    os.environ["HIP_ONLINE_TUNING"] = "1"


@cache
def use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
    sliding_window: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> bool:
    # custom paged attn always supported on V0. On V1, requires sliding window
    # disabled due to observed numerical discrepancy.
    if on_cdna():
        return (
            (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and sinks is None
        )

    else:
        return (
            _ON_GFX1X
            and (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and head_size == 128
            and block_size == 16
            and (gqa_ratio >= 3 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and alibi_slopes is None
            and kv_cache_dtype == "auto"
            and sinks is None
        )
