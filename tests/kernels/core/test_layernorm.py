# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from irwkv import ir
from irwkv.model_executor.layers.layernorm import RMSNorm
from irwkv.platforms import current_platform
from irwkv.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from irwkv.platforms.rocm import on_gfx90a

    on_mi250 = on_gfx90a()
else:
    on_mi250 = False

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True] if not on_mi250 else [True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def _rms_norm_tolerance(dtype: torch.dtype) -> dict[str, float]:
    return ir.ops.rms_norm.get_tolerance(dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_irwkv_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = x.new_empty_strided(x.size(), x.stride()) if add_residual else None
    if residual is not None:
        residual.normal_(std=scale)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm_weightless(
    default_irwkv_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size, has_weight=False).to(dtype=dtype)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None

    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    tol = _rms_norm_tolerance(dtype)
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], **tol)
        torch.testing.assert_close(out[1], ref_out[1], **tol)
    else:
        torch.testing.assert_close(out, ref_out, **tol)
