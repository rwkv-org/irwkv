# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import irwkv.kernels  # noqa: F401
from irwkv import envs, ir
from irwkv.logger import init_logger
from irwkv.model_executor.custom_op import CustomOp
from irwkv.model_executor.determinism.batch_invariant import rms_norm_batch_invariant

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from irwkv import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # When has_weight=False, pass weight=None so implementations that
        # support a weightless path can skip the per-channel multiply.
        # Implementations that require weight (e.g. oink) fall back via IR
        # op priority when weight=None is unsupported.
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.IRWKV_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            pass_weight = (
                self.pass_weight_add if residual is not None else self.pass_weight
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data if pass_weight else None,
                self.variance_epsilon,
                residual=residual,
            )

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.IRWKV_BATCH_INVARIANT and residual is not None:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            weight = self.weight.data if self.pass_weight_add else None
            return ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
                x, residual, weight, self.variance_epsilon
            )
        return self.forward_cuda(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


@CustomOp.register("layer_norm")
class LayerNorm(CustomOp):
    """Standard (mean-centered) LayerNorm.

    Drop-in for a bare `nn.LayerNorm` that dispatches to a fused XPU kernel
    when one is available, and to the native implementation otherwise.

    Normalization runs in the wider of the input and parameter dtypes and the
    result is cast back to the input dtype, so `dtype=torch.float32` gives an
    fp32 reduction for a lower-precision input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.weight: nn.Parameter | None = None
        self.bias: nn.Parameter | None = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=weight_dtype))

    def _compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.weight is None:
            return x.dtype
        return torch.promote_types(x.dtype, self.weight.dtype)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._compute_dtype(x)
        if compute_dtype != x.dtype:
            return F.layer_norm(
                x.to(compute_dtype),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).type_as(x)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def extra_repr(self) -> str:
        return f"hidden_size={self.normalized_shape[0]}, eps={self.eps}"
