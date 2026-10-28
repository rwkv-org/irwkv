# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    overload,
    runtime_checkable,
)

import torch
import torch.nn as nn
from typing_extensions import TypeIs, TypeVar

from irwkv.logger import init_logger
from irwkv.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from irwkv.config import IrwkvConfig
    from irwkv.config.model import AttnTypeStr
else:
    IrwkvConfig = Any
    AttnTypeStr = Any

logger = init_logger(__name__)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
# which has T = list[torch.Tensor]
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class IrwkvModel(Protocol[T_co]):
    """The interface required for all models in irwkv."""

    def __init__(self, irwkv_config: IrwkvConfig, prefix: str = "") -> None: ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to `input_ids`."""
        ...

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...


def _check_irwkv_model_init(model: type[object] | object) -> bool:
    model_init = getattr(model, "__init__")  # noqa: B009
    return supports_kw(model_init, "irwkv_config")


def _check_irwkv_model_embed_input_ids(model: type[object] | object) -> bool:
    model_embed_input_ids = getattr(model, "embed_input_ids", None)
    if not callable(model_embed_input_ids):
        logger.warning(
            "The model (%s) is missing the `embed_input_ids` method.",
            model,
        )
        return False

    return True


def _check_irwkv_model_forward(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    irwkv_kws = ("input_ids", "positions")
    missing_kws = tuple(kw for kw in irwkv_kws if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type) and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "irwkv-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
def is_irwkv_model(model: type[object]) -> TypeIs[type[IrwkvModel]]: ...


@overload
def is_irwkv_model(model: object) -> TypeIs[IrwkvModel]: ...


def is_irwkv_model(
    model: type[object] | object,
) -> TypeIs[type[IrwkvModel]] | TypeIs[IrwkvModel]:
    return (
        _check_irwkv_model_init(model)
        and _check_irwkv_model_embed_input_ids(model)
        and _check_irwkv_model_forward(model)
    )


@runtime_checkable
class IrwkvModelForTextGeneration(IrwkvModel[T], Protocol[T]):
    """The interface required for all generative models in irwkv."""

    def compute_logits(
        self,
        hidden_states: T,
    ) -> T | None:
        """Return `None` if TP rank > 0."""
        ...


@overload
def is_text_generation_model(
    model: type[object],
) -> TypeIs[type[IrwkvModelForTextGeneration]]: ...


@overload
def is_text_generation_model(model: object) -> TypeIs[IrwkvModelForTextGeneration]: ...


def is_text_generation_model(
    model: type[object] | object,
) -> TypeIs[type[IrwkvModelForTextGeneration]] | TypeIs[IrwkvModelForTextGeneration]:
    if not is_irwkv_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, IrwkvModelForTextGeneration)

    return isinstance(model, IrwkvModelForTextGeneration)


_T = TypeVar("_T", bound=type[nn.Module])


def attn_type(attn_type: AttnTypeStr):
    """Set the default attention type."""

    def func(model: _T) -> _T:
        model.attn_type = attn_type  # type: ignore
        return model

    return func


def get_attn_type(model: type[object] | object) -> AttnTypeStr:
    return getattr(model, "attn_type", "decoder")
