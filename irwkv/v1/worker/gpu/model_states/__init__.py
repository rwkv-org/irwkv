# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from irwkv.config import IrwkvConfig
from irwkv.v1.worker.gpu.mm.encoder_cache import EncoderCache
from irwkv.v1.worker.gpu.model_states.interface import ModelState


def init_model_state(
    irwkv_config: IrwkvConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    cls = resolve_model_state_cls(irwkv_config, model)

    # Reject enable_prompt_embeds for states that would silently ignore it.
    if (
        irwkv_config.model_config.enable_prompt_embeds
        and not cls.supports_prompt_embeds
    ):
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")

    return cls(irwkv_config, model, encoder_cache, device)


def resolve_model_state_cls(
    irwkv_config: IrwkvConfig, model: nn.Module
) -> type[ModelState]:
    # Let the model provide its own ModelState if it defines one.
    if hasattr(model, "get_model_state_cls"):
        return model.get_model_state_cls()

    from irwkv.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState
