# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from irwkv.model_executor.layers.attention_layer_base import AttentionLayerBase


def is_deferred_attention_layer(layer: torch.nn.Module) -> bool:
    """Whether an attention-like layer requires deferred post-load processing."""
    return isinstance(layer, AttentionLayerBase) and callable(
        getattr(layer, "process_weights_after_loading", None)
    )


__all__ = ["is_deferred_attention_layer"]
