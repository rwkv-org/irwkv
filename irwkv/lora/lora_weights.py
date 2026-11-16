# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from irwkv.lora.peft_helper import PEFTHelper


class LoRALayerWeights:
    """LoRA weights for a layer composed of two low rank matrices."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alpha: int,
        scaling: float,
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = scaling
        self.lora_a: torch.Tensor | None = None
        self.lora_b: torch.Tensor | None = None

    @classmethod
    def from_config(
        cls, module_name: str, peft_helper: PEFTHelper
    ) -> "LoRALayerWeights":
        return cls(
            module_name,
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.vllm_lora_scaling_factor,
        )
