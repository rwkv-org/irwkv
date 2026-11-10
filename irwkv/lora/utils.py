# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from irwkv.model_executor.models.utils import WeightsMapper


def parse_fine_tuned_lora_name(
    name: str, weights_mapper: "WeightsMapper | None" = None
) -> tuple[str, bool]:
    if name.startswith("base_model.model."):
        name = name.removeprefix("base_model.model.")
        if weights_mapper:
            mapped_name = weights_mapper._map_name(name)
            if mapped_name is None:
                raise ValueError("Mapped LoRA weight name cannot be None.")
            name = "base_model.model." + mapped_name
        else:
            name = "base_model.model." + name
    elif weights_mapper:
        mapped_name = weights_mapper._map_name(name)
        if mapped_name is None:
            raise ValueError("Mapped LoRA weight name cannot be None.")
        name = mapped_name

    start_index = 2 if name.startswith("base_model.model.") else 0
    parts = name.split(".")
    if len(parts) >= 2 and parts[-1] in ("weight", "bias"):
        if parts[-2] in ("lora_A", "lora_B") and parts[-1] == "weight":
            return ".".join(parts[start_index:-2]), parts[-2] == "lora_A"
        if parts[-2] in ("score", "classifier"):
            return parts[-2], False
    if parts[-1] in ("lora_embedding_A", "lora_embedding_B"):
        return ".".join(parts[start_index:-1]), parts[-1] == "lora_embedding_A"
    raise ValueError(f"{name} is unsupported LoRA weight")


def is_base_embedding_weights(name: str) -> bool:
    return name.endswith(
        (".embed_tokens.base_layer.weight", ".lm_head.base_layer.weight")
    )
