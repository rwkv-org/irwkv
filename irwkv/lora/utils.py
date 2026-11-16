# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def parse_fine_tuned_lora_name(name: str) -> tuple[str, bool]:
    parts = name.removeprefix("base_model.model.").rsplit(".", 2)
    if len(parts) == 3:
        module, projection, suffix = parts
        if projection in ("lora_A", "lora_B") and suffix == "weight":
            return module, projection == "lora_A"
    raise ValueError(f"{name} is unsupported LoRA weight")


def is_base_embedding_weights(name: str) -> bool:
    return name.endswith(
        (".embed_tokens.base_layer.weight", ".lm_head.base_layer.weight")
    )
