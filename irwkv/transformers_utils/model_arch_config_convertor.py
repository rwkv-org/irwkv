# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import final

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PreTrainedConfig

from irwkv.config.model_arch import (
    ModelArchitectureConfig,
)
from irwkv.transformers_utils.config import (
    ConfigFormat,
    get_safetensors_params_metadata,
)
from irwkv.utils.torch_utils import common_broadcastable_dtype


class ModelArchConfigConvertorBase:
    def __init__(
        self,
        hf_config: PreTrainedConfig,
        hf_text_config: PreTrainedConfig,
    ):
        self.hf_config = hf_config
        self.hf_text_config = hf_text_config

    def get_per_layer_hf_configs(self) -> None:
        return None

    def get_architectures(self) -> list[str]:
        # Sometimes we get here from `irwkv_config.with_hf_config(text_config)` where
        # `text_config` is a sub-config from a multi-modal model. If this is the case,
        # the sub-config will not have `architectures` and it will explicitly be `None`
        return getattr(self.hf_config, "architectures", None) or []

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_hidden_layers", 0)

    def get_total_num_attention_heads(self) -> int:
        return getattr(self.hf_text_config, "num_attention_heads", 0)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    def get_head_size(self) -> int:
        return self.hf_text_config.head_size

    def get_total_num_kv_heads(self) -> int:
        return 0

    @final
    @classmethod
    def get_torch_dtype(
        cls,
        hf_config: PreTrainedConfig,
        model_id: str,
        revision: str | None,
        config_format: str | ConfigFormat,
    ):
        # NOTE: getattr(config, "dtype", torch.float32) is not correct
        # because config.dtype can be None.
        config_dtype = getattr(hf_config, "dtype", None)

        # Fallbacks for multi-modal models if the root config
        # does not define dtype
        if config_dtype is None:
            config_dtype = getattr(hf_config.get_text_config(), "dtype", None)
        if config_dtype is None and hasattr(hf_config, "vision_config"):
            config_dtype = getattr(hf_config.vision_config, "dtype", None)
        if config_dtype is None and hasattr(hf_config, "encoder_config"):
            config_dtype = getattr(hf_config.encoder_config, "dtype", None)

        # Try to read the dtype of the weights if they are in safetensors format
        if config_dtype is None:
            param_mt = get_safetensors_params_metadata(model_id, revision=revision)

            if param_mt:
                param_dtypes: set[torch.dtype] = {
                    _SAFETENSORS_TO_TORCH_DTYPE[dtype]
                    for info in param_mt.values()
                    if (dtype := info.get("dtype", None))
                    and dtype in _SAFETENSORS_TO_TORCH_DTYPE
                }

                if param_dtypes:
                    return common_broadcastable_dtype(param_dtypes)

        if config_dtype is None:
            config_dtype = torch.float32

        return config_dtype

    def get_quantization_config(self):
        return getattr(self.hf_config, "quantization_config", None)

    def rswa_window(self) -> int | None:
        value = getattr(self.hf_config, "rswa_window", None)
        if value is None:
            return None
        return int(value)

    def derive_max_model_len_and_key(self) -> tuple[float, str | None]:
        keys = (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
            "model_max_length",
            "max_target_positions",
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        )
        lengths = [
            (value, key)
            for key in keys
            if (value := getattr(self.hf_text_config, key, None)) is not None
        ]
        return min(lengths) if lengths else (float("inf"), None)

    def convert(self) -> ModelArchitectureConfig:
        return self.convert_layer()

    def convert_layer(self) -> ModelArchitectureConfig:
        """Convert one homogeneous config, without resolving per-layer values.

        `convert` calls this once per layer, so it must not recurse back into
        per-layer resolution: the configs it receives are already layer-specific,
        and they can still carry the attributes that made the model look
        heterogeneous in the first place.
        """
        model_arch_config = ModelArchitectureConfig(
            architectures=self.get_architectures(),
            model_type=self.hf_config.model_type,
            text_model_type=getattr(self.hf_text_config, "model_type", None),
            hidden_size=self.get_hidden_size(),
            total_num_hidden_layers=self.get_num_hidden_layers(),
            total_num_attention_heads=self.get_total_num_attention_heads(),
            head_size=self.get_head_size(),
            vocab_size=self.get_vocab_size(),
            total_num_kv_heads=self.get_total_num_kv_heads(),
            quantization_config=self.get_quantization_config(),
            is_deepseek_mla=False,
            is_mm_prefix_lm=False,
            rswa_window=self.rswa_window(),
            derived_max_model_len_and_key=self.derive_max_model_len_and_key(),
        )

        return model_arch_config
