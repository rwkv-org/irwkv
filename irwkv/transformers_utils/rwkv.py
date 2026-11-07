# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV artifact configuration without a model framework dependency."""

import copy
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
RWKV_EOS_TOKEN = "✿"
RWKV_EOS_TOKEN_ID = 10060
RWKV_MAX_TOKEN_ID = 65529


def load_json(
    model: str | Path,
    filename: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    **kwargs,
) -> dict[str, Any]:
    path = Path(model) / filename
    if not path.is_file():
        if Path(model).exists():
            raise OSError(f"Missing {filename} in {model}")
        path = Path(
            hf_hub_download(
                str(model),
                filename,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


class RwkvConfig:
    """The configuration fields consumed by the native RWKV-7 model."""

    model_type = "rwkv"
    sub_configs: dict[str, type] = {}

    def __init__(self, **values):
        defaults = {
            "architectures": ["RwkvForCausalLM"],
            "architecture_version": "rwkv7",
            "vocab_size": 65536,
            "context_length": 4096,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "head_size": 64,
            "layer_norm_epsilon": 1e-5,
            "group_norm_epsilon": 64e-5,
            "bos_token_id": None,
            "eos_token_id": RWKV_EOS_TOKEN_ID,
            "pad_token_id": None,
            "tie_word_embeddings": False,
            "is_encoder_decoder": False,
            "use_cache": True,
            "wkv_mode": "fp32io16",
            "quantization_config": None,
            "_name_or_path": "",
        }
        self.__dict__.update(defaults | values)
        self.bos_token_id = None
        self.eos_token_id = RWKV_EOS_TOKEN_ID
        if self.model_type != "rwkv" or self.architecture_version != "rwkv7":
            raise ValueError("irwkv requires a canonical RWKV-7 config")
        if self.head_size != 64 or self.hidden_size <= 0:
            raise ValueError("RWKV requires positive hidden_size and head_size=64")
        if self.hidden_size % self.head_size:
            raise ValueError("RWKV hidden_size must be divisible by head_size")
        self.intermediate_size = values.get("intermediate_size")
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        self.num_attention_heads = (
            self.hidden_size // self.head_size
            if values.get("num_attention_heads") is None
            else values["num_attention_heads"]
        )
        for name, scale in (
            ("decay_low_rank_dim", 2.5),
            ("a_low_rank_dim", 2.5),
            ("v_low_rank_dim", 1.7),
            ("gate_low_rank_dim", 5.0),
        ):
            rank = max(32, round(scale * self.hidden_size**0.5 / 32) * 32)
            setattr(self, name, rank if values.get(name) is None else values[name])
            if getattr(self, name) <= 0:
                raise ValueError(f"RWKV {name} must be positive")
        if self.num_hidden_layers <= 0 or self.context_length <= 0:
            raise ValueError("RWKV layer count and context length must be positive")
        if self.layer_norm_epsilon <= 0 or self.group_norm_epsilon <= 0:
            raise ValueError("RWKV normalization epsilons must be positive")
        if self.wkv_mode not in ("fp32io16", "fp16"):
            raise ValueError("RWKV wkv_mode must be fp32io16 or fp16")
        if self.tie_word_embeddings or self.pad_token_id is not None:
            raise ValueError("RWKV requires an untied head and no padding token")
        legacy = {
            "attention_hidden_size",
            "rescale_every",
            "wkv_state_dtype",
        } & values.keys()
        if legacy:
            raise ValueError(
                f"Legacy RWKV configuration fields are unsupported: {sorted(legacy)}"
            )
        if self.intermediate_size != 4 * self.hidden_size:
            raise ValueError("RWKV intermediate_size must equal 4 * hidden_size")
        if self.num_attention_heads != self.hidden_size // self.head_size:
            raise ValueError("RWKV num_attention_heads must equal hidden_size / 64")
        dtype = values.get("dtype", values.get("torch_dtype", "bfloat16"))
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    @property
    def max_position_embeddings(self) -> int:
        return self.context_length

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    def get_text_config(self, **kwargs) -> "RwkvConfig":
        return self

    def update(self, values: dict[str, Any]) -> None:
        self.__dict__.update(values)

    def to_dict(self) -> dict[str, Any]:
        values = copy.deepcopy(self.__dict__)
        values["model_type"] = self.model_type
        for key, value in values.items():
            if isinstance(value, torch.dtype):
                values[key] = str(value).removeprefix("torch.")
        return values

    def to_json_string(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def get_config_dict(cls, model, **kwargs):
        return load_json(model, "config.json", **kwargs), kwargs

    @classmethod
    def from_pretrained(cls, model, **kwargs) -> "RwkvConfig":
        values = load_json(model, "config.json", **kwargs)
        values["_name_or_path"] = str(model)
        return cls(**values)


class GenerationConfig:
    """Sampling defaults stored alongside a converted RWKV model."""

    def __init__(self, **values):
        self.__dict__.update(values)
        self.bos_token_id = None
        self.eos_token_id = RWKV_EOS_TOKEN_ID
        self.stop_strings = [RWKV_EOS_TOKEN]

    def to_diff_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_pretrained(
        cls, model, config_file_name="generation_config.json", **kwargs
    ):
        return cls(**load_json(model, config_file_name, **kwargs))

    @classmethod
    def from_model_config(cls, config: RwkvConfig):
        names = ("bos_token_id", "eos_token_id", "pad_token_id")
        return cls(**{name: getattr(config, name) for name in names})


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible ByteLevel vocabulary alphabet."""
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = iter(range(256, 512))
    return {byte: chr(byte if byte in direct else next(extra)) for byte in range(256)}
