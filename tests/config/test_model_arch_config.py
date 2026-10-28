# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelArchitectureConfig and its integration with ModelConfig."""

from copy import copy

import pytest
from transformers import PreTrainedConfig

from irwkv.config import ModelConfig
from irwkv.config.model_arch import ModelArchitectureConfig
from irwkv.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


def test_head_size_falls_back_when_head_dim_is_zero():
    """Regression test for configs that materialize missing head_dim as 0."""
    hf_config = PreTrainedConfig(
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=10,
        head_dim=0,
        kv_lora_rank=None,
    )

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_head_size() == 128


def test_legacy_modelopt_config_without_producer_is_normalized():
    quantization_config = {
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "modelopt_quant_config": {"quant_cfg": {}},
        }
    }
    hf_config = PreTrainedConfig(quantization_config=quantization_config)

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_quantization_config()["quant_method"] == "modelopt_fp4"


def _layer(**overrides) -> ModelArchitectureConfig:
    fields = dict(
        architectures=["X"],
        model_type="x",
        text_model_type=None,
        hidden_size=64,
        total_num_hidden_layers=3,
        total_num_attention_heads=8,
        head_size=16,
        vocab_size=32,
        total_num_kv_heads=4,
        num_experts=0,
        num_experts_per_token=0,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=False,
        rswa_window=None,
        derived_max_model_len_and_key=(1.0, None),
    )
    return ModelArchitectureConfig(**(fields | overrides))


def test_from_layers_collapses_to_the_largest_layer():
    """The whole-model config must size buffers for every layer."""
    arch = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32, total_num_kv_heads=8)]
    )

    assert (arch.head_size, arch.total_num_kv_heads) == (32, 8)
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]
    assert [arch[i].total_num_kv_heads for i in range(3)] == [4, 4, 8]
    # A layer view is itself homogeneous, so reading from it cannot recurse.
    assert arch[0].per_layer_overrides is None


def test_layer_view_follows_later_edits_to_the_whole_model_config():
    """`model_arch_config` is edited after construction in a few places."""
    arch = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32)]
    )

    arch.is_mm_prefix_lm = True

    assert all(arch[i].is_mm_prefix_lm for i in range(3))
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]


def test_uniform_layers_stay_homogeneous():
    """A checkpoint can vary attributes irwkv never reads."""
    arch = ModelArchitectureConfig.from_layers([_layer()] * 3)

    assert arch.per_layer_overrides is None
    assert arch[1] is arch


def test_getitem_rejects_negative_indices():
    """Wrapping would only misbehave on heterogeneous models, so reject both."""
    homogeneous = _layer()
    heterogeneous = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32)]
    )

    for arch in (homogeneous, heterogeneous):
        with pytest.raises(IndexError):
            arch[-1]


@pytest.mark.parametrize(
    "varying",
    [
        # `bool` is an `int`; collapsing this one with `max` would make `use_mla`
        # true model wide and silently discard every per-layer KV head count.
        {"is_deepseek_mla": True},
        {"quantization_config": {"quant_method": "fp8"}},
        {"rswa_window": 512},
    ],
)
def test_from_layers_rejects_fields_with_no_whole_model_value(varying: dict):
    with pytest.raises(ValueError, match="varies across layers"):
        ModelArchitectureConfig.from_layers([_layer(), _layer(), _layer(**varying)])


def test_from_layers_rejects_a_layer_count_mismatch():
    """Draft configs can inherit a target's per-layer spec."""
    with pytest.raises(ValueError, match="per-layer configs"):
        ModelArchitectureConfig.from_layers([_layer(), _layer(head_size=32)])


class _HeterogeneousConfig(PreTrainedConfig):
    """A heterogeneous config with no convertor of its own.

    Mirrors the parts irwkv uses: per-layer configs are shallow copies with the
    varying attributes applied and heterogeneity stripped, so they do not recurse."""

    is_heterogeneous = True

    def __init__(self, per_layer: dict[str, list], **kwargs):
        # Set before `super().__init__` because it validates `per_layer_config`
        self._per_layer = per_layer
        super().__init__(**kwargs)

    @property
    def per_layer_config(self) -> list[PreTrainedConfig]:
        layers = []
        for i in range(self.num_hidden_layers):
            layer = copy(self)
            layer.is_heterogeneous = False
            for name, values in self._per_layer.items():
                setattr(layer, name, values[i])
            layers.append(layer)
        return layers


def test_transformers_heterogeneous_config_is_resolved_per_layer():
    hf_config = _HeterogeneousConfig(
        per_layer={"head_dim": [16, 16, 32], "num_key_value_heads": [4, 4, 8]},
        num_hidden_layers=3,
        hidden_size=64,
        num_attention_heads=8,
    )

    arch = ModelArchConfigConvertorBase(hf_config, hf_config).convert()

    assert (arch.head_size, arch.total_num_kv_heads) == (32, 8)
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]
    assert [arch[i].total_num_kv_heads for i in range(3)] == [4, 4, 8]


def test_heterogeneous_config_varying_nothing_irwkv_reads():
    """Transformers prunes nothing for us: the diff has to notice."""
    hf_config = _HeterogeneousConfig(
        per_layer={"bos_token_id": [1, 2, 3]},
        num_hidden_layers=3,
        hidden_size=64,
        num_attention_heads=8,
        head_dim=16,
    )

    arch = ModelArchConfigConvertorBase(hf_config, hf_config).convert()

    assert arch.per_layer_overrides is None
    assert arch[2] is arch


def test_rwkv_model_arch_config():
    model_config = ModelConfig("rwkv-rs/rwkv7-g1-st")
    hf_config = model_config.hf_config
    arch = model_config.model_arch_config

    assert arch.architectures == ["RwkvForCausalLM"]
    assert arch.model_type == arch.text_model_type == "rwkv"
    assert arch.hidden_size == model_config.get_hidden_size() == hf_config.hidden_size
    assert arch.total_num_hidden_layers == hf_config.num_hidden_layers
    assert model_config.get_total_num_hidden_layers() == hf_config.num_hidden_layers
    assert arch.total_num_attention_heads == hf_config.num_attention_heads
    assert arch.head_size == model_config.get_head_size() == hf_config.head_size
    assert arch.vocab_size == model_config.get_vocab_size() == hf_config.vocab_size
    assert arch.total_num_kv_heads == model_config.get_total_num_kv_heads()
    assert arch.num_experts == model_config.get_num_experts() == 0
    assert arch.num_experts_per_token == 0
    assert not arch.is_deepseek_mla
    assert not arch.is_mm_prefix_lm
