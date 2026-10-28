# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from irwkv.config.model import ModelConfig
from irwkv.transformers_utils import config as config_module
from irwkv.transformers_utils.config import (
    get_config,
    get_safetensors_params_metadata,
    mrope_num_dims,
    patch_legacy_rope_type,
    try_get_generation_config,
    uses_mrope,
)


def test_rwkv_config_overrides(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "rwkv", "architecture_version": "rwkv7"})
    )
    hf_config = get_config(
        tmp_path, trust_remote_code=False, hf_overrides_kw={"context_length": 2048}
    )

    assert hf_config.model_type == "rwkv"
    assert hf_config.architecture_version == "rwkv7"
    assert hf_config.context_length == 2048


def test_patch_legacy_rope_type_preserves_nope_layers():
    """NoPE layers stay disabled while later RoPE layers are normalized."""
    rope_parameters = {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "mrope_section": [24, 20, 20],
        },
    }

    patch_legacy_rope_type(rope_parameters)

    assert rope_parameters == {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
        },
    }


def test_model_config_generation_fallback_forwards_code_revision():
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="auto",
            hf_config_path=None,
            model="org/model",
            trust_remote_code=True,
            revision="model-pin",
            _hf_config_revision=None,
            code_revision="code-pin",
            config_format="auto",
            hf_token=None,
        ),
    )

    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ),
        patch.object(
            config_module,
            "get_config",
            return_value=PreTrainedConfig(),
        ) as get_config,
    ):
        ModelConfig.try_get_generation_config(model_config)

    get_config.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="model-pin",
        code_revision="code-pin",
        config_format="auto",
        token=None,
    )


def test_named_generation_config_does_not_fall_back_to_model_config():
    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ) as from_pretrained,
        patch.object(config_module, "get_config") as get_config,
    ):
        generation_config = try_get_generation_config(
            "org/model",
            trust_remote_code=False,
            config_file_name="tools_generation_config.json",
        )

    assert generation_config is None
    from_pretrained.assert_called_once_with(
        "org/model",
        revision=None,
        token=None,
        config_file_name="tools_generation_config.json",
    )
    get_config.assert_not_called()


def test_safetensors_metadata_of_repo_without_safetensors():
    """A repo storing its weights in another format is an answer, not a failure,
    so it must not be retried."""
    from huggingface_hub.errors import LocalEntryNotFoundError, NotASafetensorsRepoError

    get_safetensors_metadata = MagicMock(
        side_effect=NotASafetensorsRepoError("not a safetensors repo")
    )
    api = SimpleNamespace(
        get_safetensors_metadata=get_safetensors_metadata,
        list_repo_files=MagicMock(return_value=["pytorch_model.bin"]),
        snapshot_download=MagicMock(side_effect=LocalEntryNotFoundError("no cache")),
    )

    with patch.object(config_module, "hf_api", lambda: api):
        assert get_safetensors_params_metadata("some/pytorch-only-model") == {}

    get_safetensors_metadata.assert_called_once()


def test_safetensors_metadata_of_repo_with_a_nonstandard_file_name():
    """`get_safetensors_metadata` only knows `model.safetensors` and its index,
    so older checkpoints are read through the file listing instead."""
    from huggingface_hub.errors import NotASafetensorsRepoError
    from huggingface_hub.utils import TensorInfo

    weights = "gptq_model-4bit-128g.safetensors"
    tensor = TensorInfo(dtype="I32", shape=[5632], data_offsets=(0, 22528))
    parse_safetensors_file_metadata = MagicMock(
        return_value=SimpleNamespace(tensors={"layers.0.mlp.down_proj.qweight": tensor})
    )
    api = SimpleNamespace(
        get_safetensors_metadata=MagicMock(
            side_effect=NotASafetensorsRepoError("not a safetensors repo")
        ),
        list_repo_files=MagicMock(return_value=["config.json", weights]),
        parse_safetensors_file_metadata=parse_safetensors_file_metadata,
    )

    with patch.object(config_module, "hf_api", lambda: api):
        metadata = get_safetensors_params_metadata("some/old-gptq-model")

    assert metadata["layers.0.mlp.down_proj.qweight"]["dtype"] == "I32"
    parse_safetensors_file_metadata.assert_called_once_with(
        "some/old-gptq-model", weights, revision=None
    )


@pytest.mark.parametrize(
    ("section_key", "mrope_section", "expected_num_dims"),
    [
        ("mrope_section", [16, 24, 24], 3),
        ("mrope_section", [16, 16, 16, 16], 4),
        # Interleaved M-RoPE takes 2 sections but still consumes 3D positions
        ("mrope_section", [32, 32], 3),
        # HunYuan-VL checkpoints ship the section under its legacy name
        ("xdrope_section", [16, 16, 16, 16], 4),
    ],
)
def test_mrope_num_dims(section_key, mrope_section, expected_num_dims):
    config = PreTrainedConfig()
    config.rope_parameters = {"rope_type": "default", section_key: mrope_section}

    assert uses_mrope(config)
    assert mrope_num_dims(config) == expected_num_dims


@pytest.mark.parametrize("section_name", ["mrope_section", "xdrope_section"])
def test_mrope_num_dims_from_config_attribute(section_name):
    """Some configs expose the section as an attribute rather than under
    `rope_parameters`."""
    config = PreTrainedConfig()
    setattr(config, section_name, [16, 16, 16, 16])

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_from_nested_rope_parameters():
    """Sections nested by layer type must be found, not silently defaulted."""
    config = PreTrainedConfig()
    config.rope_parameters = {
        "full_attention": {"mrope_section": [16, 16, 16, 16]},
        "linear_attention": {"rope_type": "default"},
    }

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_without_mrope():
    assert mrope_num_dims(PreTrainedConfig()) == 0
