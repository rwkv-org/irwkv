# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.model_executor.models.registry import ModelRegistry


def test_registry_loads_only_rwkv() -> None:
    assert set(ModelRegistry.get_supported_archs()) == {"RwkvForCausalLM"}
    model_cls = ModelRegistry._try_load_model_cls("RwkvForCausalLM")
    assert model_cls is not None
    assert model_cls.__name__ == "RwkvForCausalLM"


def test_registry_rejects_other_architectures() -> None:
    with pytest.raises(ValueError, match="not supported"):
        ModelRegistry.resolve_model_cls("LlamaForCausalLM", None)
