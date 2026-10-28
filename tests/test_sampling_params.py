# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from irwkv import SamplingParams
from irwkv.exceptions import IRWKVValidationError


@dataclass
class MockModelConfig:
    hf_config = SimpleNamespace(model_type="rwkv")
    max_logprobs: int = 20
    logits_processors: list | None = None
    return_sampling_mask: bool = False

    def get_vocab_size(self) -> int:
        return 1024


def test_verify_leaves_logits_processors_to_admission():
    """verify() is runner-agnostic; LP validation lives in the admission
    layer, so an unimportable FQCN must not fail verify()."""
    SamplingParams().verify(
        MockModelConfig(logits_processors=["no.such:Cls"]), None, None
    )


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(IRWKVValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]
