# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv import SamplingParams
from irwkv.entrypoints.common.offline import OfflineInferenceMixin
from irwkv.exceptions import IRWKVValidationError


@pytest.fixture
def mixin() -> OfflineInferenceMixin:
    return object.__new__(OfflineInferenceMixin)


def test_rejects_mismatched_params(mixin: OfflineInferenceMixin):
    with pytest.raises(IRWKVValidationError) as exc_info:
        mixin._params_to_seq([SamplingParams()], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and params (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_priority(mixin: OfflineInferenceMixin):
    with pytest.raises(IRWKVValidationError) as exc_info:
        mixin._priority_to_seq([0], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and priority (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_matching_lengths_pass_through(mixin: OfflineInferenceMixin):
    assert mixin._params_to_seq([SamplingParams()], num_requests=1) == [
        SamplingParams()
    ]
    assert mixin._priority_to_seq([3], num_requests=1) == [3]
