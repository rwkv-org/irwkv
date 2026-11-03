# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import _get_protocol_attrs  # type: ignore

import pytest
from transformers import (
    PreTrainedTokenizerBase,
    TokenizersBackend,
)

from irwkv.tokenizers import TokenizerLike, get_tokenizer

TOKENIZER_NAME = "rwkv-rs/rwkv7-g1-st"
TOKENIZER_SUBFOLDER = "rwkv7-g1l-1.5b-20261031-ctx34304"


def _get_missing_attrs(obj: object, target: type):
    return [k for k in _get_protocol_attrs(target) if not hasattr(obj, k)]


def _assert_tokenizer_like(tokenizer: object):
    missing_attrs = _get_missing_attrs(tokenizer, TokenizerLike)
    assert not missing_attrs, f"Missing attrs: {missing_attrs}"


def test_tokenizer_like_protocol():
    tokenizer = get_tokenizer(TOKENIZER_NAME, subfolder=TOKENIZER_SUBFOLDER)
    assert isinstance(tokenizer, TokenizersBackend)
    _assert_tokenizer_like(tokenizer)


def test_tokenizer_revision():
    # Assume that "main" branch always exists
    tokenizer = get_tokenizer(
        TOKENIZER_NAME, revision="main", subfolder=TOKENIZER_SUBFOLDER
    )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match="not a valid git identifier"):
        get_tokenizer(TOKENIZER_NAME, revision="never", subfolder=TOKENIZER_SUBFOLDER)
