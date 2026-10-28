# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from irwkv.exceptions import IRWKVValidationError


@pytest.fixture(scope="module")
def llm(irwkv_runner):
    with irwkv_runner(
        "rwkv-rs/rwkv7-g1-st",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=128,
        max_num_seqs=4,
    ) as runner:
        yield weakref.proxy(runner.llm)


def test_empty_prompt(llm):
    with pytest.raises(IRWKVValidationError, match="decoder prompt cannot be empty"):
        llm.generate({"prompt_token_ids": []})


def test_out_of_vocab_token(llm):
    with pytest.raises(IRWKVValidationError, match="out of vocabulary"):
        llm.generate({"prompt_token_ids": [999999]})
