# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from irwkv.exceptions import IRWKVValidationError
from irwkv.sampling_params import SamplingParams


@pytest.fixture(scope="function")
def text_llm(irwkv_runner):
    with irwkv_runner(
        "rwkv-rs/rwkv7-g1-st", enforce_eager=True, seed=0, enable_chunked_prefill=True
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while IrwkvRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="function")
def llm_for_failure_test(irwkv_runner):
    """Fixture for testing issue #26081.
    Uses a small max_model_len to easily trigger length errors.
    """
    with irwkv_runner(
        "rwkv-rs/rwkv7-g1-st",
        enforce_eager=True,
        seed=0,
        max_model_len=128,
        disable_log_stats=True,
        enable_chunked_prefill=True,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while IrwkvRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


def test_chat(text_llm):
    prompt1 = "Explain the concept of entropy."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt1},
    ]
    outputs = text_llm.chat(messages)
    assert len(outputs) == 1


def test_multi_chat(text_llm):
    prompt1 = "Explain the concept of entropy."
    prompt2 = "Explain what among us is."

    conversation1 = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt1},
    ]

    conversation2 = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt2},
    ]

    messages = [conversation1, conversation2]

    outputs = text_llm.chat(messages)
    assert len(outputs) == 2


def test_llm_chat_tokenization_matches_chat_template(text_llm):
    """LLM.chat() uses the tokenizer's rendered chat prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ]
    outputs = text_llm.chat(messages)
    assert len(outputs) == 1

    prompt_token_ids = outputs[0].prompt_token_ids
    assert prompt_token_ids is not None

    expected = text_llm.get_tokenizer().apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    assert prompt_token_ids == expected


@pytest.fixture(scope="function")
def thinking_llm(irwkv_runner):
    with irwkv_runner(
        "rwkv-rs/rwkv7-g1-st",
        max_model_len=4096,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=True,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while IrwkvRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.parametrize("fake_cot_mode", [None, False, True])
def test_chat_extra_kwargs(thinking_llm, fake_cot_mode):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is 1+1?"},
    ]

    outputs = thinking_llm.chat(
        messages,
        chat_template_kwargs=(
            {} if fake_cot_mode is None else {"fake_cot_mode": fake_cot_mode}
        ),
    )
    assert len(outputs) == 1

    prompt_token_ids = outputs[0].prompt_token_ids
    assert prompt_token_ids is not None

    prompt = thinking_llm.get_tokenizer().decode(prompt_token_ids)
    expected_suffix = "<think></think" if fake_cot_mode else "<think"
    assert prompt.endswith(expected_suffix)


def test_chat_batch_failure_cleanup(llm_for_failure_test):
    """Tests that if a batch call to llm.chat() fails mid-way
    (e.g., due to one invalid prompt), the requests that
    were already enqueued are properly aborted and do not
    pollute the queue for subsequent calls.
    (Fixes Issue #26081)
    """
    llm = llm_for_failure_test
    valid_msg = [{"role": "user", "content": "Hello"}]
    long_text = "This is a very long text to test the error " * 50
    invalid_msg = [{"role": "user", "content": long_text}]

    batch_1 = [valid_msg, valid_msg, invalid_msg]
    batch_2 = [valid_msg, valid_msg]
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    with pytest.raises(IRWKVValidationError, match="maximum context length is"):
        llm.chat(batch_1, sampling_params=sampling_params)
    assert llm.llm_engine.get_num_unfinished_requests() == 0

    outputs_2 = llm.chat(batch_2, sampling_params=sampling_params)
    assert len(outputs_2) == len(batch_2)
    assert llm.llm_engine.get_num_unfinished_requests() == 0
