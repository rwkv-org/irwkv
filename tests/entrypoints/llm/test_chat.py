# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from irwkv.entrypoints.llm import LLM
from irwkv.exceptions import IRWKVValidationError
from irwkv.sampling_params import SamplingParams


@pytest.fixture
def offline_chat():
    profiles = {
        name: {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": -1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.09,
            "penalty_decay": 0.994,
        }
        for name, temperature in (
            ("generation_config.json", 0.96),
            ("fake_think_generation_config.json", 1.0),
            ("tools_generation_config.json", 0.8),
        )
    }
    llm = LLM.__new__(LLM)
    llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="rwkv"),
        generation_config="auto",
        get_diff_sampling_param=MagicMock(
            side_effect=lambda config_file_name="generation_config.json": profiles[
                config_file_name
            ]
        ),
    )
    llm.default_sampling_params = None
    llm._run_chat = MagicMock()
    llm._add_chat_requests = MagicMock()
    return llm


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
@pytest.mark.parametrize(
    ("template_kwargs", "tools", "temperature"),
    [
        ({}, None, 0.96),
        ({"fake_cot_mode": True}, None, 1.0),
        ({"fake_cot_mode": True}, [{"type": "function"}], 0.8),
    ],
)
def test_rwkv_chat_selects_sampling_profile(
    offline_chat, method, template_kwargs, tools, temperature
):
    getattr(offline_chat, method)(
        [{"role": "user", "content": "Hello"}],
        tools=tools,
        chat_template_kwargs=template_kwargs,
    )
    run = (
        offline_chat._run_chat if method == "chat" else offline_chat._add_chat_requests
    )
    params = run.call_args.kwargs["params"]
    assert (params.temperature, params.top_p, params.top_k) == (temperature, 0.9, -1)
    assert (
        params.presence_penalty,
        params.frequency_penalty,
        params.penalty_decay,
    ) == (
        1.0,
        0.09,
        0.994,
    )


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
def test_rwkv_chat_preserves_explicit_sampling(offline_chat, method):
    params = SamplingParams(temperature=0.5)
    getattr(offline_chat, method)(
        [{"role": "user", "content": "Hello"}], sampling_params=params
    )
    run = (
        offline_chat._run_chat if method == "chat" else offline_chat._add_chat_requests
    )
    assert run.call_args.kwargs["params"] is params
    offline_chat.model_config.get_diff_sampling_param.assert_not_called()


@pytest.mark.parametrize("generation_config", ["irwkv", "auto"])
def test_chat_preserves_defaults_when_rwkv_profiles_disabled(
    offline_chat, generation_config
):
    offline_chat.model_config.generation_config = generation_config
    if generation_config == "auto":
        offline_chat.model_config.hf_config.model_type = "other"
    offline_chat.chat(
        [{"role": "user", "content": "Hello"}],
        chat_template_kwargs={"fake_cot_mode": True},
    )
    assert offline_chat._run_chat.call_args.kwargs["params"].temperature == 0.96
    offline_chat.model_config.get_diff_sampling_param.assert_called_once_with()


def test_rwkv_chat_rejects_incomplete_sampling_profiles(offline_chat):
    offline_chat.model_config.get_diff_sampling_param.side_effect = None
    offline_chat.model_config.get_diff_sampling_param.return_value = {}
    with pytest.raises(ValueError, match="generation_config.json.*missing"):
        offline_chat.chat([{"role": "user", "content": "Hello"}])


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
