# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from irwkv.entrypoints.llm import LLM
from irwkv.exceptions import IRWKVValidationError
from irwkv.sampling_params import SamplingParams
from irwkv.utils.counter import Counter


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
@pytest.mark.parametrize(
    ("template_kwargs", "tools", "profile"),
    [
        ({}, None, "generation_config.json"),
        ({"fake_cot_mode": False}, None, "generation_config.json"),
        ({"fake_cot_mode": True}, None, "fake_think_generation_config.json"),
        (
            {"fake_cot_mode": True},
            [{"type": "function"}],
            "tools_generation_config.json",
        ),
    ],
)
@pytest.mark.parametrize(
    "open_defaults",
    [
        {"top_p": 0.76, "top_k": 32, "frequency_penalty": 0.1, "penalty_decay": 0.988},
        {"top_p": 0.90, "top_k": -1, "frequency_penalty": 0.09, "penalty_decay": 0.994},
    ],
    ids=["g1k", "g1l"],
)
def test_rwkv_chat_selects_sampling_profile(
    method, template_kwargs, tools, profile, open_defaults
):
    profiles = {
        "generation_config.json": {
            "temperature": 0.96,
            "presence_penalty": 1.0,
            "stop": ["END"],
            **open_defaults,
        },
        "fake_think_generation_config.json": {
            "temperature": 1.0,
            "top_p": 0.28,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "stop": ["END"],
        },
        "tools_generation_config.json": {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_decay": 0.996,
            "stop": ["END"],
        },
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
    messages = [{"role": "user", "content": "Hello"}]

    getattr(llm, method)(messages, tools=tools, chat_template_kwargs=template_kwargs)
    run = llm._run_chat if method == "chat" else llm._add_chat_requests
    params = run.call_args.kwargs["params"]
    for key, expected in profiles[profile].items():
        assert getattr(params, key) == expected

    explicit = SamplingParams(temperature=0.5)
    getattr(llm, method)(messages, sampling_params=explicit)
    assert run.call_args.kwargs["params"] is explicit
    assert llm.model_config.get_diff_sampling_param.call_count == 3


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
@pytest.mark.parametrize(
    ("model_type", "generation_config"), [("other", "auto"), ("rwkv", "irwkv")]
)
def test_chat_preserves_default_sampling_when_rwkv_profiles_disabled(
    method, model_type, generation_config
):
    llm = LLM.__new__(LLM)
    llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type),
        generation_config=generation_config,
        get_diff_sampling_param=MagicMock(return_value={"temperature": 0.7}),
    )
    llm.default_sampling_params = None
    llm._run_chat = MagicMock()
    llm._add_chat_requests = MagicMock()

    getattr(llm, method)(
        [{"role": "user", "content": "Hello"}],
        chat_template_kwargs={"fake_cot_mode": True},
    )
    run = llm._run_chat if method == "chat" else llm._add_chat_requests
    assert run.call_args.kwargs["params"].temperature == 0.7
    llm.model_config.get_diff_sampling_param.assert_called_once_with()


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
def test_chat_explicit_sampling_does_not_load_profiles(method):
    llm = LLM.__new__(LLM)
    llm.model_config = MagicMock()
    llm.default_sampling_params = None
    llm._run_chat = MagicMock()
    llm._add_chat_requests = MagicMock()
    params = [SamplingParams(temperature=0.5)]

    getattr(llm, method)([[{"role": "user", "content": "Hello"}]], params)
    run = llm._run_chat if method == "chat" else llm._add_chat_requests
    assert run.call_args.kwargs["params"] is params
    llm.model_config.get_diff_sampling_param.assert_not_called()


@pytest.mark.parametrize("method", ["chat", "enqueue_chat"])
def test_rwkv_chat_rejects_incomplete_sampling_profiles(method):
    llm = LLM.__new__(LLM)
    llm.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="rwkv"),
        generation_config="auto",
        get_diff_sampling_param=MagicMock(return_value={}),
    )
    llm.default_sampling_params = None

    with pytest.raises(ValueError, match="generation_config.json.*missing"):
        getattr(llm, method)([{"role": "user", "content": "Hello"}])


def test_enqueue_returns_external_request_id():
    llm = LLM.__new__(LLM)
    llm.request_counter = Counter()
    llm.llm_engine = MagicMock()
    llm.llm_engine.add_request.return_value = "0-internal"

    assert llm._add_request({"prompt_token_ids": [1]}, SamplingParams()) == "0"
    assert llm.llm_engine.add_request.call_args.args[0] == "0"


def test_enqueue_aborts_external_ids_if_rendering_fails():
    llm = LLM.__new__(LLM)
    llm.llm_engine = MagicMock()
    llm._add_request = MagicMock(return_value="0")

    def prompts():
        yield {"prompt_token_ids": [1]}
        raise ValueError("render failed")

    with pytest.raises(ValueError, match="render failed"):
        llm._render_and_add_requests(prompts(), [SamplingParams(), SamplingParams()])
    llm.llm_engine.abort_request.assert_called_once_with(["0"])


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
