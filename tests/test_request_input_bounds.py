# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds and alignment checks for request-controlled inputs.

Covers prompt-mask alignment, stop-string caps, bad-words dedup/tokenization
limits, stop-token-id dedup, beam-width/sequence caps, and the DeepSeek
history-scan bound.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from typing import Protocol
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

import irwkv.envs as envs
from irwkv import SamplingParams
from irwkv.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from irwkv.entrypoints.openai.completion.protocol import CompletionRequest
from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest
from irwkv.exceptions import IRWKVValidationError
from irwkv.inputs.engine import embeds_input
from irwkv.sampling_params import BeamSearchParams
from irwkv.v1.engine.input_processor import InputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def prompt_embeds_processor():
    model_config = Mock(
        max_model_len=16,
        max_logprobs=20,
        logits_processors=None,
        return_sampling_mask=False,
        supports_multimodal_inputs=False,
    )
    model_config.get_vocab_size.return_value = 32
    model_config.try_get_generation_config.return_value = {}
    config = Mock(model_config=model_config, structured_outputs_config=None)
    renderer = Mock(tokenizer=None)
    renderer.get_eos_token_id.return_value = None
    return InputProcessor(config, renderer, mm_registry=Mock())


@pytest.mark.parametrize("mask_len", [0, 1, 2, 4])
def test_prompt_embeds_mask_length_rejected_before_engine_submission(
    prompt_embeds_processor, mask_len
):
    """Reject short, long, and broadcastable masks before they reach a worker."""
    prompt = embeds_input(
        torch.zeros(3, 4),
        prompt_token_ids=[1, 2, 3],
        is_token_ids=[False] * mask_len,
    )

    with pytest.raises(IRWKVValidationError, match="prompt_is_token_ids") as exc_info:
        prompt_embeds_processor.process_inputs(
            "invalid", prompt, SamplingParams(max_tokens=1), ("generate",)
        )

    assert exc_info.value.parameter == "prompt_is_token_ids"


@pytest.mark.parametrize(
    ("token_ids", "mask"),
    [(None, None), ([1, 2, 3], None), ([1, 2, 3], [True, False, True])],
    ids=["pure-embeds", "token-ids-without-mask", "mixed"],
)
def test_prompt_embeds_valid_inputs_preserved(prompt_embeds_processor, token_ids, mask):
    embeds = torch.zeros(3, 4)
    prompt = embeds_input(embeds, prompt_token_ids=token_ids, is_token_ids=mask)

    request = prompt_embeds_processor.process_inputs(
        "valid", prompt, SamplingParams(max_tokens=1), ("generate",)
    )

    assert request.prompt_embeds is embeds
    assert request.prompt_token_ids == token_ids
    assert request.prompt_is_token_ids == mask


def _process_inputs_with_max_model_len(
    params: SamplingParams, max_model_len: int = 2048
):
    processor = Mock(spec=InputProcessor)
    processor.tokenizer = None
    processor.generation_config_fields = {}
    processor.renderer = Mock()
    processor.renderer.get_eos_token_id.return_value = None
    processor.irwkv_config = Mock()
    processor.model_config = Mock()
    processor.model_config.max_model_len = max_model_len
    prompt = {"type": "token", "prompt_token_ids": [1, 2, 3]}
    return InputProcessor.process_inputs(
        processor, "req", prompt, params, ("generate",)
    )


def test_unset_max_tokens_rejects_oversized_min_tokens():
    """Reject min_tokens that only becomes invalid after max_tokens is filled."""
    params = SamplingParams(max_tokens=None, min_tokens=2147483648)
    with pytest.raises(
        IRWKVValidationError,
        match="min_tokens must be less than or equal to max_tokens",
    ):
        _process_inputs_with_max_model_len(params)


def test_unset_max_tokens_accepts_min_tokens_within_filled_max():
    params = SamplingParams(max_tokens=None, min_tokens=10)
    request = _process_inputs_with_max_model_len(params)
    assert request.sampling_params.max_tokens == 2045
    assert request.sampling_params.min_tokens == 10


# --- Stop strings: public requests cap the number of stop strings ---------


class _StopRequest(Protocol):
    stop: str | list[str] | None


def _completion_request(stop: list[str]) -> _StopRequest:
    return CompletionRequest(model="test-model", prompt="hello", stop=stop)


def _chat_request(stop: list[str]) -> _StopRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stop=stop,
    )


def _batch_chat_request(stop: list[str]) -> _StopRequest:
    return BatchChatCompletionRequest(
        model="test-model",
        messages=[[{"role": "user", "content": "hello"}]],
        stop=stop,
    )


def _responses_request(stop: list[str]) -> _StopRequest:
    return ResponsesRequest(model="test-model", input="hello", stop=stop)


REQUEST_BUILDERS: list[Callable[[list[str]], _StopRequest]] = [
    _completion_request,
    _chat_request,
    _batch_chat_request,
    _responses_request,
]


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_accept_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    stop = ["one", "two", "three", "four"]

    request = build_request(stop)

    assert request.stop == stop


@pytest.mark.parametrize("build_request", REQUEST_BUILDERS)
def test_public_requests_reject_more_than_four_stop_strings(
    build_request: Callable[[list[str]], _StopRequest],
):
    with pytest.raises(ValidationError, match="at most 4"):
        build_request(["one", "two", "three", "four", "five"])


def test_stop_string_limit_can_be_overridden():
    env = os.environ.copy()
    env["IRWKV_MAX_STOP_STRINGS"] = "1"
    code = """
from pydantic import ValidationError
from irwkv.entrypoints.openai.completion.protocol import CompletionRequest

try:
    CompletionRequest(
        model="test-model",
        prompt="hello",
        stop=["one", "two"],
    )
except ValidationError as error:
    assert "at most 1" in str(error)
else:
    raise AssertionError("configured stop-string limit was not enforced")
"""

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- Stop token ids: duplicates are deduplicated in order ------------------


def test_duplicate_stop_token_ids_are_deduplicated_in_order():
    params = SamplingParams(stop_token_ids=[42, 7, 42, 9, 7])

    assert params.stop_token_ids == [42, 7, 9]
    assert params.all_stop_token_ids == {7, 9, 42}


# --- Bad words: dedup, and the tokenization pass is bounded ----------------


class MockTokenizer:
    max_token_id = 1024

    def __init__(self):
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return [2] if text.startswith(" ") else [1]


def test_duplicate_bad_words_are_deduplicated_in_order():
    params = SamplingParams(bad_words=["bad", "worse", "bad", "worst"])

    assert params.bad_words == ["bad", "worse", "worst"]


def test_bad_word_tokenization_stops_at_worker_limit():
    params = SamplingParams(bad_words=[f"word-{i}" for i in range(65)])
    tokenizer = MockTokenizer()

    with pytest.raises(IRWKVValidationError, match="Too many bad words"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 129


def test_bad_word_tokenization_limit_can_be_overridden(monkeypatch):
    monkeypatch.setenv("IRWKV_MAX_NUM_BAD_WORDS", "2")
    params = SamplingParams(bad_words=["bad", "worse"])
    tokenizer = MockTokenizer()

    with pytest.raises(IRWKVValidationError, match="The max number is 2"):
        params.update_from_tokenizer(tokenizer)

    assert tokenizer.calls == 3


class EmptyBaseEncodingTokenizer:
    max_token_id = 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [216] if text.startswith(" ") else []


def test_bad_word_rejects_empty_base_tokenization():
    params = SamplingParams(bad_words=["\x16"])

    with pytest.raises(
        IRWKVValidationError,
        match="must tokenize to at least one token",
    ) as exc_info:
        params.update_from_tokenizer(EmptyBaseEncodingTokenizer())

    assert exc_info.value.parameter == "bad_words"


class EmptyPrefixedEncodingTokenizer:
    max_token_id = 1024

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [] if text.startswith(" ") else [321]


def test_bad_word_skips_empty_optional_prefixed_tokenization():
    params = SamplingParams(bad_words=["word"])

    params.update_from_tokenizer(EmptyPrefixedEncodingTokenizer())

    assert params.bad_words_token_ids == [[321]]


# --- Beam search: beam width / n honor the sequence cap --------------------


def _set_max_n(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    monkeypatch.setenv("IRWKV_MAX_N_SEQUENCES", str(value))
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def test_direct_beam_width_rejects_values_over_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)

    with pytest.raises(IRWKVValidationError, match="beam_width must be at most 4"):
        BeamSearchParams(beam_width=5, max_tokens=1)


def test_chat_beam_conversion_rejects_n_before_stream_state_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=5,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    with pytest.raises(IRWKVValidationError, match="beam_width must be at most 4"):
        request.to_beam_search_params(max_tokens=1, default_sampling_params={})


def test_chat_beam_conversion_accepts_n_at_sequence_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_max_n(monkeypatch, 4)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        n=4,
        stream=True,
        use_beam_search=True,
        max_tokens=1,
    )

    params = request.to_beam_search_params(max_tokens=1, default_sampling_params={})

    assert params.beam_width == 4
