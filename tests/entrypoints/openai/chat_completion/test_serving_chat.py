# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from irwkv.config import MultiModalConfig
from irwkv.entrypoints.generate.base.protocol import (
    DeltaMessage,
    FunctionCall,
    RequestResponseMetadata,
)
from irwkv.entrypoints.generate.base.serving import build_per_request_timing_metrics
from irwkv.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from irwkv.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat,
    _get_mm_token_counts,
    _make_completion_tokens_details,
    _make_prompt_tokens_details,
)
from irwkv.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from irwkv.entrypoints.serve.engine.protocol import ErrorResponse
from irwkv.exceptions import IRWKVValidationError, QueueOverflowError
from irwkv.logprobs import Logprob
from irwkv.multimodal.inputs import PlaceholderRange
from irwkv.outputs import CompletionOutput, RequestOutput
from irwkv.renderers.hf import HfRenderer
from irwkv.renderers.online_renderer import OnlineRenderer
from irwkv.tokenizers import get_tokenizer
from irwkv.tokenizers.registry import cached_tokenizer_from_config
from irwkv.v1.engine.async_llm import AsyncLLM
from irwkv.v1.metrics.stats import RequestStateStats

_PER_REQUEST_STATS = RequestStateStats(
    queued_ts=1.0,
    scheduled_ts=1.5,
    first_token_ts=2.0,
    last_token_ts=3.0,
    num_generation_tokens=2,
)


def test_rwkv_sampling_profile_selection_prefers_tools():
    serving = object.__new__(OpenAIServingChat)
    serving.default_sampling_params = {"profile": "open_think"}
    serving._rwkv_sampling_profiles = {
        "open_think": serving.default_sampling_params,
        "fake_think": {"profile": "fake_think"},
        "tools": {"profile": "tools"},
    }
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert serving._sampling_defaults_for_request(request)["profile"] == "open_think"
    assert (
        serving._sampling_defaults_for_request(
            request, {"rwkv_generation_prompt": "fake_think"}
        )["profile"]
        == "fake_think"
    )

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert (
        serving._sampling_defaults_for_request(
            request, {"rwkv_generation_prompt": "fake_think"}
        )["profile"]
        == "tools"
    )


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
MODEL_NAME_SHORT = "rwkv"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
    BaseModelPath(name=MODEL_NAME_SHORT, model_path=MODEL_NAME_SHORT),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    revision = None
    code_revision = None
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}

    def get_vocab_size(self) -> int:
        return 65536


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockIrwkvConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _build_renderer(model_config: MockModelConfig):
    return HfRenderer(
        MockIrwkvConfig(model_config, parallel_config=MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


def _build_online_renderer(
    engine, model_registry: OpenAIModelRegistry
) -> OnlineRenderer:
    return OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
    )


def _build_mock_engine() -> MagicMock:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    return mock_engine


def _build_serving_chat(
    engine: AsyncLLM,
    *,
    reasoning_parser: str = "",
    tool_parser: str | None = None,
    enable_auto_tools: bool = False,
) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = _build_online_renderer(engine, models.registry)

    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
        enable_auto_tools=enable_auto_tools,
    )

    return serving_chat


def _build_minimal_metrics_serving_chat(
    enable_per_request_metrics: bool,
    enable_force_include_usage: bool = False,
) -> OpenAIServingChat:
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.response_role = "assistant"
    serving.parser_cls = None
    serving.enable_auto_tools = False
    serving._include_reasoning_tokens_details = False
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.enable_force_include_usage = enable_force_include_usage
    serving.request_logger = None
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = enable_per_request_metrics
    return serving


def _make_metrics_request_output(
    metrics: RequestStateStats | None = _PER_REQUEST_STATS,
    token_ids: tuple[int, ...] = (100, 101),
) -> RequestOutput:
    return RequestOutput(
        request_id="test-id",
        prompt="Test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="Hello",
                token_ids=list(token_ids),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        metrics=metrics,
    )


async def _single_request_output(
    request_output: RequestOutput,
) -> AsyncIterator[RequestOutput]:
    yield request_output


async def _stream_request_outputs(
    *request_outputs: RequestOutput,
) -> AsyncIterator[RequestOutput]:
    for request_output in request_outputs:
        yield request_output


async def _collect_metrics_stream_chunks(
    serving: OpenAIServingChat,
    request: ChatCompletionRequest,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    async for line in serving.chat_completion_stream_generator(
        request,
        _single_request_output(_make_metrics_request_output()),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
    ):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload != "[DONE]":
            chunks.append(json.loads(payload))
    return chunks


def test_build_per_request_timing_metrics_valid_timestamps():
    metrics = build_per_request_timing_metrics(
        _PER_REQUEST_STATS, num_generation_tokens=10
    )

    assert metrics.time_to_first_token_ms == pytest.approx(500.0)
    assert metrics.generation_time_ms == pytest.approx(1000.0)
    assert metrics.queue_time_ms == pytest.approx(500.0)
    assert metrics.mean_itl_ms == pytest.approx(1000.0 / 9, rel=1e-4)
    assert metrics.tokens_per_second == pytest.approx(10.0 / 1.5, rel=1e-4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_finish_reason", "expected"), [("stop", "tool_calls"), ("length", "length")]
)
@pytest.mark.parametrize(
    "tool_choice",
    ["auto", "required", {"type": "function", "function": {"name": "weather"}}],
)
async def test_chat_tool_finish_reason_preserves_length(
    engine_finish_reason, expected, tool_choice
):
    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=False)
    serving.enable_auto_tools = True
    serving.parser_cls = MagicMock(tool_parser_cls=object())
    parser = MagicMock()
    parser.parse.return_value = (
        None,
        None,
        [FunctionCall(name="weather", arguments="{}", id="call_test")],
    )
    parser.count_reasoning_tokens.return_value = 0
    output = _make_metrics_request_output()
    output.outputs[0].finish_reason = engine_finish_reason
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[{"type": "function", "function": {"name": "weather"}}],
        tool_choice=tool_choice,
    )
    response = await serving.chat_completion_full_generator(
        request,
        _single_request_output(output),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Weather?"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
        parser=parser,
    )
    assert response.choices[0].message.tool_calls[0].function.name == "weather"
    assert response.choices[0].finish_reason == expected


@pytest.mark.asyncio
async def test_chat_per_request_metrics_follow_server_flag():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=False,
    )
    request_output = _make_metrics_request_output()

    disabled_serving = _build_minimal_metrics_serving_chat(
        enable_per_request_metrics=False
    )
    disabled_response = await disabled_serving.chat_completion_full_generator(
        request,
        _single_request_output(request_output),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
    )
    assert disabled_response.metrics is None
    details = disabled_response.usage.completion_tokens_details
    assert details is None or details.reasoning_tokens == 0

    enabled_serving = _build_minimal_metrics_serving_chat(
        enable_per_request_metrics=True
    )
    enabled_response = await enabled_serving.chat_completion_full_generator(
        request,
        _single_request_output(request_output),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
    )
    assert enabled_response.metrics is not None
    assert enabled_response.metrics.time_to_first_token_ms == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_chat_per_request_metrics_suppressed_for_n_greater_than_one():
    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=True)
    response = await serving.chat_completion_full_generator(
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=10,
            stream=False,
            n=2,
        ),
        _single_request_output(_make_metrics_request_output()),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
    )
    assert response.metrics is None


@pytest.mark.asyncio
async def test_chat_full_generator_gives_each_choice_parser_the_prompt():
    prompts: list[list[int] | None] = []

    class RecordingParser:
        tool_parser_cls = None

        def __init__(self, *args, **kwargs):
            self.prompt_token_ids = None

        def set_prompt_token_ids(self, prompt_token_ids):
            self.prompt_token_ids = list(prompt_token_ids)

        def parse(self, model_output, request, **kwargs):
            prompts.append(self.prompt_token_ids)
            return None, model_output, None

        def count_reasoning_tokens(self, token_ids):
            return 0

    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=False)
    serving.parser_cls = RecordingParser
    serving.model_config = None
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {}
    request_output = _make_metrics_request_output()
    request_output.prompt_token_ids = [7, 8]
    request_output.outputs.append(
        CompletionOutput(
            index=1,
            text="Hi",
            token_ids=[102],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        )
    )

    await serving.chat_completion_full_generator(
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=10,
            n=2,
        ),
        _single_request_output(request_output),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
        parser=RecordingParser(),
    )

    assert prompts == [[7, 8], [7, 8]]


@pytest.mark.asyncio
async def test_chat_streaming_metrics_ride_on_usage_chunk():
    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=True)
    chunks = await _collect_metrics_stream_chunks(
        serving,
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True},
        ),
    )

    usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
    assert usage_chunks
    assert usage_chunks[-1]["metrics"]["time_to_first_token_ms"] == pytest.approx(500.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("include_reasoning", [True, False])
@pytest.mark.parametrize("include_details", [True, False])
@pytest.mark.parametrize("separate_finish", [True, False])
@pytest.mark.parametrize(
    "stream_options, force_usage, include_usage, continuous_usage",
    [
        (None, False, False, False),
        ({"include_usage": True}, False, True, False),
        ({"include_usage": True, "continuous_usage_stats": True}, False, True, True),
        ({"include_usage": False, "continuous_usage_stats": True}, False, False, False),
        (None, True, True, True),
    ],
)
async def test_streaming_reasoning_usage_counts_across_deltas(
    n,
    include_reasoning,
    include_details,
    separate_finish,
    stream_options,
    force_usage,
    include_usage,
    continuous_usage,
):
    """Final usage needs one recount per choice, not per streamed delta."""
    serving = _build_minimal_metrics_serving_chat(
        enable_per_request_metrics=False, enable_force_include_usage=force_usage
    )
    serving._include_reasoning_tokens_details = include_details
    serving.model_config = None

    parsers = [MagicMock() for _ in range(n)]
    request_outputs = []
    num_steps = 3 if separate_finish else 2
    for parser in parsers:
        parser.parse_delta.side_effect = [
            DeltaMessage(reasoning="reasoning") if include_reasoning else None,
            DeltaMessage(content="answer"),
            DeltaMessage(),
        ]
        parser.count_reasoning_tokens.side_effect = lambda token_ids: sum(
            token_id == 20 for token_id in token_ids
        )
    serving.parser_cls = MagicMock(side_effect=parsers)
    for step in range(num_steps):
        for i in reversed(range(n)):
            tokens = ((10,) + (20,) * (i + 1), (11, 20, 30), ())[step]
            result = _make_metrics_request_output(metrics=None, token_ids=tokens)
            result.outputs[0].index = i
            result.outputs[0].text = ("<think>reasoning", "</think>answer", "")[step]
            finished = step == num_steps - 1
            result.outputs[0].finish_reason = "stop" if finished else None
            result.finished = finished and i == 0
            request_outputs.append(result)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        n=n,
        include_reasoning=include_reasoning,
        return_token_ids=True,
        stream_options=stream_options,
    )
    metadata = RequestResponseMetadata(request_id="chatcmpl-test-id")
    chunks: list[dict[str, Any]] = []
    async for line in serving.chat_completion_stream_generator(
        request,
        _stream_request_outputs(*request_outputs),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=metadata,
    ):
        payload = line.removeprefix("data: ").strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))

    assert all("error" not in chunk for chunk in chunks)
    expected_count = sum(i + 2 for i in range(n))
    expected_details = {"reasoning_tokens": expected_count} if include_details else None
    assert metadata.final_usage_info is not None
    assert metadata.final_usage_info.model_dump()["completion_tokens_details"] == (
        expected_details
    )
    final_chunks = [chunk for chunk in chunks if not chunk["choices"]]
    assert len(final_chunks) == int(include_usage)
    if include_usage:
        assert final_chunks[0]["usage"].get("completion_tokens_details") == (
            expected_details
        )
    for i, parser in enumerate(parsers):
        expected_calls = (
            (num_steps if continuous_usage else 1) if include_details else 0
        )
        assert parser.count_reasoning_tokens.call_count == expected_calls
        if include_details:
            assert parser.count_reasoning_tokens.call_args.args == (
                (10,) + (20,) * (i + 1) + (11, 20, 30),
            )
        choices = [
            chunk
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["index"] == i
        ]
        assert (
            "".join(c["choices"][0]["delta"].get("content") or "" for c in choices)
            == "answer"
        )
        if not include_reasoning:
            assert all(c["choices"][0].get("token_ids") is None for c in choices)
        if continuous_usage and include_details:
            counts = [
                c["usage"]["completion_tokens_details"]["reasoning_tokens"]
                for c in choices
            ]
            expected_counts = [0, i + 1] if include_reasoning else [0]
            assert counts == expected_counts + [i + 2] * (num_steps - 1)


def _reasoning_lifecycle_output(
    index: int = 0,
    tokens: tuple[int, ...] = (20, 30),
    text: str = "answer",
    finish: str | None = None,
) -> RequestOutput:
    result = _make_metrics_request_output(None, tokens)
    result.outputs[0].index = index
    result.outputs[0].text = text
    result.outputs[0].finish_reason = finish
    result.finished = finish is not None
    return result


_EMPTY_OUTPUTS = _reasoning_lifecycle_output(tokens=(), text="")
_EMPTY_OUTPUTS.outputs = []

_REASONING_LIFECYCLE_CASES = [
    ("empty-generator", 1, [], 0, 0),
    ("empty-outputs", 1, [_EMPTY_OUTPUTS], 0, 0),
    (
        "empty-finish",
        1,
        [_reasoning_lifecycle_output(tokens=(), text="", finish="stop")],
        0,
        0,
    ),
    (
        "empty-prefill",
        1,
        [
            _reasoning_lifecycle_output(tokens=(), text=""),
            _reasoning_lifecycle_output(finish="stop"),
        ],
        2,
        1,
    ),
    (
        "one-empty-choice",
        2,
        [_reasoning_lifecycle_output(index=1, finish="stop")],
        2,
        1,
    ),
    (
        "uneven-finish",
        2,
        [
            _reasoning_lifecycle_output(index=1, finish="stop"),
            _reasoning_lifecycle_output(),
            _reasoning_lifecycle_output(tokens=(20,), finish="stop"),
        ],
        5,
        3,
    ),
    (
        "empty-final-delta",
        1,
        [
            _reasoning_lifecycle_output(),
            _reasoning_lifecycle_output(tokens=(), text="", finish="stop"),
        ],
        2,
        1,
    ),
    (
        "ignore-finished-choice",
        2,
        [
            _reasoning_lifecycle_output(index=1, finish="stop"),
            _reasoning_lifecycle_output(index=1, tokens=(20, 20, 20)),
            _reasoning_lifecycle_output(finish="stop"),
        ],
        4,
        2,
    ),
    ("exception-before", 1, [RuntimeError("fixture engine error")], None, None),
    (
        "exception-after",
        1,
        [_reasoning_lifecycle_output(), RuntimeError("fixture engine error")],
        None,
        None,
    ),
    (
        "generation-error",
        1,
        [
            _reasoning_lifecycle_output(),
            _reasoning_lifecycle_output(finish="error"),
        ],
        None,
        None,
    ),
    (
        "cancel-after",
        1,
        [_reasoning_lifecycle_output(), asyncio.CancelledError()],
        None,
        None,
    ),
    (
        "close-after",
        1,
        [
            _reasoning_lifecycle_output(),
            _reasoning_lifecycle_output(finish="stop"),
        ],
        None,
        None,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_mode", ["none", "final", "continuous"])
@pytest.mark.parametrize(
    "name, n, outputs, expected_tokens, expected_reasoning",
    _REASONING_LIFECYCLE_CASES,
    ids=[case[0] for case in _REASONING_LIFECYCLE_CASES],
)
async def test_streaming_reasoning_usage_lifecycle_edges(
    usage_mode,
    name,
    n,
    outputs,
    expected_tokens,
    expected_reasoning,
):
    """Cancel, errors, and empty choices still match baseline stream usage."""
    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=False)
    serving.model_config = None
    serving._include_reasoning_tokens_details = True
    parsers = [MagicMock() for _ in range(n)]
    for parser in parsers:
        parser.parse_delta.side_effect = lambda **kwargs: DeltaMessage(
            content=kwargs["delta_text"]
        )
        parser.count_reasoning_tokens.side_effect = lambda ids: ids.count(20)
    serving.parser_cls = MagicMock(side_effect=parsers)
    stream_options = (
        None
        if usage_mode == "none"
        else {
            "include_usage": True,
            "continuous_usage_stats": usage_mode == "continuous",
        }
    )
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=True,
        n=n,
        return_token_ids=True,
        stream_options=stream_options,
    )
    metadata = RequestResponseMetadata(request_id="chatcmpl-lifecycle")

    async def source():
        for result in outputs:
            if isinstance(result, BaseException):
                raise result
            yield result

    iterator = source()
    stream = serving.chat_completion_stream_generator(
        request,
        iterator,
        "chatcmpl-lifecycle",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=metadata,
    )
    chunks: list[object] = []
    outcome = "exhausted"
    try:
        async for line in stream:
            raw = line.removeprefix("data: ").strip()
            if raw == "[DONE]":
                chunks.append(raw)
                continue
            item = json.loads(raw)
            chunks.append(item)
            if name == "close-after" and any(
                choice["delta"].get("content") for choice in item.get("choices", [])
            ):
                await stream.aclose()
                outcome = "closed"
                break
    except asyncio.CancelledError:
        outcome = "cancelled"
    finally:
        await stream.aclose()
        await iterator.aclose()

    error_chunks = [c for c in chunks if isinstance(c, dict) and "error" in c]
    call_count = sum(parser.count_reasoning_tokens.call_count for parser in parsers)
    if expected_tokens is not None:
        assert outcome == "exhausted" and chunks[-1] == "[DONE]" and not error_chunks
        assert metadata.final_usage_info is not None
        assert metadata.final_usage_info.completion_tokens == expected_tokens
        assert (
            metadata.final_usage_info.completion_tokens_details.reasoning_tokens
            == expected_reasoning
        )
        nonempty_choices = sum(
            bool(parser.count_reasoning_tokens.call_args) for parser in parsers
        )
        if usage_mode == "continuous":
            assert call_count >= nonempty_choices
        else:
            assert call_count == nonempty_choices
        return

    assert metadata.final_usage_info is None
    if name in {"cancel-after", "close-after"}:
        assert outcome == ("cancelled" if name == "cancel-after" else "closed")
        assert "[DONE]" not in chunks and not error_chunks
    else:
        assert outcome == "exhausted" and chunks[-1] == "[DONE]"
        assert len(error_chunks) == 1
    if usage_mode != "continuous" or name == "exception-before":
        assert call_count == 0
    else:
        assert call_count >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_reasoning", "expected_tokens"),
    [(True, ["<tool_call>", "answer"]), (False, [])],
)
async def test_streaming_logprobs_kept_when_parser_suppresses_delta(
    include_reasoning: bool, expected_tokens: list[str]
):
    """Logprobs of tokens the parser swallows (e.g. <tool_call>) must still
    be streamed, unless per-token metadata is hidden with the reasoning."""
    serving = _build_minimal_metrics_serving_chat(enable_per_request_metrics=False)
    serving.model_config = None
    serving.return_tokens_as_token_ids = False

    parser = MagicMock()
    parser.parse_delta.side_effect = [None, DeltaMessage(content="answer")]
    parser.count_reasoning_tokens.return_value = 0
    serving.parser_cls = MagicMock(return_value=parser)

    def make_output(token_id: int, text: str, finish_reason: str | None):
        request_output = _make_metrics_request_output(
            metrics=None, token_ids=(token_id,)
        )
        completion = request_output.outputs[0]
        completion.text = text
        completion.finish_reason = finish_reason
        completion.logprobs = [
            {token_id: Logprob(logprob=-0.5, rank=1, decoded_token=text)}
        ]
        return request_output

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=10,
        stream=True,
        logprobs=True,
        include_reasoning=include_reasoning,
    )
    chunks: list[dict[str, Any]] = []
    async for line in serving.chat_completion_stream_generator(
        request,
        _stream_request_outputs(
            make_output(10, "<tool_call>", None), make_output(11, "answer", "stop")
        ),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "Test"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
    ):
        payload = line.removeprefix("data: ").strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))

    streamed_tokens = [
        entry["token"]
        for chunk in chunks
        for choice in chunk["choices"]
        if choice.get("logprobs")
        for entry in choice["logprobs"]["content"]
    ]
    assert streamed_tokens == expected_tokens


@dataclass
class MockEngine:
    model_config: MockModelConfig = field(default_factory=MockModelConfig)
    input_processor: MagicMock = field(default_factory=MagicMock)
    renderer: MagicMock = field(default_factory=MagicMock)
    errored: bool = False

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        pass


async def _async_serving_chat_init():
    engine = MockEngine()

    models = OpenAIServingModels(engine, BASE_MODEL_PATHS)
    online_renderer = _build_online_renderer(engine, models.registry)

    serving_completion = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
    )
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.chat_template == CHAT_TEMPLATE
    assert serving_completion._include_reasoning_tokens_details is False


def test_mm_prompt_tokens_details():
    # Text-only input has no multimodal placeholders.
    assert _get_mm_token_counts({"type": "tokens"}) == {}

    # Per-modality counts sum each modality's placeholder ranges.
    counts = _get_mm_token_counts(
        {
            "mm_placeholders": {
                "image": [
                    PlaceholderRange(offset=0, length=576),
                    PlaceholderRange(offset=600, length=24),
                ],
                "video": [PlaceholderRange(offset=700, length=1200)],
            }
        }
    )
    assert counts == {"image": 600, "video": 1200}

    # Gated off, or nothing to report -> no details.
    assert _make_prompt_tokens_details(False, 5, 0, counts) is None
    assert _make_prompt_tokens_details(True, None, None, None) is None

    # Zero cached_tokens is still reported (not None), matching the cached-only
    # behavior; multimodal counts ride alongside even when cached_tokens is None.
    details = _make_prompt_tokens_details(True, 0, 0, None)
    assert details.cached_tokens == 0
    assert details.created_cache_tokens == 0
    assert details.multimodal_tokens is None
    details = _make_prompt_tokens_details(True, None, None, counts)
    assert details.cached_tokens is None
    assert details.created_cache_tokens is None
    assert details.multimodal_tokens == {"image": 600, "video": 1200}
    details = _make_prompt_tokens_details(True, 3, 0, counts)
    assert details.cached_tokens == 3
    assert details.created_cache_tokens == 0
    assert details.multimodal_tokens == {"image": 600, "video": 1200}


def test_completion_tokens_details():
    assert _make_completion_tokens_details(7).reasoning_tokens == 7


@pytest.mark.asyncio
async def test_serving_chat_returns_correct_model_name():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine)
    messages = [{"role": "user", "content": "what is 1+1?"}]

    async def return_model_name(*args, **kwargs):
        return args[3]

    serving_chat.chat_completion_full_generator = return_model_name

    # Test that full name is returned when short name is requested
    req = ChatCompletionRequest(model=MODEL_NAME_SHORT, messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME

    # Test that full name is returned when empty string is specified
    req = ChatCompletionRequest(model="", messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME

    # Test that full name is returned when no model is specified
    req = ChatCompletionRequest(messages=messages)
    assert await serving_chat.create_chat_completion(req) == MODEL_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_admission_rejection_escapes_before_response_starts(stream):
    """Overload rejections must propagate out of create_chat_completion.

    For streaming this is the only chance to return a real HTTP status: once
    StreamingResponse is constructed the 200 has already been sent and the
    rejection would degrade into an in-band SSE error chunk.
    """
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    mock_engine.check_admission.side_effect = QueueOverflowError()

    serving_chat = _build_serving_chat(mock_engine)
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
        stream=stream,
    )

    with pytest.raises(QueueOverflowError):
        await serving_chat.create_chat_completion(req)

    mock_engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_serving_chat_should_set_correct_max_tokens():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )
    prompt_ids = await _render_chat_prompt_token_ids(serving_chat, req.messages)
    available_tokens = mock_engine.model_config.max_model_len - len(prompt_ids)

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == available_tokens

    req.max_tokens = 10
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Model author's generation_config.json sets max_tokens (auto, no override)
    # — should act as fallback only, not ceiling
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {"max_tokens": 10}

    # Reinitialize the engine with new settings
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    # Test Case 1: No max_tokens specified in request
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Test Case 2: Request's max_tokens set higher than generation_config
    # default so request-provided max_tokens takes precedence
    req.max_tokens = 15

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 15

    # Test Case 3: Request's max_tokens set lower than server accepts
    req.max_tokens = 5

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 5

    # User explicitly sets max_tokens via --override-generation-config
    # — should act as a ceiling
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {"max_tokens": 10}
    mock_model_config.override_generation_config = {"max_new_tokens": 10}

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine)

    # Test Case 3.1: No max_tokens — uses override as default
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Test Case 3.2: Request max_tokens higher — capped by user ceiling from override
    req.max_tokens = 15

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 10

    # Test Case 3.3: Request max_tokens lower — respected
    req.max_tokens = 5

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 5

    # Setting server's max_tokens in the generation_config.json
    # higher than context_window - prompt_tokens
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {"max_tokens": 200}

    # Reinitialize the engine with new settings
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    # Test case 1: No max_tokens specified, defaults to context_window
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == available_tokens

    # Test Case 2: Request's max_tokens set higher than server accepts
    req.max_tokens = 100

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == available_tokens

    # Test Case 3: Request's max_tokens set lower than server accepts
    req.max_tokens = 5

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].max_tokens == 5


@pytest.mark.asyncio
async def test_serving_chat_truncate_prompt_tokens_max_token_accounting():
    """When truncate_prompt_tokens is set, max_tokens must be calculated using
    the truncated prompt length, not the original prompt length.

    Regression: without the fix, get_max_tokens received the untruncated prompt
    length, causing the output budget to be underestimated.
    """
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine)

    messages = [{"role": "user", "content": "what is 1+1?"}]
    prompt_ids = await _render_chat_prompt_token_ids(serving_chat, messages)
    available_tokens = mock_engine.model_config.max_model_len - len(prompt_ids)
    assert len(prompt_ids) > 5

    req = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    assert mock_engine.generate.call_args.args[1].max_tokens == available_tokens

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=messages,
        truncate_prompt_tokens=5,
    )
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    assert mock_engine.generate.call_args.args[1].max_tokens == 95

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=messages,
        truncate_prompt_tokens=-1,
    )
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)
    assert mock_engine.generate.call_args.args[1].max_tokens == available_tokens


@pytest.mark.asyncio
async def test_serving_chat_truncation_side_controls_prompt_truncation():
    model_config = MockModelConfig()
    model_config.model = MODEL_NAME_SHORT
    model_config.tokenizer = MODEL_NAME
    mock_engine = MockEngine(
        model_config=model_config,
        renderer=_build_renderer(model_config),
    )

    serving_chat = _build_serving_chat(mock_engine)
    messages = [
        {
            "role": "user",
            "content": "Summarize how prompt truncation works in one sentence.",
        }
    ]

    full_token_ids = await _render_chat_prompt_token_ids(
        serving_chat,
        messages,
        model_name=MODEL_NAME_SHORT,
    )
    assert len(full_token_ids) > 4

    right_token_ids = await _render_chat_prompt_token_ids(
        serving_chat,
        messages,
        model_name=MODEL_NAME_SHORT,
        truncate_prompt_tokens=4,
        truncation_side="right",
    )
    assert right_token_ids == full_token_ids[:4]

    left_token_ids = await _render_chat_prompt_token_ids(
        serving_chat,
        messages,
        model_name=MODEL_NAME_SHORT,
        truncate_prompt_tokens=4,
        truncation_side="left",
    )
    assert left_token_ids == full_token_ids[-4:]


def test_serving_chat_starts_without_tokenizer():
    """Regression test: building the serving objects must not require a
    tokenizer, which is absent under `skip_tokenizer_init`."""
    mock_engine = _build_mock_engine()
    mock_engine.model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_engine.renderer = HfRenderer(
        MockIrwkvConfig(mock_engine.model_config, parallel_config=MockParallelConfig()),
        None,
    )

    serving_chat = _build_serving_chat(mock_engine)

    assert serving_chat.parser_cls is None


@pytest.mark.asyncio
async def test_serving_chat_could_load_correct_generation_config():
    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "temperature": 0.5,
        "repetition_penalty": 1.05,
    }

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    # Initialize the serving chat
    serving_chat = _build_serving_chat(mock_engine)

    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.5
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test the param when user set it
    req.temperature = 0.1

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.1
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test When temperature==0.0
    req.temperature = 0.0

    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert mock_engine.generate.call_args.args[1].temperature == 0.0
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05


@pytest.mark.parametrize("model_type", ["any"])
@pytest.mark.asyncio
async def test_serving_chat_did_set_correct_cache_salt(model_type):
    mock_model_config = MockModelConfig()
    mock_model_config.hf_config.model_type = model_type

    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = mock_model_config
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    serving_chat = _build_serving_chat(mock_engine)

    orig_render_chat_request = serving_chat.render_chat_request
    captured_inputs = []

    async def render_chat_request(request):
        result = await orig_render_chat_request(request)

        assert isinstance(result, tuple)
        conversation, engine_inputs = result
        captured_inputs.extend(engine_inputs)

        return result

    serving_chat.render_chat_request = render_chat_request

    # Test cache_salt
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    # By default, cache_salt in the engine prompt is not set
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert len(captured_inputs) == 1
    assert "cache_salt" not in captured_inputs[0]

    captured_inputs.clear()

    # Test with certain cache_salt
    req.cache_salt = "test_salt"
    with suppress(Exception):
        await serving_chat.create_chat_completion(req)

    assert len(captured_inputs) == 1
    assert captured_inputs[0]["cache_salt"] == "test_salt"


@pytest.mark.asyncio
async def test_serving_chat_data_parallel_rank_extraction():
    """Test that data_parallel_rank is properly extracted from header and
    passed to engine."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    # Mock the generate method to return an async generator
    async def mock_generate(*args, **kwargs):
        # Yield a fake RequestOutput
        from irwkv.outputs import CompletionOutput, RequestOutput

        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="test response",
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = AsyncMock(side_effect=mock_generate)

    serving_chat = _build_serving_chat(mock_engine)

    # Test when data_parallel_rank is present in header
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 1+1?"}],
    )

    # Mock request with X-data-parallel-rank header
    mock_raw_request = MagicMock()
    mock_raw_request.headers = {"X-data-parallel-rank": "2"}
    mock_raw_request.state = MagicMock()

    with suppress(Exception):
        await serving_chat.create_chat_completion(req, mock_raw_request)

    # Verify that data_parallel_rank was passed to engine.generate
    assert "data_parallel_rank" in mock_engine.generate.call_args.kwargs
    assert mock_engine.generate.call_args.kwargs["data_parallel_rank"] == 2

    # Test when data_parallel_rank is not present (defaults to None)
    req_no_dp = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "what is 2+2?"}],
    )

    # Mock request with no header
    mock_raw_request_no_dp = MagicMock()
    mock_raw_request_no_dp.headers = {}
    mock_raw_request_no_dp.state = MagicMock()

    with suppress(Exception):
        await serving_chat.create_chat_completion(req_no_dp, mock_raw_request_no_dp)

    # Verify that data_parallel_rank defaults to None
    assert "data_parallel_rank" in mock_engine.generate.call_args.kwargs
    assert mock_engine.generate.call_args.kwargs["data_parallel_rank"] is None


async def _render_chat_prompt_token_ids(
    serving_chat: OpenAIServingChat,
    messages: list[dict[str, str]],
    *,
    model_name: str = MODEL_NAME,
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
) -> list[int]:
    request = ChatCompletionRequest(
        model=model_name,
        messages=messages,
        max_tokens=1,
        temperature=0.0,
        return_token_ids=True,
        truncate_prompt_tokens=truncate_prompt_tokens,
        truncation_side=truncation_side,
    )

    result = await serving_chat.render_chat_request(request)
    assert not isinstance(result, ErrorResponse)

    _, engine_inputs = result
    assert len(engine_inputs) == 1

    prompt_token_ids = serving_chat._extract_prompt_components(
        engine_inputs[0]
    ).token_ids
    assert prompt_token_ids is not None
    return prompt_token_ids


@pytest.mark.asyncio
async def test_tool_choice_validation_without_parser():
    """Test that tool_choice='required' or named tool without tool_parser
    returns an appropriate error message."""
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = _build_online_renderer(mock_engine, models.registry)

    # Create serving_chat without tool_parser (enable_auto_tools=False)
    serving_chat = OpenAIServingChat(
        mock_engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
        enable_auto_tools=False,  # No tool parser
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    # Test tool_choice="required" without tool_parser
    req_required = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        tool_choice="required",
    )
    response_required = await serving_chat.create_chat_completion(req_required)
    assert isinstance(response_required, ErrorResponse)
    assert "tool_choice" in response_required.error.message
    assert "--tool-call-parser" in response_required.error.message

    # Test named tool_choice without tool_parser
    req_named = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    response_named = await serving_chat.create_chat_completion(req_named)
    assert isinstance(response_named, ErrorResponse)
    assert "tool_choice" in response_named.error.message
    assert "--tool-call-parser" in response_named.error.message
    # The function name should appear in a clean, readable form -
    # guards against leaking Pydantic's internal repr of the
    # ChatCompletionNamedToolChoiceParam/ChatCompletionNamedFunction
    # objects directly into the client-facing error message.
    assert "get_weather" in response_named.error.message
    assert "ChatCompletionNamedFunction" not in response_named.error.message
    assert "ChatCompletionNamedToolChoiceParam" not in response_named.error.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_finish_reason", "expected_finish_reason"),
    [("stop", "tool_calls"), ("length", "length")],
)
async def test_streaming_n_gt1_independent_tool_parsers(
    engine_finish_reason: str, expected_finish_reason: str
):
    """n>1 streaming must use independent parser instances
    and token-id histories per choice.
    """
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = _build_online_renderer(mock_engine, models.registry)

    serving_chat = OpenAIServingChat(
        mock_engine,
        models,
        response_role="assistant",
        online_renderer=online_renderer,
        chat_template=CHAT_TEMPLATE,
        chat_template_content_format="auto",
        request_logger=None,
        enable_auto_tools=True,
        tool_parser="rwkv",
    )

    tokenizer = get_tokenizer(MODEL_NAME)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    num_choices = 2

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "test"}],
        n=num_choices,
        stream=True,
        tools=tools,
        tool_choice="auto",
    )

    tool_call_text = (
        "**Tool Call:**\n```json\n"
        '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
        "```"
    )
    all_token_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)

    # Compute proper delta text for each token so that concatenated deltas
    # reproduce the original string exactly.
    steps: list[tuple[str, int]] = []
    prev_decoded = ""
    for i, tid in enumerate(all_token_ids):
        decoded_so_far = tokenizer.decode(all_token_ids[: i + 1])
        delta = decoded_so_far[len(prev_decoded) :]
        steps.append((delta, tid))
        prev_decoded = decoded_so_far

    async def result_generator():
        for delta_text, token_id in steps:
            yield RequestOutput(
                request_id="test-req",
                prompt="test",
                prompt_token_ids=[1, 2, 3],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=choice_idx,
                        text=delta_text,
                        token_ids=[token_id],
                        cumulative_logprob=0.0,
                        logprobs=None,
                    )
                    for choice_idx in range(num_choices)
                ],
                finished=False,
            )
        # Final output with finish_reason
        yield RequestOutput(
            request_id="test-req",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=choice_idx,
                    text="",
                    token_ids=[],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=engine_finish_reason,
                )
                for choice_idx in range(num_choices)
            ],
            finished=True,
        )

    # Collect tool-call deltas and finish_reasons per choice from the SSE
    # stream.
    tc_deltas_by_choice: dict[int, list[dict]] = {i: [] for i in range(num_choices)}
    finish_reasons_by_choice: dict[int, list[str]] = {i: [] for i in range(num_choices)}
    async for chunk_str in serving_chat.chat_completion_stream_generator(
        request=request,
        result_generator=result_generator(),
        request_id="test-req",
        model_name=MODEL_NAME,
        conversation=[],
        tokenizer=tokenizer,
        request_metadata=RequestResponseMetadata(
            request_id="test-req",
            model_name=MODEL_NAME,
        ),
    ):
        if not chunk_str.strip() or "data: [DONE]" in chunk_str:
            continue
        if chunk_str.startswith("data: "):
            data = json.loads(chunk_str[6:].strip())
            for choice in data.get("choices", []):
                idx = choice["index"]
                delta = choice.get("delta", {})
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        tc_deltas_by_choice[idx].append(tc)
                if choice.get("finish_reason") is not None:
                    finish_reasons_by_choice[idx].append(choice["finish_reason"])

    # Both choices must independently produce the correct tool call.
    for choice_idx in range(num_choices):
        deltas = tc_deltas_by_choice[choice_idx]
        assert len(deltas) > 0, (
            f"Choice {choice_idx}: expected tool-call deltas but got none"
        )

        name = None
        args_buf = ""
        for tc in deltas:
            fn = tc.get("function", {})
            if fn.get("name"):
                name = fn["name"]
            if fn.get("arguments"):
                args_buf += fn["arguments"]

        assert name == "get_weather", (
            f"Choice {choice_idx}: expected 'get_weather', got {name!r}"
        )
        parsed_args = json.loads(args_buf)
        assert parsed_args == {"city": "Tokyo"}, (
            f"Choice {choice_idx}: expected {{'city': 'Tokyo'}}, got {parsed_args}"
        )

        reasons = finish_reasons_by_choice[choice_idx]
        assert len(reasons) == 1, (
            f"Choice {choice_idx}: expected exactly 1 finish_reason, got {reasons}"
        )
        assert reasons[0] == expected_finish_reason, (
            f"Choice {choice_idx}: expected finish_reason={expected_finish_reason!r}, "
            f"got '{reasons[0]}'"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], [-1], [1.5], [1.0], ["1"], [True], "abc", 5])
async def test_chat_kv_transfer_prompt_token_ids_rejects_invalid_ids(ids):
    """``kv_transfer_params`` is untyped, so the ids are checked on reuse."""
    serving_chat = _build_serving_chat(_build_mock_engine())

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        kv_transfer_params={"prompt_token_ids": ids},
    )
    with pytest.raises(IRWKVValidationError, match="non-negative integers"):
        await serving_chat.render_chat_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ids", "stream", "match"),
    [
        ([1.5], False, "non-negative integers"),
        ([999999], True, r"out of vocabulary.*kv_transfer_params\.prompt_token_ids"),
    ],
)
async def test_chat_kv_transfer_prompt_token_ids_rejection_notifies_kv_connector(
    ids, stream, match
):
    """A render-time 400 on a decode request notifies the KV connector."""
    mock_engine = _build_mock_engine()
    mock_engine.notify_kv_transfer_request_rejected = AsyncMock()
    serving_chat = _build_serving_chat(mock_engine)
    serving_chat.has_kv_connector = True

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        kv_transfer_params={"prompt_token_ids": ids, "do_remote_prefill": True},
    )
    with pytest.raises(IRWKVValidationError, match=match):
        await serving_chat.create_chat_completion(request)

    mock_engine.notify_kv_transfer_request_rejected.assert_awaited_once_with(
        request.request_id, {"do_remote_prefill": True}, data_parallel_rank=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[10, 20, 30], [1.5]])
async def test_chat_kv_transfer_prompt_token_ids_ignored_with_echo(ids):
    """``echo`` needs the rendered conversation, so the ids are not used."""
    serving_chat = _build_serving_chat(_build_mock_engine())

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        kv_transfer_params={"prompt_token_ids": ids, "do_remote_prefill": True},
        echo=True,
    )
    result = await serving_chat.render_chat_request(request)
    assert not isinstance(result, ErrorResponse)

    conversation, engine_inputs = result
    assert [msg["role"] for msg in conversation] == ["user"]
    assert engine_inputs[0]["prompt_token_ids"] != ids
    assert request.kv_transfer_params == {"do_remote_prefill": True}


@pytest.mark.parametrize(
    "content",
    [
        [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ],
        # A part written without a "type" is identified by its media key.
        [{"image_url": "https://example.com/a.png"}],
        # A media key counts even when the part claims to be text, which is
        # how chat_utils reads a part carrying a "uuid".
        [
            {
                "type": "text",
                "text": "look",
                "uuid": "u1",
                "image_url": "https://example.com/a.png",
            }
        ],
        [{"type": "hologram", "hologram": "https://a/b.holo"}],
        [{"type": ["image_url"]}],
    ],
)
def test_chat_kv_transfer_prompt_token_ids_ignored_with_non_text_content(content):
    """The ids are dropped so that ``messages`` is rendered with its media."""
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        kv_transfer_params={
            "prompt_token_ids": [10, 20, 30],
            "do_remote_prefill": True,
        },
    )
    assert request.kv_transfer_params == {"do_remote_prefill": True}


def test_chat_kv_transfer_prompt_token_ids_allows_text_only_parts():
    """Text-bearing part types are not multimodal, so the ids are kept."""
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "refusal", "refusal": "no"},
                ],
            },
            {"role": "user", "content": "plain string"},
        ],
        kv_transfer_params={"prompt_token_ids": [10, 20, 30]},
    )
    assert request.kv_transfer_params == {"prompt_token_ids": [10, 20, 30]}
