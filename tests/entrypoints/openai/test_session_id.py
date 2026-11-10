# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from irwkv.entrypoints.generate.base.serving import GenerateBaseServing
from irwkv.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from irwkv.entrypoints.openai.completion.protocol import CompletionRequest
from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest


def _raw_request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (key.lower().encode("latin-1"), value.encode("latin-1"))
                for key, value in headers.items()
            ],
        }
    )


@pytest.mark.parametrize(
    "openai_request",
    [
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ),
        CompletionRequest(model="test-model", prompt="hi"),
        ResponsesRequest(model="test-model", input="hi"),
    ],
)
def test_get_session_id_accepts_body_field(openai_request):
    openai_request.session_id = "body-session"

    session_id = GenerateBaseServing._get_session_id(
        openai_request,
        _raw_request({"X-Session-ID": "header-session"}),
    )

    assert session_id == "body-session"


def test_get_session_id_accepts_session_header():
    request = CompletionRequest(model="test-model", prompt="hi")

    session_id = GenerateBaseServing._get_session_id(
        request,
        _raw_request({"X-Session-ID": "header-session"}),
    )

    assert session_id == "header-session"


def test_get_session_id_ignores_correlation_header():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        irwkv_xargs={"session_id": "xargs-session"},
    )

    session_id = GenerateBaseServing._get_session_id(
        request,
        _raw_request({"X-Correlation-ID": "correlation-session"}),
    )

    assert session_id == "xargs-session"


def test_get_session_id_keeps_irwkv_xargs_as_compatibility_fallback():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        irwkv_xargs={"session_id": "xargs-session"},
    )

    session_id = GenerateBaseServing._get_session_id(request, None)

    assert session_id == "xargs-session"


def test_get_session_id_ignores_empty_and_non_string_values():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        session_id="",
        irwkv_xargs={"session_id": 7},
    )

    session_id = GenerateBaseServing._get_session_id(request, None)

    assert session_id is None


@pytest.mark.asyncio
async def test_lora_registry_reload_preserves_unique_versions():
    from irwkv.entrypoints.openai.models.protocol import BaseModelPath
    from irwkv.entrypoints.openai.models.serving import OpenAIServingModels
    from irwkv.entrypoints.serve.lora.protocol import (
        LoadLoRAAdapterRequest,
        UnloadLoRAAdapterRequest,
    )

    engine = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        renderer=None,
        input_processor=None,
        collective_rpc=AsyncMock(return_value=[True]),
    )
    models = OpenAIServingModels(engine, [BaseModelPath("base", "/base")])
    await models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="tenant", lora_path="/a")
    )
    first = models.get_lora("tenant")
    await models.load_lora_adapter(
        LoadLoRAAdapterRequest(lora_name="tenant", lora_path="/a", load_inplace=True)
    )
    second = models.get_lora("tenant")
    assert first is not None and second is not None
    assert first.lora_int_id != second.lora_int_id
    assert {card.id for card in (await models.show_available_models()).data} == {
        "base",
        "tenant",
    }
    await models.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name="tenant"))
    assert models.get_lora("tenant") is None
