# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from irwkv.entrypoints.generate.base.serving import GenerateBaseServing
from irwkv.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from irwkv.entrypoints.openai.completion.protocol import CompletionRequest
from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest


def test_rwkv_upload_and_session_management_use_the_engine_cache():
    import torch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from irwkv.entrypoints.openai.rwkv import attach_router
    from irwkv.v1.worker.gpu.model_states.rwkv_cache import RwkvRuntimeCache

    state_layer = SimpleNamespace(state_dtype=torch.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=1, num_attention_heads=1, head_size=64, hidden_size=64
        ),
        model=SimpleNamespace(start_layer=0, end_layer=1),
        get_rwkv_state_layer=lambda: state_layer,
    )
    cache = RwkvRuntimeCache(model)

    async def collective_rpc(method, args):
        assert method == "rwkv_cache"
        return [cache.manage(*args)]

    app = FastAPI()
    app.state.engine_client = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="rwkv")),
        collective_rpc=collective_rpc,
    )
    attach_router(app)
    client = TestClient(app)
    payload = io.BytesIO()
    torch.save({"blocks.0.att.time_state": torch.ones(1, 64, 64)}, payload)
    response = client.post(
        "/v1/state/upload", files={"file": ("state.pth", payload.getvalue())}
    )
    assert response.status_code == 200
    state_id = response.json()["state_id"]
    assert client.get("/v1/state/list").json()["data"] == [{"state_id": state_id}]
    assert client.get("/state/status").json()["session_cache_hits"] == 0
    assert client.post("/v1/state/delete", json={"state_id": state_id}).json()[
        "deleted"
    ]
    assert client.get("/v1/state/list").json()["data"] == []
    assert (
        client.post("/v1/adapters", files={"file": ("adapter.pth", b"bad")}).status_code
        == 400
    )

    async def failed_collective_rpc(method, args):
        return [{"error": "adapter rank does not match the server"}]

    app.state.engine_client.collective_rpc = failed_collective_rpc
    response = client.post(
        "/v1/adapters",
        data={"adapter_id": "bad-rank"},
        files={"file": ("adapter.pth", b"checkpoint")},
    )
    assert response.status_code == 400
    assert "rank" in response.json()["detail"]


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
