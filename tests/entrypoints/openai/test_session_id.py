# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import multiprocessing
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
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


class _ReplicaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._reply({"object": "list", "data": [{"id": "rwkv"}]})
        else:
            self._reply({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._reply(
            {
                "worker": cast(ThreadingHTTPServer, self.server).server_port,
                "session_id": self.headers.get("X-Session-ID"),
                "path": self.path,
                "body": body,
            }
        )


@pytest.mark.parametrize("policy", ["consistent_hash", "sticky_least_loaded"])
def test_router_preserves_session_across_generation_endpoints(policy):
    pytest.importorskip("vllm_router")
    from irwkv import RouterArgs, launch_router

    workers = [ThreadingHTTPServer(("127.0.0.1", 0), _ReplicaHandler) for _ in range(2)]
    for worker in workers:
        threading.Thread(target=worker.serve_forever, daemon=True).start()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    args = RouterArgs(
        worker_urls=[f"http://127.0.0.1:{worker.server_port}" for worker in workers],
        host="127.0.0.1",
        port=port,
        policy=policy,
        worker_startup_timeout_secs=20,
        worker_startup_check_interval=1,
        log_level="warn",
    )
    process = multiprocessing.get_context("spawn").Process(
        target=launch_router, args=(args,)
    )
    process.start()
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            router_url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                assert process.is_alive(), f"router exited with {process.exitcode}"
                try:
                    state = client.get(f"{router_url}/workers").json()
                    if state["total"] == 2 and all(
                        worker["is_healthy"] for worker in state["workers"]
                    ):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("router workers did not become healthy")

            requests = [
                (
                    "/inference/v1/generate",
                    {
                        "model": "rwkv",
                        "token_ids": [1, 2],
                        "cache_salt": "policy-version-42",
                        "sampling_params": {"max_tokens": 1},
                    },
                ),
                (
                    "/v1/chat/completions",
                    {"model": "rwkv", "messages": [{"role": "user", "content": "hi"}]},
                ),
                ("/v1/completions", {"model": "rwkv", "prompt": "hi"}),
            ]

            def send(path, body, session_id):
                response = client.post(
                    router_url + path,
                    json=body,
                    headers={"X-Session-ID": session_id},
                )
                response.raise_for_status()
                result = response.json()
                assert result["session_id"] == session_id
                assert result["path"] == path
                assert all(result["body"][key] == value for key, value in body.items())
                return result["worker"]

            first = send(*requests[0], "session-a")
            for request in requests[1:]:
                assert send(*request, "session-a") == first

            second = send(*requests[0], "session-b")
            if policy == "sticky_least_loaded":
                assert second != first
            response = client.post(
                f"{router_url}/finish_session", params={"session_id": "session-a"}
            )
            assert response.json() == {"status": "ok", "session_id": "session-a"}
            if policy == "sticky_least_loaded":
                assert send(*requests[0], "session-c") == first
            else:
                assert send(*requests[0], "session-a") == first
    finally:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        for worker in workers:
            worker.shutdown()
            worker.server_close()
    assert not process.is_alive()
