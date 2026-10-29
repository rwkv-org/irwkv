# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from examples.basic.online_serving import (
    openai_chat_completion_client_with_compaction as compaction,
)

ChatContext = compaction.ChatContext
CompactionError = compaction.CompactionError


@pytest.mark.asyncio
async def test_cli_rejects_overwriting_original_history(tmp_path):
    source = tmp_path / "history.json"
    source.write_text("[]")
    alias = tmp_path / "alias.json"
    alias.hardlink_to(source)
    with pytest.raises(CompactionError, match="different file"):
        await compaction.main(
            SimpleNamespace(messages=source, output=alias, tools=None)
        )
    assert source.read_text() == "[]"


def history():
    return [
        {"role": "system", "content": "Always retain exact numbers."},
        {"role": "developer", "content": "Do not change tool IDs."},
        {"role": "user", "content": "Old evidence " * 80},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Check release 122."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "1937 examples"},
        {"role": "tool", "tool_call_id": "a", "content": "122 steps"},
        {"role": "assistant", "content": "Both results received."},
        {"role": "user", "content": "Continue with those constraints."},
    ]


class Server:
    def __init__(self, summary="Earlier evidence retained.", finish="stop"):
        self.summary = summary
        self.finish = finish
        self.requests = []
        self.generation_count = 0

    async def handle(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/tokenize":
            # A deterministic test service, with framing and schema overhead.
            count = len(json.dumps(body["messages"], ensure_ascii=False)) + 50
            if body.get("tools"):
                count += 200
            return httpx.Response(200, json={"count": count, "max_model_len": 10000})
        self.generation_count += 1
        await asyncio.sleep(0)
        text = self.summary(body) if callable(self.summary) else self.summary
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": self.finish,
                        "message": {"role": "assistant", "content": text},
                    }
                ]
            },
        )


@pytest.mark.asyncio
async def test_under_budget_does_not_summarize_or_alias_history():
    server = Server()
    original = history()
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        result = await ChatContext(client, "model").compact(
            original, target_tokens=9000
        )
    assert result.messages == original
    assert result.summary_calls == server.generation_count == 0
    result.messages[5]["tool_calls"][0]["function"]["name"] = "changed"
    assert original[5]["tool_calls"][0]["function"]["name"] == "read"


@pytest.mark.asyncio
async def test_compaction_keeps_instructions_recent_parallel_tools_and_output_reserve():
    server = Server()
    original = history()
    before = copy.deepcopy(original)
    tools = [{"type": "function", "function": {"name": "read"}}]
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        result = await ChatContext(
            client, "model", chat_template_kwargs={"fake_cot_mode": True}
        ).compact(
            original,
            target_tokens=1500,
            max_tokens=300,
            tools=tools,
            cache_salt="tenant-a",
        )
    assert result.messages[:2] == original[:2]
    assert result.messages[3:] == original[4:]
    assert result.messages[2]["role"] == "assistant"
    assert result.input_tokens_after + 300 <= 1500 < result.input_tokens_before + 300
    assert original == before
    generated = [b for path, b in server.requests if path == "/v1/chat/completions"]
    assert len(generated) == 1 and generated[0]["cache_salt"] == "tenant-a"
    assert "tools" not in generated[0]
    assert json.loads(generated[0]["messages"][1]["content"]) == original[2:4]
    assert all(
        b["chat_template_kwargs"] == {"fake_cot_mode": True} for _, b in server.requests
    )
    assert server.requests[-1][1]["tools"] == tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary,finish",
    [("partial", "length"), ("", "stop"), ("<think>unfinished", "stop")],
)
async def test_failed_summary_does_not_replace_history(summary, finish):
    server = Server(summary, finish)
    original = history()
    before = copy.deepcopy(original)
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        with pytest.raises(CompactionError, match="Summary did not finish"):
            await ChatContext(client, "model").compact(
                original, target_tokens=1300, max_tokens=300
            )
    assert original == before


@pytest.mark.asyncio
async def test_protected_overflow_and_oversized_old_turn_fail_before_generation():
    for target, window, error in [
        (600, 8192, "leave no summary budget"),
        (1300, 500, "old turn exceeds"),
    ]:
        server = Server()
        original = history()
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(server.handle)
        ) as client:
            with pytest.raises(CompactionError, match=error):
                await ChatContext(client, "model").compact(
                    original,
                    target_tokens=target,
                    max_tokens=300,
                    summary_window=window,
                )
        assert server.generation_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["missing_result", "orphan", "duplicate_call", "media", "late_system"]
)
async def test_invalid_transcripts_are_rejected_without_sending_private_history(
    mutation,
):
    original = history()
    if mutation == "missing_result":
        original.pop(7)
    elif mutation == "orphan":
        original[6]["tool_call_id"] = "unknown"
    elif mutation == "duplicate_call":
        original[5]["tool_calls"][1]["id"] = "a"
    elif mutation == "media":
        original[2]["content"] = [{"type": "image_url", "image_url": {"url": "data:"}}]
    else:
        original.insert(4, {"role": "system", "content": "New instructions"})
    before = copy.deepcopy(original)
    server = Server()
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        with pytest.raises(CompactionError):
            await ChatContext(client, "model").compact(original)
    assert original == before and server.requests == []


@pytest.mark.asyncio
async def test_map_batches_keep_tool_exchanges_whole_and_recheck_final_size():
    original = history()
    original[2]["content"] = "earlier information " * 40
    original[4:4] = [
        {"role": "user", "content": "more history " * 60},
        {"role": "assistant", "content": "noted"},
    ]
    server = Server()
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        result = await ChatContext(client, "model").compact(
            original,
            target_tokens=1200,
            max_tokens=200,
            keep_last_turns=1,
            summary_window=1800,
            summary_max_tokens=100,
        )
    batches = [
        json.loads(b["messages"][1]["content"])
        for path, b in server.requests
        if path == "/v1/chat/completions"
    ]
    assert [m for batch in batches for m in batch] == original[2:-1]
    assert result.summary_calls == len(batches) > 1
    for batch in batches:
        calls = {c["id"] for m in batch for c in m.get("tool_calls", [])}
        results = {m["tool_call_id"] for m in batch if m["role"] == "tool"}
        assert calls == results

    server = Server("Too long " * 200)
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        with pytest.raises(CompactionError, match="do not fit"):
            await ChatContext(client, "model").compact(
                original, target_tokens=1200, max_tokens=200
            )


@pytest.mark.asyncio
async def test_concurrent_compactions_do_not_share_summaries():
    server = Server(lambda body: body["cache_salt"])
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        context = ChatContext(client, "model")
        results = await asyncio.gather(
            *(
                context.compact(
                    history(), target_tokens=1300, max_tokens=300, cache_salt=salt
                )
                for salt in ("tenant-a", "tenant-b")
            )
        )
    assert "tenant-a" in results[0].messages[2]["content"]
    assert "tenant-b" not in results[0].messages[2]["content"]
    assert "tenant-b" in results[1].messages[2]["content"]


@pytest.mark.asyncio
async def test_completion_uses_prepared_history_and_the_counted_request_options():
    server = Server()
    original = history()
    before = copy.deepcopy(original)
    tools = [{"type": "function", "function": {"name": "read"}}]
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        prepared, completion = await ChatContext(
            client, "model", chat_template_kwargs={"fake_cot_mode": True}
        ).create_chat_completion(
            original,
            target_tokens=1500,
            max_tokens=300,
            tools=tools,
            cache_salt="tenant-a",
        )
    assert original == before
    assert prepared.summary_calls == 1 and server.generation_count == 2
    assert completion["choices"][0]["message"]["content"] == server.summary
    count_path, counted = server.requests[-2]
    answer_path, answer = server.requests[-1]
    assert count_path == "/tokenize" and answer_path == "/v1/chat/completions"
    assert answer["messages"] == counted["messages"] == prepared.messages
    assert answer["tools"] == counted["tools"] == tools
    assert answer["chat_template_kwargs"] == counted["chat_template_kwargs"]
    assert answer["cache_salt"] == "tenant-a" and answer["max_tokens"] == 300
    assert prepared.input_tokens_after + answer["max_tokens"] <= 1500


@pytest.mark.asyncio
async def test_failed_compaction_does_not_send_an_answer_or_change_history():
    server = Server(finish="length")
    original = history()
    before = copy.deepcopy(original)
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(server.handle)
    ) as client:
        with pytest.raises(CompactionError, match="finish_reason='length'"):
            await ChatContext(client, "model").create_chat_completion(
                original, target_tokens=1300, max_tokens=300
            )
    assert original == before
    assert server.generation_count == 1
