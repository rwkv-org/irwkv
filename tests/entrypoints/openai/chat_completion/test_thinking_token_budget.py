# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for ``thinking_token_budget`` with RWKV."""

import openai
import pytest
import pytest_asyncio

from irwkv.tokenizers import get_tokenizer
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]
THINK_BUDGET = 5

REASONING_START_STR = "<think>"
REASONING_END_STR = "</think>"


def _count_reasoning_decode_token_ids_between_markers(
    full_token_ids: list[int],
    reasoning_start_ids: list[int],
    reasoning_end_ids: list[int],
) -> int | None:
    """Count decode tokens in the thinking span (after last start, before first end)."""
    if not reasoning_start_ids or not reasoning_end_ids:
        raise ValueError("reasoning marker token id lists must be non-empty")

    def _last_subseq_index(haystack: list[int], needle: list[int]) -> int:
        n = len(needle)
        if n > len(haystack):
            return -1
        for i in range(len(haystack) - n, -1, -1):
            if haystack[i : i + n] == needle:
                return i
        return -1

    last_start = _last_subseq_index(full_token_ids, reasoning_start_ids)
    if last_start < 0:
        return None

    pos_after_start = last_start + len(reasoning_start_ids)
    end_n = len(reasoning_end_ids)
    for j in range(pos_after_start, len(full_token_ids) - end_n + 1):
        if full_token_ids[j : j + end_n] == reasoning_end_ids:
            return j - pos_after_start
    return len(full_token_ids) - pos_after_start


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "rwkv",
        "--reasoning-config",
        '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}',
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_auto_reasoning_config():
    args = [
        "--reasoning-parser",
        "rwkv",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(request, server, server_with_auto_reasoning_config):
    server_map = {
        "default": server,
        "auto_config": server_with_auto_reasoning_config,
    }
    target_server = server_map[request.param]
    async with target_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_mixed_requests(client: openai.AsyncOpenAI):
    """Test that mixed requests (some with thinking_token_budget, some without)
    complete successfully without errors."""
    response_with_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    response_without_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
    )

    msg_with = response_with_budget.choices[0].message
    msg_without = response_without_budget.choices[0].message

    assert msg_with.content or getattr(msg_with, "reasoning", None)
    assert msg_without.content or getattr(msg_without, "reasoning", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_limits_reasoning(client: openai.AsyncOpenAI):
    """Test that thinking_token_budget limits the number of reasoning tokens.

    Counts reasoning decode tokens by id, which is robust to how tokens are
    grouped into streamed chunks (a single chunk can carry several tokens under
    async scheduling / stream_interval > 1). Counting chunks under-counts.
    """
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    start_ids = list(tokenizer.encode(REASONING_START_STR, add_special_tokens=False))
    end_ids = list(tokenizer.encode(REASONING_END_STR, add_special_tokens=False))

    prompt_token_ids: list[int] = []
    decode_token_ids: list[int] = []
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        stream=True,
        extra_body={"thinking_token_budget": THINK_BUDGET, "return_token_ids": True},
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        if getattr(chunk, "prompt_token_ids", None):
            prompt_token_ids = list(chunk.prompt_token_ids)
        delta_ids = getattr(chunk.choices[0], "token_ids", None)
        if delta_ids:
            decode_token_ids.extend(delta_ids)

    reasoning_token_count = _count_reasoning_decode_token_ids_between_markers(
        prompt_token_ids + decode_token_ids, start_ids, end_ids
    )
    assert reasoning_token_count is not None, "missing reasoning start marker in ids"
    assert reasoning_token_count == THINK_BUDGET, (
        f"reasoning tokens ({reasoning_token_count}) != "
        f"thinking_token_budget ({THINK_BUDGET})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_streaming_with_thinking_disabled_stays_in_content(
    client: openai.AsyncOpenAI,
):
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Which is larger, 4 or 12?"
                " Output exactly one token: 4 or 12.",
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"fake_cot_mode": True}},
    }

    response = await client.chat.completions.create(**request_kwargs)
    message = response.choices[0].message
    assert message.content is not None and message.content.strip() != ""
    assert getattr(message, "reasoning", None) in (None, "")

    stream = await client.chat.completions.create(
        **request_kwargs,
        stream=True,
    )

    content_chunks = []
    reasoning_chunks = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_chunks.append(delta.content)
        if getattr(delta, "reasoning", None):
            reasoning_chunks.append(delta.reasoning)

    assert "".join(content_chunks).strip() != ""
    assert reasoning_chunks == []
