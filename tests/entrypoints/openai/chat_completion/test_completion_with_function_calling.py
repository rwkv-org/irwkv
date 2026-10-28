# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "rwkv",
        "--reasoning-parser",
        "rwkv",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_named_tool_choice_must_match_tools(client: openai.AsyncOpenAI):
    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "unknown"}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_named_tool_choice(client, stream):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_completion_tokens=256,
        temperature=0,
        stream=stream,
    )
    if stream:
        names = []
        arguments = []
        reasons = []
        async for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                reasons.append(choice.finish_reason)
            for call in choice.delta.tool_calls or []:
                if call.function.name:
                    names.append(call.function.name)
                if call.function.arguments:
                    arguments.append(call.function.arguments)
        assert "".join(names) == "get_weather"
        assert isinstance(json.loads("".join(arguments))["city"], str)
        assert reasons == ["tool_calls"]
    else:
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        call = choice.message.tool_calls[0]
        assert call.function.name == "get_weather"
        assert isinstance(json.loads(call.function.arguments)["city"], str)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_max_tokens_with_tool_choice_required(client, tool_choice, stream):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        tools=TOOLS,
        tool_choice=tool_choice,
        max_completion_tokens=1,
        temperature=0,
        stream=stream,
    )
    if stream:
        reasons = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].finish_reason:
                reasons.append(chunk.choices[0].finish_reason)
        assert reasons == ["length"]
    else:
        assert response.choices[0].finish_reason == "length"
        assert response.choices[0].message.tool_calls is None
