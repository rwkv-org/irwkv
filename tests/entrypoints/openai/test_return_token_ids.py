# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.tokenizers import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
PROMPT = "Complete this sentence: I love coding 🚀"
MESSAGES = [{"role": "user", "content": PROMPT}]


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--generation-config",
        "irwkv",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("return_token_ids", [True, False, None])
async def test_completion_token_ids(server, return_token_ids):
    extra_body = (
        {} if return_token_ids is None else {"return_token_ids": return_token_ids}
    )
    async with server.get_async_client() as client:
        result = await client.completions.create(
            model=MODEL_NAME,
            prompt=PROMPT,
            max_tokens=8,
            temperature=0,
            extra_body=extra_body,
        )

    choice = result.choices[0]
    if not return_token_ids:
        assert choice.prompt_token_ids is None
        assert choice.token_ids is None
        return
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    assert choice.prompt_token_ids == tokenizer.encode(PROMPT)
    assert choice.token_ids
    assert tokenizer.decode(choice.token_ids, skip_special_tokens=True) == choice.text


@pytest.mark.asyncio
@pytest.mark.parametrize("return_token_ids", [True, False, None])
async def test_chat_token_ids(server, return_token_ids):
    extra_body = (
        {} if return_token_ids is None else {"return_token_ids": return_token_ids}
    )
    async with server.get_async_client() as client:
        result = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            max_completion_tokens=8,
            temperature=0,
            extra_body=extra_body,
        )

    choice = result.choices[0]
    if not return_token_ids:
        assert result.prompt_token_ids is None
        assert choice.token_ids is None
        return
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    expected = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    assert result.prompt_token_ids == expected
    assert choice.token_ids
    assert (
        tokenizer.decode(choice.token_ids, skip_special_tokens=True)
        == choice.message.content
    )


@pytest.mark.asyncio
async def test_streamed_chat_token_ids(server):
    async with server.get_async_client() as client:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            max_completion_tokens=8,
            temperature=0,
            stream=True,
            extra_body={"return_token_ids": True},
        )
        token_ids: list[int] = []
        text = ""
        prompt_ids = None
        async for chunk in stream:
            if getattr(chunk, "prompt_token_ids", None) is not None:
                assert prompt_ids is None
                prompt_ids = chunk.prompt_token_ids
            if chunk.choices:
                choice = chunk.choices[0]
                token_ids.extend(getattr(choice, "token_ids", None) or [])
                text += choice.delta.content or ""

    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    expected = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True
    )["input_ids"]
    assert prompt_ids == expected
    assert token_ids
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == text
