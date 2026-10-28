# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.tokenizers import get_tokenizer
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--skip-tokenizer-init",
        "--load-format",
        "dummy",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_token_in_token_out(server):
    """Return prompt and generated token IDs without a server tokenizer."""
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    text = "Hello, world! How are you today?"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    async with server.get_async_client() as client:
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt=token_ids,
            max_tokens=20,
            temperature=0,
            echo=True,
            extra_body={
                "return_token_ids": True,
            },
        )

        # Verify all fields are present
        assert (
            completion.choices[0].token_ids is not None
            and 0 < len(completion.choices[0].token_ids) <= 20
        )
        assert completion.choices[0].prompt_token_ids == token_ids
