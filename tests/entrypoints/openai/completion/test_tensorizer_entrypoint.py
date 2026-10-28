# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import tempfile

import openai
import pytest
import pytest_asyncio
import torch.cuda

from irwkv.engine.arg_utils import EngineArgs
from irwkv.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    tensorize_irwkv_model,
)
from irwkv.platforms import current_platform
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


def _cleanup():
    gc.collect()
    torch.accelerator.empty_cache()


@pytest.fixture(autouse=True)
def cleanup():
    _cleanup()


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.fixture(scope="module")
def model_uri(tmp_dir):
    yield f"{tmp_dir}/model.tensors"


@pytest.fixture(scope="module")
def tensorize_model(model_uri):
    tensorizer_config = TensorizerConfig(tensorizer_uri=model_uri)
    args = EngineArgs(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        enforce_eager=True,
    )

    tensorize_irwkv_model(args, tensorizer_config)

    # Manually invoke a _cleanup() here, as the cleanup()
    # fixture won't be guaranteed to be called after this
    # when this fixture is used for a test
    _cleanup()
    yield


@pytest.fixture(scope="module")
def server(model_uri, tensorize_model):
    # In this case, model_uri is a directory with a model.tensors
    # file and all necessary model artifacts, particularly a
    # HF `config.json` file. In this case, Tensorizer can infer the
    # `TensorizerConfig` so --model-loader-extra-config can be completely
    # omitted.

    ## Start OpenAI API server
    args = [
        "--load-format",
        "tensorizer",
        "--served-model-name",
        MODEL_NAME,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.2",
        "--enforce-eager",
        "--generation-config",
        "irwkv",
    ]
    if current_platform.is_rocm():
        args += ["--attention-backend", "TRITON_ATTN"]

    model_dir = os.path.dirname(model_uri)
    with RemoteOpenAIServer(model_dir, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_completion(client: openai.AsyncOpenAI, model_name: str):
    _cleanup()
    completion = await client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
        logprobs=1,
    )

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.model == MODEL_NAME
    assert len(completion.choices) == 1
    assert completion.choices[0].finish_reason in {"length", "stop"}
    assert completion.usage is not None
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )

    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.2",
        "--enforce-eager",
        "--generation-config",
        "irwkv",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as reference_server:
        async with reference_server.get_async_client() as reference_client:
            reference = await reference_client.completions.create(
                model=model_name,
                prompt="Hello, my name is",
                max_tokens=5,
                temperature=0.0,
                logprobs=1,
            )
    assert completion.choices[0].text == reference.choices[0].text
    assert completion.choices[0].logprobs.token_logprobs == pytest.approx(
        reference.choices[0].logprobs.token_logprobs, abs=1e-4
    )
