# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tempfile import TemporaryDirectory

import httpx
import pytest

from irwkv import envs
from irwkv.version import __version__ as IRWKV_VERSION
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


@pytest.fixture(scope="module")
def server():
    with TemporaryDirectory() as tmpdir:
        args = [
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--enforce-eager",
            "--max-num-seqs",
            "4",
            "--uds",
            f"{tmpdir}/irwkv.sock",
        ]

        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.asyncio
async def test_show_version(server: RemoteOpenAIServer):
    transport = httpx.HTTPTransport(uds=server.uds)
    client = httpx.Client(transport=transport)
    response = client.get(server.url_for("version"))
    response.raise_for_status()

    payload = response.json()
    assert payload["version"] == IRWKV_VERSION
    if envs.IRWKV_USE_RUST_FRONTEND:
        assert payload["rust_frontend_version"] == IRWKV_VERSION
