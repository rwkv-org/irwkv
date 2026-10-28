# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests
import urllib3

from ..utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


def generate_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed certificate for testing."""
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_file, key_file


class RemoteOpenAIServerSSL(RemoteOpenAIServer):
    """RemoteOpenAIServer subclass that supports SSL with self-signed certs."""

    @property
    def url_root(self) -> str:
        return f"https://{self.host}:{self.port}"

    def _wait_for_server(self, *, url: str, timeout: float):
        """Override to use HTTPS with SSL verification disabled."""
        # Suppress InsecureRequestWarning for self-signed certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        start = time.time()
        while True:
            try:
                if requests.get(url, verify=False).status_code == 200:
                    break
            except Exception:
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from None


@pytest.fixture(scope="function")
def server():
    args = ["--max-model-len", "1024", "--enforce-eager", "--load-format", "dummy"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="function")
def ssl_server():
    """Start a irwkv server with SSL enabled using a self-signed certificate."""
    with tempfile.TemporaryDirectory() as cert_dir:
        cert_file, key_file = generate_self_signed_cert(Path(cert_dir))
        args = [
            "--max-model-len",
            "1024",
            "--enforce-eager",
            "--load-format",
            "dummy",
            "--ssl-certfile",
            str(cert_file),
            "--ssl-keyfile",
            str(key_file),
        ]

        with RemoteOpenAIServerSSL(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.mark.benchmark
def test_bench_serve_token_ids_without_tokenizer(tmp_path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(json.dumps({"prompt": [1, 2, 3]}) + "\n")
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        command = [
            "irwkv",
            "bench",
            "serve",
            "--model",
            MODEL_NAME,
            "--host",
            server.host,
            "--port",
            str(server.port),
            "--backend",
            "openai",
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            "custom",
            "--dataset-path",
            str(dataset),
            "--custom-output-len",
            "3",
            "--num-prompts",
            "1",
            "--skip-tokenizer-init",
            "--ignore-eos",
            "--save-result",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "result.json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "result.json").read_text())
    assert metrics["completed"] == 1
    assert metrics["total_input_tokens"] == 3
    assert metrics["total_output_tokens"] == 3


@pytest.mark.benchmark
def test_bench_serve(server):
    # Test default model detection and input/output len
    command = [
        "irwkv",
        "bench",
        "serve",
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_insecure(ssl_server):
    """Test --insecure flag with an HTTPS server using a self-signed certificate."""
    base_url = f"https://{ssl_server.host}:{ssl_server.port}"
    command = [
        "irwkv",
        "bench",
        "serve",
        "--base-url",
        base_url,
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "5",
        "--insecure",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


@pytest.mark.benchmark
def test_bench_serve_chat(server):
    command = [
        "irwkv",
        "bench",
        "serve",
        "--model",
        MODEL_NAME,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        "5",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
