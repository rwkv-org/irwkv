# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``irwkv bench throughput`` CLI.

Throughput reuses ``bench serve``'s ``datasets.get_samples`` dispatch and
covers request formatting and CLI dataset resolution.
"""

import subprocess
from argparse import Namespace

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from irwkv.benchmarks.datasets import SampleRequest
from irwkv.benchmarks.throughput import (
    _run_irwkv_chat_requests,
    _to_serve_args,
    add_cli_args,
    get_requests,
)
from irwkv.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


@pytest.mark.benchmark
def test_bench_throughput():
    command = [
        "irwkv",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


def test_bench_throughput_accepts_custom_audio_args():
    parser = FlexibleArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(
        [
            "--dataset-name",
            "custom_audio",
            "--dataset-path",
            "audio.jsonl",
            "--no-oversample",
            "--custom-output-len",
            "32",
            "--enable-multimodal-chat",
        ]
    )

    assert args.dataset_name == "custom_audio"
    assert args.no_oversample
    assert args.custom_output_len == 32
    assert args.enable_multimodal_chat


def test_irwkv_chat_requests_include_multimodal_content():
    class FakeLLM:
        def __init__(self):
            self.prompts = None

        def chat(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            self.prompts = prompts
            return []

    llm = FakeLLM()
    audio_content = {
        "type": "input_audio",
        "input_audio": {"data": "abc", "format": "wav"},
    }
    request = SampleRequest(
        prompt="Transcribe this audio.",
        prompt_len=1,
        expected_output_len=8,
        multi_modal_data=audio_content,
    )

    _run_irwkv_chat_requests(
        llm,
        [request],
        n=1,
        disable_detokenize=False,
        do_profile=False,
        prequeue_requests=False,
    )

    assert llm.prompts == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    audio_content,
                ],
            }
        ]
    ]


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("rwkv-rs/rwkv7-g1-st")


# -----------------------------
# Issue #50838: MMVU via throughput
# -----------------------------


@pytest.mark.benchmark
def test_get_requests_resolves_mmvu(
    monkeypatch: pytest.MonkeyPatch,
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Issue #50838: --dataset-path yale-nlp/MMVU is accepted by throughput.

    Previously throughput's validate_args rejected MMVU ("is not supported by
    hf dataset"). It now resolves through the shared get_samples. The HF
    download is stubbed so the resolution + dispatch path is exercised without
    network access.
    """
    import irwkv.benchmarks.datasets.datasets as dsmod

    class _StubMMVUDataset:
        # MMVU is multimodal in content but, like the real class, does not set
        # IS_MULTIMODAL -- so it is not subject to the backend gate.
        IS_MULTIMODAL = False
        SUPPORTED_DATASET_PATHS = {"yale-nlp/MMVU": lambda x: x["question"]}

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def sample(self, *, num_requests, tokenizer, **kwargs):  # noqa: ARG002
            return [
                SampleRequest(prompt="q", prompt_len=1, expected_output_len=1)
                for _ in range(num_requests)
            ]

    monkeypatch.setattr(dsmod, "MMVUDataset", _StubMMVUDataset)

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "hf",
            "--dataset-path",
            "yale-nlp/MMVU",
            "--backend",
            "irwkv-chat",
            "--num-prompts",
            "3",
        ]
    )
    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 3
    assert all(isinstance(r, SampleRequest) for r in requests)


@pytest.mark.benchmark
def test_get_requests_allows_multimodal_on_plain_irwkv_backend(
    monkeypatch: pytest.MonkeyPatch,
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Regression: --backend irwkv must accept multimodal datasets such as ASR.

    _run_irwkv_requests forwards multi_modal_data straight to LLM.generate, so
    the irwkv-chat-only gate in get_samples rejected a backend that always
    worked.
    """
    import irwkv.benchmarks.datasets.datasets as dsmod

    class _StubASRDataset:
        IS_MULTIMODAL = True
        SUPPORTED_DATASET_PATHS = {"openslr/librispeech_asr"}

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def sample(self, *, num_requests, tokenizer, **kwargs):  # noqa: ARG002
            return [
                SampleRequest(prompt="q", prompt_len=1, expected_output_len=1)
                for _ in range(num_requests)
            ]

    monkeypatch.setattr(dsmod, "ASRDataset", _StubASRDataset)

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "hf",
            "--dataset-path",
            "openslr/librispeech_asr",
            "--backend",
            "irwkv",
            "--num-prompts",
            "3",
        ]
    )
    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 3
    assert all(isinstance(r, SampleRequest) for r in requests)


def test_to_serve_args_keeps_legacy_flag_fallback() -> None:
    """Callers that only set the legacy flags still get them mapped."""
    args = Namespace(
        input_len=256,
        output_len=8,
        prefix_len=16,
        backend="irwkv",
    )
    serve_args = _to_serve_args(args)

    assert serve_args.random_input_len == 256
    assert serve_args.random_output_len == 8
    assert serve_args.random_prefix_len == 16
    assert serve_args.backend == "irwkv"
    assert not serve_args.enable_multimodal_chat
