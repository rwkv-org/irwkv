# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from http import HTTPStatus

import openai
import pytest
import pytest_asyncio
import requests
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from irwkv import version
from tests.utils import RemoteOpenAIServer

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
PREV_MINOR_VERSION = version._prev_minor_version()


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--enforce-eager",
        "--max-num-seqs",
        "4",
    ]


@pytest.fixture(
    scope="module",
    params=[
        "",
        "--enable-chunked-prefill",
        f"--show-hidden-metrics-for-version={PREV_MINOR_VERSION}",
    ],
)
def server(default_server_args, request):
    args = [*default_server_args]
    if request.param:
        args.append(request.param)

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as cl:
        yield cl


_PROMPT = "Hello my name is Robert and I love magic"


def _get_expected_values(num_requests: int, prompt_ids: list[int], max_tokens: int):
    num_prompt_tokens = len(prompt_ids)

    # {metric_family: [(suffix, expected_value)]}
    return {
        "irwkv:time_to_first_token_seconds": [("_count", num_requests)],
        "irwkv:inter_token_latency_seconds": [
            ("_count", num_requests * (max_tokens - 1))
        ],
        "irwkv:e2e_request_latency_seconds": [("_count", num_requests)],
        "irwkv:request_queue_time_seconds": [("_count", num_requests)],
        "irwkv:request_inference_time_seconds": [("_count", num_requests)],
        "irwkv:request_prefill_time_seconds": [("_count", num_requests)],
        "irwkv:request_decode_time_seconds": [("_count", num_requests)],
        "irwkv:request_num_preemptions": [("_count", num_requests)],
        "irwkv:request_prompt_tokens": [
            ("_sum", num_requests * num_prompt_tokens),
            ("_count", num_requests),
        ],
        "irwkv:request_generation_tokens": [
            ("_sum", num_requests * max_tokens),
            ("_count", num_requests),
        ],
        "irwkv:request_params_n": [("_count", num_requests)],
        "irwkv:request_params_max_tokens": [
            ("_sum", num_requests * max_tokens),
            ("_count", num_requests),
        ],
        "irwkv:iteration_tokens_total": [
            (
                "_sum",
                num_requests * (num_prompt_tokens + max_tokens),
            ),
            ("_count", num_requests * max_tokens),
        ],
        "irwkv:prompt_tokens": [("_total", num_requests * num_prompt_tokens)],
        "irwkv:generation_tokens": [("_total", num_requests * max_tokens)],
        "irwkv:request_success": [("_total", num_requests)],
    }


@pytest.mark.asyncio
async def test_metrics_counts(
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer.encode(_PROMPT)
    num_requests = 10
    max_tokens = 10

    for _ in range(num_requests):
        # sending a request triggers the metrics to be logged.
        await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt_ids,
            max_tokens=max_tokens,
            extra_body={"min_tokens": max_tokens},
        )

    response = requests.get(server.url_for("metrics"))
    print(response.text)
    assert response.status_code == HTTPStatus.OK

    # Loop over all expected metric_families
    expected_values = _get_expected_values(num_requests, prompt_ids, max_tokens)
    for metric_family, suffix_values_list in expected_values.items():
        if (
            not server.show_hidden_metrics
            and metric_family in HIDDEN_DEPRECATED_METRICS
        ):
            continue

        found_metric = False

        # Check to see if the metric_family is found in the prom endpoint.
        for family in text_string_to_metric_families(response.text):
            if family.name == metric_family:
                found_metric = True

                # Check that each suffix is found in the prom endpoint.
                for suffix, expected_value in suffix_values_list:
                    metric_name_w_suffix = f"{metric_family}{suffix}"
                    values = [
                        sample.value
                        for sample in family.samples
                        if sample.name == metric_name_w_suffix
                    ]
                    assert values, (
                        f"Did not find {metric_name_w_suffix} in prom endpoint"
                    )
                    assert sum(values) == expected_value, (
                        f"{metric_name_w_suffix} expected {expected_value}, "
                        f"found {sum(values)}"
                    )
                break

        assert found_metric, f"Did not find {metric_family} in prom endpoint"


EXPECTED_METRICS_V1 = [
    "irwkv:num_requests_running",
    "irwkv:num_requests_waiting",
    "irwkv:num_requests_waiting_by_reason",
    "irwkv:kv_cache_usage_perc",
    "irwkv:prefix_cache_queries",
    "irwkv:prefix_cache_hits",
    "irwkv:num_preemptions_total",
    "irwkv:prompt_tokens_total",
    "irwkv:generation_tokens_total",
    "irwkv:iteration_tokens_total",
    "irwkv:cache_config_info",
    "irwkv:request_success_total",
    "irwkv:request_prompt_tokens_sum",
    "irwkv:request_prompt_tokens_bucket",
    "irwkv:request_prompt_tokens_count",
    "irwkv:request_generation_tokens_sum",
    "irwkv:request_generation_tokens_bucket",
    "irwkv:request_generation_tokens_count",
    "irwkv:request_params_n_sum",
    "irwkv:request_params_n_bucket",
    "irwkv:request_params_n_count",
    "irwkv:request_params_max_tokens_sum",
    "irwkv:request_params_max_tokens_bucket",
    "irwkv:request_params_max_tokens_count",
    "irwkv:time_to_first_token_seconds_sum",
    "irwkv:time_to_first_token_seconds_bucket",
    "irwkv:time_to_first_token_seconds_count",
    "irwkv:inter_token_latency_seconds_sum",
    "irwkv:inter_token_latency_seconds_bucket",
    "irwkv:inter_token_latency_seconds_count",
    "irwkv:e2e_request_latency_seconds_sum",
    "irwkv:e2e_request_latency_seconds_bucket",
    "irwkv:e2e_request_latency_seconds_count",
    "irwkv:request_queue_time_seconds_sum",
    "irwkv:request_queue_time_seconds_bucket",
    "irwkv:request_queue_time_seconds_count",
    "irwkv:request_inference_time_seconds_sum",
    "irwkv:request_inference_time_seconds_bucket",
    "irwkv:request_inference_time_seconds_count",
    "irwkv:request_prefill_time_seconds_sum",
    "irwkv:request_prefill_time_seconds_bucket",
    "irwkv:request_prefill_time_seconds_count",
    "irwkv:request_decode_time_seconds_sum",
    "irwkv:request_decode_time_seconds_bucket",
    "irwkv:request_decode_time_seconds_count",
    "irwkv:request_num_preemptions_sum",
    "irwkv:request_num_preemptions_bucket",
    "irwkv:request_num_preemptions_count",
]

HIDDEN_DEPRECATED_METRICS: list[str] = []


@pytest.mark.asyncio
async def test_metrics_exist(
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
):
    await client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
    )

    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    for metric in EXPECTED_METRICS_V1:
        if metric in HIDDEN_DEPRECATED_METRICS and not server.show_hidden_metrics:
            continue
        assert metric in response.text

    cache_config_samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        if family.name == "irwkv:cache_config_info"
        for sample in family.samples
    ]
    assert cache_config_samples
    for sample in cache_config_samples:
        assert sample.labels.get("kv_cache_size_tokens") not in (None, "None", "")
        assert sample.labels.get("kv_cache_max_concurrency") not in (None, "None", "")


@pytest.mark.asyncio
async def test_abort_metrics_reset(
    server: RemoteOpenAIServer,
    client: openai.AsyncClient,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer.encode(_PROMPT)

    running_requests, waiting_requests, kv_cache_usage = _get_running_metrics_from_api(
        server,
    )

    # Expect no running requests or kvcache usage
    assert running_requests == 0
    assert waiting_requests == 0
    assert kv_cache_usage == 0.0

    # Start some long-running requests that we can abort
    tasks = []
    for _ in range(3):
        task = asyncio.create_task(
            client.completions.create(
                model=MODEL_NAME,
                prompt=prompt_ids,
                max_tokens=500,  # Long generation to give time to abort
                temperature=0.0,
                extra_body={"min_tokens": 500},
            )
        )
        tasks.append(task)

    # Poll until we see running requests rather than using a fixed sleep,
    # since generation speed varies across hardware.
    try:
        await _poll_until(
            lambda: _get_running_metrics_from_api(server)[0] > 0,
            timeout=10.0,
            interval=0.1,
            description="running_requests > 0",
        )
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        pytest.fail("Requests never appeared as running in metrics")

    # Check that we have running requests
    running_requests, waiting_requests, kv_cache_usage = _get_running_metrics_from_api(
        server,
    )

    # Expect running requests and kvcache usage
    assert running_requests > 0
    assert kv_cache_usage > 0

    # Cancel all tasks to abort the requests
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Poll until metrics reset rather than using a fixed sleep
    await _poll_until(
        lambda: _get_running_metrics_from_api(server) == (0, 0, 0),
        timeout=10.0,
        interval=0.2,
        description="gauge metrics back to zero",
    )

    # Verify running and waiting requests counts and KV cache usage are zero
    running_requests_after, waiting_requests_after, kv_cache_usage_after = (
        _get_running_metrics_from_api(server)
    )

    assert running_requests_after == 0, (
        f"Expected 0 running requests after abort, got {running_requests_after}"
    )
    assert waiting_requests_after == 0, (
        f"Expected 0 waiting requests after abort, got {waiting_requests_after}"
    )
    assert kv_cache_usage_after == 0, (
        f"Expected 0% KV cache usage after abort, got {kv_cache_usage_after}"
    )


async def _poll_until(
    predicate, *, timeout: float, interval: float = 0.5, description: str = "condition"
):
    """Poll until predicate() returns True, or raise TimeoutError."""
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError(f"Timed out after {timeout}s waiting for: {description}")


def _get_running_metrics_from_api(server: RemoteOpenAIServer):
    """Return (running_count, waiting_count, kv_cache_usage)."""
    response = requests.get(server.url_for("metrics"))
    assert response.status_code == HTTPStatus.OK

    # Verify running and waiting requests counts and KV cache usage are zero
    running_requests, waiting_requests, kv_cache_usage = None, None, None

    kv_cache_usage_metric = "irwkv:kv_cache_usage_perc"

    for family in text_string_to_metric_families(response.text):
        if family.name == "irwkv:num_requests_running":
            for sample in family.samples:
                if sample.name == "irwkv:num_requests_running":
                    running_requests = sample.value
                    break
        elif family.name == "irwkv:num_requests_waiting":
            for sample in family.samples:
                if sample.name == "irwkv:num_requests_waiting":
                    waiting_requests = sample.value
                    break
        elif family.name == kv_cache_usage_metric:
            for sample in family.samples:
                if sample.name == kv_cache_usage_metric:
                    kv_cache_usage = sample.value
                    break

    assert running_requests is not None
    assert waiting_requests is not None
    assert kv_cache_usage is not None

    return running_requests, waiting_requests, kv_cache_usage
