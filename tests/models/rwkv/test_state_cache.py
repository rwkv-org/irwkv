# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import struct
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from irwkv.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from irwkv.distributed.kv_transfer.kv_connector.v1.rwkv_connector import (
    RwkvCacheIndex,
    RwkvStateConnector,
)
from irwkv.v1.rwkv_state import (
    RwkvState,
    RwkvStateCache,
    context_prefix_sha1,
    context_sha1,
    model_fingerprint,
)


@pytest.fixture
def state():
    return RwkvState(
        "model",
        (1, 2, 257),
        {
            "tmix": torch.randn(2, 128).bfloat16(),
            "cmix": torch.randn(2, 128).bfloat16(),
            "wkv": torch.randn(2, 2, 64, 64),
        },
        "tenant",
    )


def restore(data, state, **kwargs):
    return RwkvState.from_bytes(
        data,
        model_id=kwargs.get("model_id", state.model_id),
        shapes={
            name: (value.shape, value.dtype) for name, value in state.tensors.items()
        },
        max_bytes=1024 * 1024,
        max_tokens=1000,
    )


def test_binary_state_round_trip_preserves_every_bit_and_context(state):
    result = restore(state.to_bytes(), state)
    assert result.tokens == state.tokens and result.cache_salt == state.cache_salt
    assert (
        context_sha1(state.tokens)
        == hashlib.sha1(struct.pack("<III", 1, 2, 257)).hexdigest()
    )
    for name in state.tensors:
        assert torch.equal(state.tensors[name], result.tensors[name])


def test_prefix_hashes_match_independent_little_endian_hashes():
    tokens = [1, 2, 257, 65535]
    actual = context_prefix_sha1(tokens, [4, 0, 3, 1, 4])
    expected = {
        n: hashlib.sha1(struct.pack(f"<{n}I", *tokens[:n])).hexdigest()
        for n in (0, 1, 3, 4)
    }
    assert actual == expected


@pytest.mark.parametrize(
    "change",
    ["truncated", "trailing", "checksum", "oversize", "model", "nan", "context"],
)
def test_invalid_state_is_rejected_before_restore(state, change):
    data = state.to_bytes()
    if change == "truncated":
        data = data[:-3]
    elif change == "trailing":
        data += b"garbage"
    elif change == "checksum":
        data = data[:16] + bytes([data[16] ^ 1]) + data[17:]
    elif change == "oversize":
        data = data[:8] + struct.pack("<Q", 2**40) + data[16:]
    elif change == "nan":
        state.tensors["wkv"].fill_(float("nan"))
        data = state.to_bytes()
    elif change == "context":
        state.tensors["wkv"] = torch.zeros(1)
        data = state.to_bytes()
        state.tensors["wkv"] = torch.zeros(2, 2, 64, 64)
    with pytest.raises(ValueError):
        restore(
            data, state, model_id="different" if change == "model" else state.model_id
        )


def test_lru_cache_honors_budget_and_keeps_recently_used_snapshot(state):
    cache = RwkvStateCache(2 * state.nbytes)
    first = cache.add(state)
    second = cache.add(replace(state, tokens=(1, 2, 258)))
    cache.get(first)
    third = cache.add(replace(state, tokens=(1, 2, 259)))
    assert set(cache.entries) == {first, third}
    assert second not in cache.entries and cache.nbytes <= cache.max_bytes


def test_cache_dedup_preserves_id_and_refreshes_lru_without_evicting(state):
    cache = RwkvStateCache(2 * state.nbytes)
    first = cache.add(state)
    second = cache.add(replace(state, tokens=(1, 2, 258)))
    generation = cache.generation
    duplicate = cache.add(replace(state, request_id="another request"))
    assert duplicate == first
    assert list(cache.entries) == [second, first]
    assert cache.get(first) is state
    assert cache.nbytes == 2 * state.nbytes
    assert cache.generation == generation


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "another model"},
        {"cache_salt": "another tenant"},
        {"tokens": (1, 2, 258)},
        {"tokens": (1, 2)},
    ],
)
def test_cache_dedup_requires_same_model_salt_and_entire_context(state, change):
    cache = RwkvStateCache(3 * state.nbytes)
    first = cache.add(state)
    second = cache.add(replace(state, **change))
    assert first != second and len(cache.entries) == 2


@pytest.mark.parametrize("uploaded_first", [False, True])
def test_cache_dedup_never_merges_uploaded_states_by_context(state, uploaded_first):
    cache = RwkvStateCache(4 * state.nbytes)
    uploaded = replace(
        state,
        automatic=False,
        tensors={name: tensor + 1 for name, tensor in state.tensors.items()},
    )
    for snapshot in (uploaded, state) if uploaded_first else (state, uploaded):
        cache.add(snapshot)
    uploaded_id = cache.add(uploaded)
    automatic_id = cache.add(replace(state, request_id="duplicate"))
    assert len(cache.entries) == 3
    assert cache.get(uploaded_id) is uploaded
    assert cache.get(automatic_id) is state
    assert cache.nbytes == 3 * state.nbytes


@pytest.mark.parametrize("evicted", [False, True])
def test_removed_context_can_be_cached_again(state, evicted):
    cache = RwkvStateCache(state.nbytes)
    original = cache.add(state)
    if evicted:
        cache.add(replace(state, tokens=(1, 2, 258)))
    else:
        cache.remove(original)
    restored = cache.add(state)
    assert restored != original and list(cache.entries) == [restored]
    assert cache.nbytes == state.nbytes


@pytest.mark.parametrize("pending", [False, True])
def test_connector_dedup_releases_duplicate_slot_without_another_copy(state, pending):
    from irwkv.distributed.kv_transfer.kv_connector.v1.rwkv_connector import (
        RwkvConnectorMetadata,
    )

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"max_bytes": 4 * state.nbytes}
        )
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    connector._model_id = state.model_id
    event = Mock()
    event.query.return_value = False
    if pending:
        connector._pending_saves["original"] = (event, state)
        connector._pending_contexts[state.context_key] = "original"
    else:
        connector._cache.add(state)
    connector.bind_connector_metadata(
        RwkvConnectorMetadata(cursors={"duplicate": len(state.tokens)})
    )
    connector.start_load_kv(None)
    connector.bind_connector_metadata(
        RwkvConnectorMetadata(
            saves={"duplicate": (2, state.tokens + (4,), state.cache_salt)}
        )
    )
    connector.start_load_kv(None)
    assert connector.get_transfer_results(set()).finished_sending == {"duplicate"}
    if pending:
        event.query.return_value = True
        assert connector.get_transfer_results(set()).finished_sending == {"original"}
    assert list(connector._cache.entries.values()) == [state]
    assert connector._cache.nbytes == state.nbytes
    assert not connector._pending_contexts


def test_cleared_pending_context_does_not_remove_its_replacement(state):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"max_bytes": 4 * state.nbytes}
        )
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    event = Mock()
    event.query.return_value = True
    connector._pending_saves["old"] = (event, state)
    connector._pending_contexts[state.context_key] = "old"
    connector.state_operation("clear")
    replacement_event = Mock()
    replacement_event.query.return_value = False
    connector._pending_saves["new"] = (replacement_event, state)
    connector._pending_contexts[state.context_key] = "new"
    assert connector.get_transfer_results(set()).finished_sending == {"old"}
    assert connector._pending_contexts[state.context_key] == "new"
    assert not connector._cache.entries


def test_model_fingerprint_uses_weights_cache_and_original_vocabulary(
    tmp_path, monkeypatch
):
    import huggingface_hub

    from irwkv.tokenizers import rwkv

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    vocab = tmp_path / "original-vocab.txt"
    vocab.write_text("original vocabulary")
    monkeypatch.setattr(rwkv, "RWKV_VOCAB_PATH", vocab)

    def snapshot(repo, **kwargs):
        assert repo == "rwkv/model"
        assert kwargs["cache_dir"] == "weights-cache" and kwargs["local_files_only"]
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    config = SimpleNamespace(
        model="rwkv/model",
        revision="revision",
        tokenizer=str(tokenizer),
        hf_config=SimpleNamespace(to_dict=lambda: {"hidden_size": 128}),
    )
    first = model_fingerprint(config, "weights-cache")
    (tokenizer / "tokenizer.json").write_text('{"changed": true}')
    assert model_fingerprint(config, "weights-cache") == first
    vocab.write_text("changed original vocabulary")
    assert model_fingerprint(config, "weights-cache") != first


def test_lookup_requires_exact_consumed_prefix_and_cache_salt(state):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    connector.update_index(RwkvCacheIndex(1, [state.metadata("state-id")]))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1, 2, 257, 4],
        prompt_embeds=None,
        cache_salt="tenant",
        kv_transfer_params=None,
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (3, True)
    request.prompt_token_ids[1] = 9
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    request.prompt_token_ids = [1, 2, 257]
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    request.prompt_token_ids.append(4)
    request.cache_salt = "another tenant"
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_index(RwkvCacheIndex(0, []))
    assert connector._index.generation == 1


def test_conversation_branches_reuse_longest_matching_history(state):
    cache = RwkvStateCache(4 * state.nbytes)
    first = cache.add(state)
    second = cache.add(replace(state, tokens=state.tokens + (10, 11)))
    branch = cache.add(replace(state, tokens=state.tokens + (20, 21)))
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    connector.update_index(RwkvCacheIndex(cache.generation, cache.metadata()))
    request = SimpleNamespace(
        request_id="conversation",
        prompt_embeds=None,
        cache_salt=state.cache_salt,
        kv_transfer_params=None,
    )
    for suffix, expected in [
        ((30,), first),
        ((10, 11, 30), second),
        ((20, 21, 30), branch),
    ]:
        request.prompt_token_ids = list(state.tokens + suffix)
        count = len(cache.get(expected).tokens)
        assert connector.get_num_new_matched_tokens(request, 0) == (count, True)
        assert connector._matches[request.request_id] == expected
    assert set(cache.entries) == {first, second, branch}


def test_uploaded_state_requires_explicit_selection(state):
    uploaded = restore(state.to_bytes(), state)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    connector.update_index(RwkvCacheIndex(1, [uploaded.metadata("uploaded")]))
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[1, 2, 257, 4],
        prompt_embeds=None,
        cache_salt="tenant",
        kv_transfer_params=None,
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    request.kv_transfer_params = {"rwkv_state_id": "uploaded"}
    assert connector.get_num_new_matched_tokens(request, 0) == (3, True)


@pytest.mark.parametrize("local_reset", [False, True])
def test_engine_reset_clears_host_cache_before_acknowledging(state, local_reset):
    from irwkv.v1.engine.core import EngineCore

    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
    )
    connector = RwkvStateConnector(config, KVConnectorRole.SCHEDULER, None)
    connector.update_index(RwkvCacheIndex(1, [state.metadata("cached")]))
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        connector=connector, reset_prefix_cache=Mock(return_value=local_reset)
    )
    core.model_executor = SimpleNamespace(
        collective_rpc=Mock(return_value=[(None, RwkvCacheIndex(2, []))])
    )
    assert core.reset_prefix_cache(reset_connector=True) == local_reset
    core.scheduler.reset_prefix_cache.assert_called_once_with(False, False)
    assert not connector._reset_pending
    if local_reset:
        core.model_executor.collective_rpc.assert_called_once_with(
            "rwkv_state", args=("clear", None)
        )
        assert not connector._index.entries
    else:
        core.model_executor.collective_rpc.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_scheduler_resumes_unaligned_state_and_holds_slot_until_saved(
    state, asynchronous
):
    from irwkv.sampling_params import SamplingParams
    from irwkv.v1.attention.backends.rwkv_attn import RwkvStateSpec
    from irwkv.v1.request import Request
    from tests.v1.core.utils import create_model_runner_output, create_scheduler

    spec = RwkvStateSpec(
        block_size=128,
        shapes=((2, 128),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="none",
    )
    scheduler = create_scheduler(
        use_kv_connector="RwkvStateConnector",
        kv_role="kv_both",
        kv_cache_spec=spec,
        max_model_len=128,
        max_num_batched_tokens=32,
        block_size=128,
        num_blocks=4,
        async_scheduling=asynchronous,
    )
    scheduler.connector.update_index(RwkvCacheIndex(1, [state.metadata("state")]))
    request = Request(
        "request", [1, 2, 257, 4], SamplingParams(max_tokens=1), cache_salt="tenant"
    )
    scheduler.add_request(request)
    loading = scheduler.schedule()
    assert loading.total_num_scheduled_tokens == 0
    assert loading.kv_connector_metadata.loads["request"][0] == "state"
    scheduler.update_from_output(
        loading, create_model_runner_output([], finished_recving={"request"})
    )
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"request": 1}
    scheduler.update_from_output(
        resumed, create_model_runner_output([request], token_id=5)
    )
    saving = scheduler.schedule()
    slot, tokens, salt = saving.kv_connector_metadata.saves["request"]
    assert tokens == (1, 2, 257, 4, 5) and salt == "tenant"
    assert resumed.kv_connector_metadata.cursors == {"request": 4}
    assert scheduler.kv_cache_manager.get_block_ids("request")[0] == [slot]
    scheduler.update_from_output(
        saving, create_model_runner_output([], finished_sending={"request"})
    )
    assert scheduler.kv_cache_manager.get_block_ids("request")[0] == []


def test_python_api_transfers_binary_state_without_json_encoding(state):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from irwkv.entrypoints.serve.rwkv_state import router

    data = state.to_bytes()

    async def operate(operation, value):
        if operation == "export":
            return data
        if operation == "import":
            assert value == data
            return state.metadata("uploaded")
        return []

    app = FastAPI()
    app.state.engine_client = SimpleNamespace(rwkv_state=operate)
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/v1/rwkv/states/state")
        assert response.content == data
        assert response.headers["content-type"] == "application/octet-stream"
        response = client.post("/v1/rwkv/states", content=data)
        assert response.status_code == 201
        assert response.json()["context_sha1"] == state.context_sha1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA state transfer")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cuda_snapshot_consumed_cursor_and_async_restore(monkeypatch, dtype):
    from irwkv.distributed.kv_transfer.kv_connector.v1 import rwkv_connector
    from irwkv.model_executor.models.rwkv import RwkvStateLayer
    from tests.models.rwkv.test_rwkv import _state_irwkv_config

    config = _state_irwkv_config(dtype)
    config.device_config.device = torch.device("cuda")
    config.load_config = SimpleNamespace(download_dir=None)
    config.kv_transfer_config = SimpleNamespace(
        kv_connector_extra_config={"max_bytes": 1024**2}
    )
    monkeypatch.setattr(
        rwkv_connector, "model_fingerprint", lambda _, cache_dir: "model"
    )
    layer = RwkvStateLayer(config, 2, "model.layers.0.rwkv_state")
    size = layer.get_kv_cache_spec(config).page_size_bytes
    raw = torch.zeros(4 * size, dtype=torch.int8, device="cuda")
    layer.bind_kv_cache(raw)
    for tensor in layer._state_pools().values():
        tensor[:, 1].normal_()
    expected = {name: pool[:, 1].cpu() for name, pool in layer._state_pools().items()}
    connector = RwkvStateConnector(config, KVConnectorRole.WORKER, None)
    connector.register_kv_caches({layer.prefix: raw})
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(cursors={"old": 3})
    )
    connector.start_load_kv(None)
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(saves={"old": (1, (1, 2, 3, 4), None)})
    )
    connector.start_load_kv(None)
    torch.cuda.synchronize()
    result = connector.get_transfer_results({"old"})
    assert result.finished_sending == {"old"}
    state_id, state = next(iter(connector._cache.entries.items()))
    assert state.tokens == (1, 2, 3)
    for name in expected:
        assert torch.equal(state.tensors[name], expected[name])
        layer._state_pools()[name][:, 1].zero_()
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(loads={"new": (state_id, 2)})
    )
    connector.start_load_kv(None)
    torch.cuda.synchronize()
    assert connector.get_transfer_results(set()).finished_recving == {"new"}
    for name in expected:
        assert torch.equal(layer._state_pools()[name][:, 2].cpu(), expected[name])
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(loads={"missing": ("evicted", 3)})
    )
    connector.start_load_kv(None)
    result = connector.get_transfer_results(set())
    assert result.failed_recving == result.finished_recving == {"missing"}
    assert connector.get_block_ids_with_load_errors() == {3}
    imported_id = connector._cache.add(replace(state, automatic=False))
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(
            loads={"uploaded": (imported_id, 2)},
            cursors={"uploaded": 3},
        )
    )
    connector.start_load_kv(None)
    connector.bind_connector_metadata(
        rwkv_connector.RwkvConnectorMetadata(
            saves={"uploaded": (2, (1, 2, 3, 4), None)},
        )
    )
    connector.start_load_kv(None)
    torch.cuda.synchronize()
    connector.get_transfer_results({"uploaded"})
    assert not next(reversed(connector._cache.entries.values())).automatic
    connector.shutdown()


@pytest.mark.skipif(
    not os.environ.get("IRWKV_RWKV_TEST_MODEL"),
    reason="Requires a local RWKV checkpoint",
)
@pytest.mark.parametrize(
    "asynchronous,enforce_eager",
    [(False, True), (True, True), (True, False)],
    ids=["eager-sync", "eager-async", "graph-async"],
)
def test_real_model_uploaded_snapshot_matches_full_prefill(
    irwkv_runner, asynchronous, enforce_eager
):
    from irwkv.sampling_params import SamplingParams

    model = os.environ["IRWKV_RWKV_TEST_MODEL"]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=3,
        ignore_eos=True,
        presence_penalty=0,
        frequency_penalty=0,
        penalty_decay=1,
        detokenize=False,
        logprobs=3,
    )
    with irwkv_runner(
        model,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        async_scheduling=asynchronous,
        kv_transfer_config={
            "kv_connector": "RwkvStateConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"max_bytes": 256 * 1024**2},
        },
    ) as runner:
        core = runner.llm.llm_engine.engine_core
        tokens = list(range(1, 66))
        first = runner.llm.generate([{"prompt_token_ids": tokens}], sampling)[0]
        deadline = time.monotonic() + 5
        entries = core.call_utility("rwkv_state", "list", None)
        while not entries and time.monotonic() < deadline:
            time.sleep(0.01)
            entries = core.call_utility("rwkv_state", "list", None)
        assert entries, "Finished request did not produce a host snapshot"
        entry = entries[-1]
        continuation = tokens + list(first.outputs[0].token_ids) + [100, 101]
        assert (
            context_sha1(continuation[: entry["num_tokens"]]) == entry["context_sha1"]
        )
        data = core.call_utility("rwkv_state", "export", entry["id"])
        core.call_utility("rwkv_state", "clear", None)
        expected = runner.llm.generate([{"prompt_token_ids": continuation}], sampling)[
            0
        ]
        uploaded = core.call_utility("rwkv_state", "import", data)
        assert "error" not in uploaded
        constrained = sampling.clone()
        constrained.extra_args = {
            "kv_transfer_params": {"rwkv_state_id": uploaded["id"]}
        }
        actual = runner.llm.generate([{"prompt_token_ids": continuation}], constrained)[
            0
        ]
        assert actual.num_cached_tokens == entry["num_tokens"]
        assert actual.outputs[0].token_ids == expected.outputs[0].token_ids
        for actual_probs, expected_probs in zip(
            actual.outputs[0].logprobs, expected.outputs[0].logprobs
        ):
            assert actual_probs.keys() == expected_probs.keys()
            for token in actual_probs:
                assert actual_probs[token].logprob == pytest.approx(
                    expected_probs[token].logprob, abs=1e-4
                )
