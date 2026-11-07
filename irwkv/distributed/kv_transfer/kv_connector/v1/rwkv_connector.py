# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host prefix snapshots for RWKV's single mutable state per request."""

import hashlib
from dataclasses import dataclass, field

import torch

from irwkv.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from irwkv.v1.metrics.cache_hit_source import CacheHitSource
from irwkv.v1.rwkv_state import (
    RwkvState,
    RwkvStateCache,
    context_prefix_sha1,
    model_fingerprint,
)


@dataclass
class RwkvConnectorMetadata(KVConnectorMetadata):
    loads: dict[str, tuple[str, int]] = field(default_factory=dict)
    saves: dict[str, tuple[int, tuple[int, ...], str | None]] = field(
        default_factory=dict
    )
    cursors: dict[str, int] = field(default_factory=dict)
    removed: set[str] = field(default_factory=set)
    reset: bool = False


@dataclass
class RwkvCacheIndex(KVConnectorWorkerMetadata):
    generation: int
    entries: list[dict]

    def aggregate(self, other):
        raise ValueError("RWKV host state caching currently requires one GPU worker")


class RwkvStateConnector(KVConnectorBase_V1, SupportsHMA):
    """Reuse exact consumed prefixes without token-position state block tables."""

    _cache_hit_source = CacheHitSource.HOST

    def __init__(self, irwkv_config, role, kv_cache_config):
        super().__init__(irwkv_config, role, kv_cache_config)
        self._requests = {}
        self._index = RwkvCacheIndex(-1, [])
        self._loads = {}
        self._saves = {}
        self._matches = {}
        self._cursors = {}
        self._automatic = {}
        self._pending_saves = {}
        self._pending_contexts = {}
        self._pending_loads = {}
        self._failed_loads = set()
        self._invalid_blocks = set()
        config = irwkv_config.kv_transfer_config.kv_connector_extra_config
        self._cache = RwkvStateCache(int(config.get("max_bytes", 4 * 1024**3)))
        self._layer = None
        self._model_id = None
        self._copy_stream = None
        self._last_index_generation = -1
        self._reset_pending = False
        self._discard_saves = set()
        self._base_model_id = None
        if role == KVConnectorRole.WORKER:
            self._model_id = model_fingerprint(
                irwkv_config.model_config,
                irwkv_config.load_config.download_dir,
            )
            self._base_model_id = self._model_id

    @property
    def requires_kv_delivery(self):
        return False

    def register_kv_caches(self, kv_caches):
        context = self._irwkv_config.compilation_config.static_forward_context
        layers = [context[name] for name in kv_caches]
        if len(layers) != 1 or not hasattr(layers[0], "snapshot_state_slot"):
            raise ValueError("RWKV host cache expects one local recurrent state pool")
        self._layer = layers[0]
        self._copy_stream = torch.cuda.Stream()

    def on_new_request(self, request):
        self._requests[request.request_id] = request

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        tokens = request.prompt_token_ids
        if tokens is None or request.prompt_embeds is not None:
            return 0, False
        hint = (request.kv_transfer_params or {}).get("rwkv_state_id")
        candidates = [
            entry
            for entry in self._index.entries
            if num_computed_tokens < entry["num_tokens"] < len(tokens)
            and entry["cache_salt"] == request.cache_salt
            and (
                entry["id"] == hint
                if hint is not None
                else entry.get("automatic", True)
            )
        ]
        hashes = context_prefix_sha1(tokens, (e["num_tokens"] for e in candidates))
        for entry in sorted(candidates, key=lambda e: e["num_tokens"], reverse=True):
            length = entry["num_tokens"]
            if hashes[length] == entry["context_sha1"]:
                self._matches[request.request_id] = entry["id"]
                return length - num_computed_tokens, True
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            state_id = self._matches.pop(request.request_id)
            self._loads[request.request_id] = (state_id, blocks.get_block_ids()[0][-1])

    def build_connector_meta(self, scheduler_output):
        cursors = {
            req_id: self._requests[req_id].num_computed_tokens + count
            for req_id, count in scheduler_output.num_scheduled_tokens.items()
        }
        metadata = RwkvConnectorMetadata(
            self._loads,
            self._saves,
            cursors,
            scheduler_output.finished_req_ids
            | (scheduler_output.preempted_req_ids or set()),
        )
        metadata.reset, self._reset_pending = self._reset_pending, False
        self._loads, self._saves = {}, {}
        return metadata

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids[0])

    def request_finished(self, request, block_ids):
        self._requests.pop(request.request_id, None)
        self._matches.pop(request.request_id, None)
        if not block_ids or request.prompt_embeds is not None:
            return False, None
        self._saves[request.request_id] = (
            block_ids[-1],
            tuple(request.all_token_ids),
            request.cache_salt,
        )
        return True, None

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            self.update_index(metadata)

    def update_index(self, index):
        if index is not None and index.generation >= self._index.generation:
            self._index = index

    def build_connector_worker_meta(self):
        if self._last_index_generation == self._cache.generation:
            return None
        self._last_index_generation = self._cache.generation
        return RwkvCacheIndex(self._cache.generation, self._cache.metadata())

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if metadata.reset:
            self.state_operation("clear")
        # Finished requests retain their source blocks until the D2H event
        # completes. A new forward can therefore run on other slots meanwhile.
        for req_id, (slot, tokens, salt) in metadata.saves.items():
            count = self._cursors.pop(req_id, 0)
            automatic = self._automatic.pop(req_id, True)
            if not 0 < count <= len(tokens):
                self._pending_saves[req_id] = (None, None)
                continue
            tokens = tokens[:count]
            context_key = self._model_id, salt, tokens
            if automatic and (
                self._cache.find(self._model_id, tokens, salt) is not None
                or context_key in self._pending_contexts
            ):
                self._pending_saves[req_id] = (None, None)
                continue
            state_size = self._layer.state_slot_nbytes
            reserve = state_size + 36 * count + 64
            pending = sum(
                s.nbytes for _, s in self._pending_saves.values() if s is not None
            )
            if reserve + pending > self._cache.max_bytes:
                self._pending_saves[req_id] = (None, None)
                continue
            while self._cache.nbytes + pending + reserve > self._cache.max_bytes:
                self._cache.remove(next(iter(self._cache.entries)))
            self._copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._copy_stream):
                tensors = self._layer.snapshot_state_slot(slot)
                event = torch.cuda.Event()
                event.record()
            state = RwkvState(
                self._model_id,
                tokens,
                tensors,
                salt,
                automatic=automatic,
                request_id=req_id,
            )
            self._pending_saves[req_id] = (event, state)
            if automatic:
                self._pending_contexts[state.context_key] = req_id
        for req_id, (state_id, slot) in metadata.loads.items():
            try:
                state = self._cache.get(state_id)
            except KeyError:
                self._failed_loads.add(req_id)
                self._invalid_blocks.add(slot)
                continue
            self._layer.restore_state_slot(slot, state.tensors)
            self._automatic[req_id] = state.automatic
            event = torch.cuda.Event()
            event.record()
            self._pending_loads[req_id] = (event, state)
        for req_id in metadata.removed:
            self._cursors.pop(req_id, None)
            self._automatic.pop(req_id, None)
        self._cursors.update(metadata.cursors)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_transfer_results(self, finished_req_ids):
        result = KVConnectorTransferResults()
        for req_id, (event, state) in list(self._pending_saves.items()):
            if event is not None and not event.query():
                continue
            if state is not None and req_id not in self._discard_saves:
                self._cache.add(state)
            if (
                state is not None
                and self._pending_contexts.get(state.context_key) == req_id
            ):
                del self._pending_contexts[state.context_key]
            self._discard_saves.discard(req_id)
            del self._pending_saves[req_id]
            result.finished_sending.add(req_id)
        for req_id, (event, _) in list(self._pending_loads.items()):
            if event.query():
                result.finished_recving.add(req_id)
                del self._pending_loads[req_id]
        result.failed_recving = self._failed_loads
        result.finished_recving.update(self._failed_loads)
        self._failed_loads = set()
        return result

    def get_block_ids_with_load_errors(self):
        blocks, self._invalid_blocks = self._invalid_blocks, set()
        return blocks

    def state_operation(self, operation, value=None):
        if operation == "list":
            return self._cache.metadata()
        if operation == "export":
            return self._cache.get(value).to_safetensors()
        if operation in ("clear", "version"):
            if operation == "version":
                self._model_id = hashlib.sha256(
                    f"{self._base_model_id}:{value}".encode()
                ).hexdigest()
            for state_id in list(self._cache.entries):
                self._cache.remove(state_id)
            self._discard_saves.update(self._pending_saves)
            self._pending_contexts.clear()
        elif operation == "delete":
            self._cache.remove(value)
        elif operation == "import":
            state = RwkvState.from_bytes(
                value,
                model_id=self._model_id,
                shapes=self._layer.state_slot_shapes,
                max_bytes=self._layer.state_slot_nbytes
                + self._irwkv_config.model_config.max_model_len * 4
                + 65536,
                max_tokens=self._irwkv_config.model_config.max_model_len,
            )
            vocab_size = self._irwkv_config.model_config.get_vocab_size()
            if any(token >= vocab_size for token in state.tokens):
                raise ValueError("RWKV state contains an out-of-vocabulary token")
            state_id = self._cache.add(state)
            return state.metadata(state_id)
        else:
            raise ValueError(f"Unknown RWKV state operation {operation!r}")

    def reset_cache(self):
        self._index = RwkvCacheIndex(self._index.generation, [])
        self._reset_pending = True
        return True

    def shutdown(self):
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
