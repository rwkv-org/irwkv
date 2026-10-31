# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-resident RWKV uploads and per-session recurrent-state snapshots."""

import hashlib
import io
import json
import math
import zipfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from irwkv import envs
from irwkv.model_executor.models.config import RwkvForCausalLMConfig
from irwkv.utils.cache import LRUCache


@dataclass
class RwkvSessionState:
    identity: tuple[Any, ...]
    token_ids: list[int]
    state: tuple[torch.Tensor, ...]
    ready: torch.cuda.Event | None = None
    token_buffer: torch.Tensor | None = None
    num_tokens: torch.Tensor | None = None

    def poll(self) -> bool:
        if self.ready is None:
            return True
        if not self.ready.query():
            return False
        assert self.num_tokens is not None and self.token_buffer is not None
        self.token_ids = self.token_buffer[: int(self.num_tokens.item())].tolist()
        self.ready = None
        self.token_buffer = self.num_tokens = None
        return True


@dataclass
class RwkvRuntimeRequest:
    session_id: str | None
    identity: tuple[Any, ...]
    adapter_key: str
    adapter: dict[str, torch.Tensor] | None
    state: tuple[torch.Tensor, ...] | None
    initial_state: tuple[torch.Tensor, ...] | None
    num_cached_tokens: int = 0
    saved: bool = False


class RwkvRuntimeCache:
    """Host-resident uploads and one independent snapshot per session."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.states: dict[str, tuple[torch.Tensor, ...]] = {}
        self.adapters: LRUCache[str, tuple[str, dict[str, torch.Tensor]]] = LRUCache(16)
        self.sessions: OrderedDict[str, RwkvSessionState] = OrderedDict()
        self.requests: dict[str, RwkvRuntimeRequest] = {}
        self.capacity = max(0, envs.IRWKV_RWKV_SESSION_CACHE_SIZE)
        self.upload_capacity = max(0, envs.IRWKV_RWKV_UPLOAD_MAX_BYTES)
        self.hits = 0
        self.misses = 0
        self.uploads = 0
        self._active_adapter = ""
        self._base_adapter: dict[str, torch.Tensor] | None = None

    def _check_upload_enabled(self) -> None:
        if not envs.IRWKV_ALLOW_RUNTIME_RWKV_UPDATING:
            raise ValueError(
                "runtime uploads require IRWKV_ALLOW_RUNTIME_RWKV_UPDATING=1"
            )

    @staticmethod
    def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
        return sum(
            {
                t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in tensors
            }.values()
        )

    def _upload_tensors(
        self, *, state_id: str | None = None, adapter_ids: Sequence[str] = ()
    ) -> list[torch.Tensor]:
        states = {
            id(state): state for key, state in self.states.items() if key != state_id
        }
        adapters = {
            id(adapter): adapter
            for key, (_, adapter) in self.adapters.cache.items()
            if key not in adapter_ids
        }
        for request in self.requests.values():
            if request.initial_state is not None:
                states[id(request.initial_state)] = request.initial_state
            if request.adapter is not None:
                adapters[id(request.adapter)] = request.adapter
        return [t for state in states.values() for t in state] + [
            t for adapter in adapters.values() for t in adapter.values()
        ]

    def _reserve_upload(
        self, key: str, tensors: Sequence[torch.Tensor], *, adapter: bool, publish: bool
    ) -> None:
        removed = [key] if adapter else []
        remaining = [k for k in self.adapters.order if k not in removed]
        while (
            len(remaining) + int(adapter) > 16
            or self._tensor_bytes(
                self._upload_tensors(
                    state_id=None if adapter else key, adapter_ids=removed
                )
                + list(tensors)
            )
            > self.upload_capacity
        ):
            if not remaining:
                raise ValueError(
                    "runtime upload capacity exceeded; delete unused uploads"
                )
            removed.append(remaining.pop(0))
        if publish:
            for adapter_id in removed:
                self.adapters.pop(adapter_id, None)

    @staticmethod
    def _load(payload: bytes) -> dict[str, torch.Tensor]:
        try:
            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    if sum(entry.file_size for entry in archive.infolist()) > 512 << 20:
                        raise ValueError("expanded tensor checkpoint exceeds 512 MiB")
            if payload[:2] == b"PK" or payload[:1] == b"\x80":
                values = torch.load(
                    io.BytesIO(payload), map_location="cpu", weights_only=True
                )
            else:
                from safetensors.torch import load

                values = load(payload)
        except Exception as error:
            raise ValueError(f"invalid tensor checkpoint: {error}") from error
        if isinstance(values, dict) and "state_dict" in values:
            values = values["state_dict"]
        if not isinstance(values, dict) or not values:
            raise ValueError("upload must contain a nonempty tensor state_dict")
        if any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor)
            for k, v in values.items()
        ):
            raise ValueError("upload state_dict must contain only tensors")
        if any(not torch.isfinite(v).all() for v in values.values()):
            raise ValueError("upload contains nonfinite tensors")
        return values

    def upload_state(self, payload: bytes, *, publish: bool = True) -> dict[str, Any]:
        self._check_upload_enabled()
        values = self._load(payload)
        config = self.model.config
        expected = {
            f"blocks.{i}.att.time_state" for i in range(config.num_hidden_layers)
        }
        if set(values) != expected:
            raise ValueError(
                "state must contain blocks.N.att.time_state for every layer"
            )
        shape = (config.num_attention_heads, config.head_size, config.head_size)
        if any(v.shape != shape or not v.is_floating_point() for v in values.values()):
            raise ValueError("uploaded state shape or dtype does not match the model")
        state_layer = self.model.get_rwkv_state_layer()
        start, end = self.model.model.start_layer, self.model.model.end_layer
        wkv = (
            torch.stack(
                [
                    values[f"blocks.{i}.att.time_state"].transpose(-1, -2)
                    for i in range(start, end)
                ]
            )
            .to(dtype=state_layer.state_dtype)
            .contiguous()
        )
        shift = torch.zeros(end - start, config.hidden_size, dtype=torch.bfloat16)
        state_id = "state-" + hashlib.sha256(payload).hexdigest()
        if not torch.isfinite(wkv).all():
            raise ValueError("state overflows the configured recurrent-state dtype")
        state = (shift, shift.clone(), wkv)
        if torch.cuda.is_available():
            state = tuple(t.pin_memory() for t in state)
        self._reserve_upload(state_id, state, adapter=False, publish=publish)
        if publish:
            self.states[state_id] = state
        return {
            "state_id": state_id,
            "object": "rwkv.state",
            "size_bytes": len(payload),
        }

    def upload_adapter(
        self,
        adapter_id: str,
        payload: bytes,
        *,
        adapter_config: dict[str, Any] | None = None,
        publish: bool = True,
    ) -> dict[str, Any]:
        self._check_upload_enabled()
        if not adapter_id:
            raise ValueError("missing adapter_id")
        if getattr(self.model.config, "rwkv_lora", None) is None:
            raise ValueError(
                "start the model with rwkv_lora to upload LoRA/NoRA adapters"
            )
        raw_values = self._load(payload)
        values = self._load_lightning_adapter(payload, raw_values)
        metadata = b""
        if values is raw_values:
            self._validate_adapter_config(adapter_config)
            metadata = json.dumps(adapter_config, sort_keys=True).encode()
        normalized = {
            name.removeprefix("base_model.model.").replace(
                ".default.weight", ".weight"
            ): v
            for name, v in values.items()
        }
        if len(normalized) != len(values):
            raise ValueError("duplicate normalized adapter tensor names")
        values = normalized
        parameters = {
            name: p for name, p in self.model.named_parameters() if ".lora_" in name
        }
        local = {}
        for name, parameter in parameters.items():
            value = values.get(name)
            if (
                value is None
                or value.shape != parameter.shape
                or not value.is_floating_point()
                or value.layout != torch.strided
            ):
                raise ValueError(f"missing or incompatible adapter tensor: {name}")
            local[name] = value.to(dtype=torch.float32, copy=True).contiguous()
            if parameter.is_cuda:
                local[name] = local[name].pin_memory()
        valid_names = set(parameters)
        # Other pipeline stages own the remaining adapter tensors.
        for name in values:
            if name in valid_names:
                continue
            if name.startswith("model.layers.") and ".lora_" in name:
                layer = int(name.split(".")[2])
                if (
                    0 <= layer < self.model.config.num_hidden_layers
                    and not self.model.model.start_layer
                    <= layer
                    < self.model.model.end_layer
                ):
                    continue
            if (
                name.startswith("lm_head.lora_")
                and name not in parameters
                and not hasattr(self.model.lm_head, "weight")
            ):
                continue
            raise ValueError(f"unexpected adapter tensor: {name}")
        if any(not torch.isfinite(v).all() for v in local.values()):
            raise ValueError("adapter overflows float32 storage")
        version = hashlib.sha256(payload + metadata).hexdigest()
        self._reserve_upload(
            adapter_id, list(local.values()), adapter=True, publish=publish
        )
        if publish:
            self.adapters[adapter_id] = (version, local)
        return {"adapter_id": adapter_id, "version": version}

    def _validate_adapter_config(self, metadata: dict[str, Any] | None) -> None:
        if not isinstance(metadata, dict):
            raise ValueError(
                "PEFT uploads require adapter_config.json including use_nora"
            )
        config = self.model.config.rwkv_lora
        selected: dict[str, Any] = {
            key: metadata.get(key) for key in RwkvForCausalLMConfig.LORA_KEYS
        }
        RwkvForCausalLMConfig.verify_lora_config(selected)
        if any(selected[key] != config[key] for key in ("r", "lora_alpha", "use_nora")):
            raise ValueError("PEFT rank, lora_alpha and use_nora must match rwkv_lora")
        if set(selected["target_modules"]) != set(config["target_modules"]):
            raise ValueError("PEFT target_modules must match rwkv_lora")
        if (
            metadata.get("peft_type", "LORA") != "LORA"
            or metadata.get("bias", "none") != "none"
            or any(
                metadata.get(key)
                for key in (
                    "use_rslora",
                    "use_dora",
                    "fan_in_fan_out",
                    "rank_pattern",
                    "alpha_pattern",
                    "modules_to_save",
                    "lora_bias",
                    "layer_replication",
                    "layers_to_transform",
                    "layers_pattern",
                )
            )
        ):
            raise ValueError(
                "unsupported PEFT scaling, layout, bias or module overrides"
            )
        expected_base = getattr(self.model.config, "rwkv_base_fingerprint", None)
        if (
            expected_base is not None
            and metadata.get("base_fingerprint") != expected_base
        ):
            raise ValueError("PEFT base_fingerprint does not match the model")

    def _load_lightning_adapter(
        self, payload: bytes, values: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Validate and map Lightning's initialization-only NoRA factors."""
        source = io.BytesIO(payload)
        if not zipfile.is_zipfile(source):
            return values
        with zipfile.ZipFile(source) as archive:
            if "archive/nora.json" not in archive.namelist():
                return values
            if archive.getinfo("archive/nora.json").file_size > 8 << 20:
                raise ValueError("Lightning NoRA manifest exceeds 8 MiB")
            manifest = json.loads(archive.read("archive/nora.json"))
        config = self.model.config.rwkv_lora
        if (
            not isinstance(manifest, dict)
            or manifest.get("format_version") != 1
            or manifest.get("method") != "nora"
            or manifest.get("layout") != "low_rank_ba"
            or manifest.get("kind") != "inference_adapter"
            or manifest.get("initialization") != "nora_init"
            or manifest.get("dtype") != "bfloat16"
        ):
            raise ValueError("unsupported Lightning NoRA format")
        if config["use_nora"]:
            raise ValueError(
                "Lightning NoRA normalizes A only at initialization; configure "
                "rwkv_lora.use_nora=false to preserve its raw A/B factors"
            )
        rank = manifest.get("rank")
        if (
            rank != config["r"]
            or manifest.get("alpha") != rank
            or manifest.get("scale") != 1.0
            or config["lora_alpha"] != rank
        ):
            raise ValueError(
                "Lightning NoRA rank/alpha must match rwkv_lora with alpha/rank=1"
            )
        version = manifest.pop("content_version", None)
        digest = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        modules = {
            "att.receptance.weight": "linear_attn.r_proj",
            "att.key.weight": "linear_attn.k_proj",
            "att.value.weight": "linear_attn.v_proj",
            "att.output.weight": "linear_attn.o_proj",
            "ffn.key.weight": "mlp.key",
            "ffn.value.weight": "mlp.value",
        }
        converted: dict[str, torch.Tensor] = {}
        seen: set[str] = set()
        targets = manifest.get("targets")
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(target, dict) for target in targets)
        ):
            raise ValueError("Lightning NoRA requires a nonempty target list")
        for target in targets:
            layer = target["layer"]
            name = target["target"]
            if (
                name not in modules
                or not 0 <= layer < self.model.config.num_hidden_layers
            ):
                raise ValueError("unknown Lightning NoRA target or layer")
            for factor, key, shape_key in (
                ("B", "name", "shape"),
                ("A", "a_name", "a_shape"),
            ):
                source_name = f"blocks.{layer}.{name}.{factor}"
                value = values.get(source_name)
                if (
                    target.get(key) != source_name
                    or source_name in seen
                    or value is None
                    or value.dtype != torch.bfloat16
                    or list(value.shape) != target.get(shape_key)
                    or value.ndim != 2
                    or value.shape[0 if factor == "A" else 1] != rank
                    or (factor == "A" and value.shape[1] != target["in_features"])
                ):
                    raise ValueError(f"invalid Lightning NoRA tensor: {source_name}")
                seen.add(source_name)
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
                converted[
                    f"model.layers.{layer}.{modules[name]}.lora_{factor}.weight"
                ] = value
        if seen != set(values) or digest.hexdigest() != version:
            raise ValueError(
                "Lightning NoRA tensor set or content digest "
                "does not match its manifest"
            )
        expected_base = getattr(self.model.config, "rwkv_base_fingerprint", None)
        if expected_base is not None and expected_base != manifest.get(
            "base_fingerprint"
        ):
            raise ValueError("Lightning NoRA base_fingerprint does not match the model")
        return converted

    def prepare(self, data: dict[str, Any]) -> dict[str, Any]:
        options = data.get("options") or {}
        adapter_id = options.get("adapter_id")
        adapter = None
        adapter_key = ""
        if adapter_id:
            if adapter_id not in self.adapters:
                raise ValueError(f"unknown adapter_id: {adapter_id}")
            version, adapter = self.adapters[adapter_id]
            if options.get("adapter_version", version) != version:
                raise ValueError("adapter_version does not match the uploaded adapter")
            scale = options.get("adapter_scale", 1.0)
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(scale)
            ):
                raise ValueError("adapter_scale must be a finite number")
            adapter_key = f"{version}:{float(scale).hex()}"
            if scale != 1:
                adapter = {
                    name: (
                        (value * scale).pin_memory()
                        if value.is_pinned()
                        else value * scale
                    )
                    if ".lora_B." in name
                    else value
                    for name, value in adapter.items()
                }
                if any(not torch.isfinite(value).all() for value in adapter.values()):
                    raise ValueError("adapter_scale overflows the adapter dtype")
        elif "adapter_version" in options or "adapter_scale" in options:
            raise ValueError("adapter_version and adapter_scale require adapter_id")
        state_id = options.get("state_id")
        if state_id and state_id not in self.states:
            raise ValueError(f"unknown state_id: {state_id}")
        initial_state = self.states.get(state_id) if state_id else None
        identity = (
            adapter_key,
            state_id,
            data.get("cache_salt"),
            data.get("weight_version"),
        )
        retained = self._upload_tensors() + list((adapter or {}).values())
        if self._tensor_bytes(retained) > self.upload_capacity:
            raise ValueError("runtime upload capacity exceeded by adapter_scale")
        session_id = data.get("session_id")
        state = initial_state
        cached_tokens = 0
        if session_id and self.capacity:
            snapshot = self.sessions.get(session_id)
            tokens = data["token_ids"]
            if (
                not data.get("skip_session", False)
                and snapshot is not None
                and snapshot.poll()
                and snapshot.identity == identity
                and 0 < len(snapshot.token_ids) < len(tokens)
                and tokens[: len(snapshot.token_ids)] == snapshot.token_ids
            ):
                state = snapshot.state
                cached_tokens = len(snapshot.token_ids)
                self.sessions.move_to_end(session_id)
                self.hits += 1
            else:
                self.misses += 1
            # Each admitted branch reads an immutable snapshot; the newest
            # admission alone may publish, independent of completion order.
            for previous in self.requests.values():
                if previous.session_id == session_id:
                    previous.session_id = None
        self.requests[data["request_id"]] = RwkvRuntimeRequest(
            session_id,
            identity,
            adapter_key,
            adapter,
            state,
            initial_state,
            cached_tokens,
        )
        return {"num_cached_tokens": cached_tokens, "adapter_key": adapter_key}

    def activate_adapter(self, request_ids: Sequence[str]) -> bool:
        if not self.requests and not self._active_adapter:
            return False
        requests = [self.requests.get(req_id) for req_id in request_ids]
        needs_state = any(
            request is not None and (request.session_id or request.state is not None)
            for request in requests
        )
        keys = {
            request.adapter_key if request is not None else "" for request in requests
        }
        if len(keys) != 1:
            raise RuntimeError("RWKV batch contains different adapter identities")
        key = keys.pop()
        if key == self._active_adapter:
            return needs_state
        parameters = {
            name: p for name, p in self.model.named_parameters() if ".lora_" in name
        }
        if self._base_adapter is None:
            self._base_adapter = {
                name: p.detach().clone() for name, p in parameters.items()
            }
        weights = requests[0].adapter if requests[0] is not None else None
        weights = weights if weights is not None else self._base_adapter
        for name, parameter in parameters.items():
            parameter.data.copy_(weights[name], non_blocking=True)
        self._active_adapter = key
        self.uploads += 1
        return needs_state

    def save_session(
        self,
        req_id: str,
        token_ids: list[int],
        state: tuple[torch.Tensor, ...],
        *,
        ready: torch.cuda.Event | None = None,
        token_buffer: torch.Tensor | None = None,
        num_tokens: torch.Tensor | None = None,
    ) -> None:
        request = self.requests[req_id]
        if (
            request.session_id
            and self.capacity
            and (token_ids or ready)
            and not request.saved
        ):
            self.sessions[request.session_id] = RwkvSessionState(
                request.identity, token_ids, state, ready, token_buffer, num_tokens
            )
            self.sessions.move_to_end(request.session_id)
            while len(self.sessions) > self.capacity:
                self.sessions.popitem(last=False)
        request.saved = True

    def manage(self, operation: str, data: dict[str, Any]) -> dict[str, Any]:
        if operation == "validate_upload_state":
            return self.upload_state(data["payload"], publish=False)
        if operation == "validate_upload_adapter":
            return self.upload_adapter(
                data["adapter_id"],
                data["payload"],
                adapter_config=data.get("adapter_config"),
                publish=False,
            )
        if operation == "upload_state":
            return self.upload_state(data["payload"])
        if operation == "upload_adapter":
            return self.upload_adapter(
                data["adapter_id"],
                data["payload"],
                adapter_config=data.get("adapter_config"),
            )
        if operation == "delete_state":
            return {"deleted": self.states.pop(data["state_id"], None) is not None}
        if operation == "delete_adapter":
            adapter = self.adapters.get(data["adapter_id"])
            if (
                adapter is not None
                and data.get("adapter_version", adapter[0]) != adapter[0]
            ):
                return {"deleted": False}
            return {"deleted": self.adapters.pop(data["adapter_id"], None) is not None}
        if operation == "delete_session":
            session_id = data["session_id"]
            for request in self.requests.values():
                if request.session_id == session_id:
                    request.session_id = None
            return {"deleted": self.sessions.pop(session_id, None) is not None}
        if operation == "clear":
            self.sessions.clear()
            for request in self.requests.values():
                request.state = request.initial_state
                request.num_cached_tokens = 0
                request.session_id = None
            if self._active_adapter and self._base_adapter is not None:
                for name, parameter in self.model.named_parameters():
                    if name in self._base_adapter:
                        parameter.data.copy_(self._base_adapter[name])
            self._active_adapter = ""
            self._base_adapter = None
            return {"cleared": True}
        if operation == "cancel":
            self.requests.pop(data["request_id"], None)
            return {"cancelled": True}
        if operation == "status":
            for snapshot in self.sessions.values():
                snapshot.poll()
            tensors = [tensor for state in self.states.values() for tensor in state]
            tensors.extend(
                tensor
                for _, adapter in self.adapters.values()
                for tensor in adapter.values()
            )
            tensors.extend(
                tensor
                for snapshot in self.sessions.values()
                for tensor in snapshot.state
            )
            tensors.extend(
                t
                for snapshot in self.sessions.values()
                for t in (snapshot.token_buffer, snapshot.num_tokens)
                if t is not None
            )
            for request in self.requests.values():
                tensors.extend(request.state or ())
                tensors.extend(request.initial_state or ())
                tensors.extend((request.adapter or {}).values())
            tensors.extend((self._base_adapter or {}).values())
            storages = {
                tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in tensors
                if tensor.device.type == "cpu"
            }
            return {
                "ram_bytes": sum(storages.values()),
                "states": [{"state_id": state_id} for state_id in self.states],
                "adapters": [
                    {"adapter_id": key, "version": version}
                    for key, (version, _) in self.adapters.items()
                ],
                "sessions": [
                    {"session_id": key, "cached_tokens": len(value.token_ids)}
                    for key, value in self.sessions.items()
                ],
                "session_cache_hits": self.hits,
                "session_cache_misses": self.misses,
                "session_cache_capacity": self.capacity,
                "upload_capacity_bytes": self.upload_capacity,
                "upload_bytes": self._tensor_bytes(self._upload_tensors()),
                "adapter_capacity": 16,
                "uploads": self.uploads,
            }
        raise ValueError(f"unknown RWKV cache operation: {operation}")
