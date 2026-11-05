# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV Lightning checkpoints and bounded, request-pinned host storage."""

import hashlib
import io
import json
import math
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

MAX_ADAPTERS = 16
MAX_UPLOAD_BYTES = 512 << 20
_TARGETS = {
    "att.receptance": "linear_attn.r_proj",
    "att.key": "linear_attn.k_proj",
    "att.value": "linear_attn.v_proj",
    "att.output": "linear_attn.o_proj",
    "ffn.key": "mlp.key",
    "ffn.value": "mlp.value",
}


@dataclass(frozen=True)
class RWKVAdapter:
    adapter_id: str
    version: str
    kind: str
    tensors: dict[str, torch.Tensor]
    rank: int = 0
    scale: float = 1.0

    def info(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "version": self.version,
            "kind": self.kind,
            "rank": self.rank,
            "scale": self.scale,
            "ram_bytes": sum(
                t.numel() * t.element_size() for t in self.tensors.values()
            ),
        }


def load_rwkv_adapter(
    adapter_id: str,
    data: bytes,
    config: Any,
    *,
    kind: str = "adapter",
    metadata: dict[str, Any] | None = None,
) -> RWKVAdapter:
    """Validate a checkpoint before it can replace a resident entry."""
    if not isinstance(adapter_id, str) or not adapter_id or len(adapter_id) > 256:
        raise ValueError("adapter_id must contain 1 to 256 characters")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("checkpoint must contain 1 byte to 512 MiB")
    if kind not in ("adapter", "state"):
        raise ValueError("kind must be adapter or state")
    metadata = dict(metadata or {})
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifests = [n for n in archive.namelist() if n.endswith("/nora.json")]
        if len(manifests) > 1:
            raise ValueError("ambiguous NoRA manifests")
        if manifests:
            if metadata or kind != "adapter":
                raise ValueError("embedded NoRA metadata cannot be overridden")
            if archive.getinfo(manifests[0]).file_size > 8 << 20:
                raise ValueError("NoRA manifest exceeds 8 MiB")
            metadata = json.loads(archive.read(manifests[0]))
            if not isinstance(metadata, dict):
                raise ValueError("NoRA manifest must be an object")
        if sum(i.file_size for i in archive.infolist()) > MAX_UPLOAD_BYTES:
            raise ValueError("unpacked checkpoint exceeds 512 MiB")
    weights = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("checkpoint must be a non-empty tensor dictionary")
    for name, tensor in weights.items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("checkpoint must contain only finite floating tensors")

    if kind == "state":
        names = [f"blocks.{i}.att.time_state" for i in range(config.num_hidden_layers)]
        shape = (config.num_attention_heads, config.head_size, config.head_size)
        if set(weights) != set(names) or any(weights[n].shape != shape for n in names):
            raise ValueError("state layers or head dimensions do not match the model")
        tensors = {
            "state": torch.stack([weights[n].transpose(-1, -2).float() for n in names])
        }
        rank, scale = 0, 1.0
    else:
        tensors = {}
        for name, tensor in weights.items():
            match = re.fullmatch(r"blocks\.(\d+)\.(.+)\.weight\.([AB])", name)
            if match and match[2] in _TARGETS:
                name = (
                    f"model.layers.{match[1]}.{_TARGETS[match[2]]}"
                    f".lora_{match[3]}.weight"
                )
            elif name in ("head.weight.A", "head.weight.B"):
                name = f"lm_head.lora_{name[-1]}.weight"
            else:
                name = name.removeprefix("base_model.model.")
                name = name.replace(".lora_A.default.", ".lora_A.")
                name = name.replace(".lora_B.default.", ".lora_B.")
            if name in tensors:
                raise ValueError(f"duplicate adapter tensor: {name}")
            tensors[name] = tensor.detach().float().contiguous()
        prefixes = {name.rsplit(".lora_", 1)[0] for name in tensors}
        ranks = set()
        for prefix in prefixes:
            a = tensors.get(f"{prefix}.lora_A.weight")
            b = tensors.get(f"{prefix}.lora_B.weight")
            match = re.fullmatch(r"model.layers\.(\d+)\.(.+)", prefix)
            if prefix == "lm_head":
                out_features, in_features = config.vocab_size, config.hidden_size
            elif (
                match
                and int(match[1]) < config.num_hidden_layers
                and match[2] in _TARGETS.values()
            ):
                in_features = (
                    config.intermediate_size
                    if match[2] == "mlp.value"
                    else config.hidden_size
                )
                out_features = (
                    config.intermediate_size
                    if match[2] == "mlp.key"
                    else config.hidden_size
                )
            else:
                raise ValueError(f"unsupported adapter projection: {prefix}")
            if (
                a is None
                or b is None
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[0] <= 0
                or a.shape[1] != in_features
                or b.shape != (out_features, a.shape[0])
            ):
                raise ValueError(f"invalid A/B pair for {prefix}")
            ranks.add(a.shape[0])
        if len(tensors) != 2 * len(prefixes) or len(ranks) != 1:
            raise ValueError("adapter requires complete A/B pairs of a single rank")
        rank = ranks.pop()
        if manifests:
            if (
                metadata.get("method") != "nora"
                or metadata.get("format_version") != 1
                or metadata.get("kind") != "inference_adapter"
                or metadata.get("layout") != "low_rank_ba"
                or metadata.get("initialization") != "nora_init"
                or metadata.get("rank") != rank
                or metadata.get("alpha") != rank
                or metadata.get("scale") != 1
            ):
                raise ValueError("unsupported Lightning NoRA manifest")
            # Lightning normalizes at initialization, not during inference.
            kind, scale = "nora", 1.0
        else:
            alpha = metadata.get("lora_alpha")
            if (
                isinstance(alpha, bool)
                or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha)
                or metadata.get("r") != rank
                or metadata.get("use_nora", False) is not False
            ):
                raise ValueError(
                    "LoRA requires r, finite lora_alpha and use_nora=false"
                )
            kind, scale = "lora", alpha / rank
    digest = hashlib.sha256(data)
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    return RWKVAdapter(adapter_id, digest.hexdigest(), kind, tensors, rank, scale)


class RWKVAdapterStore:
    """LRU storage; active requests pin entries until they finish or abort."""

    def __init__(self) -> None:
        self.entries: OrderedDict[str, RWKVAdapter] = OrderedDict()
        self.references: dict[str, int] = {}
        self.hits = 0
        self.evictions = 0

    def put(self, entry: RWKVAdapter) -> RWKVAdapter:
        old = self.entries.get(entry.adapter_id)
        if old is not None and old.version == entry.version:
            return self.get(entry.adapter_id)
        if self.references.get(entry.adapter_id, 0):
            raise ValueError("cannot replace an adapter used by an active request")
        if old is None and len(self.entries) >= MAX_ADAPTERS:
            victim = next(
                (key for key in self.entries if not self.references.get(key, 0)), None
            )
            if victim is None:
                raise ValueError("all 16 cached adapters are in use")
            del self.entries[victim]
            self.evictions += 1
        self.entries[entry.adapter_id] = entry
        self.entries.move_to_end(entry.adapter_id)
        return entry

    def get(self, adapter_id: str, version: str | None = None) -> RWKVAdapter:
        entry = self.entries.get(adapter_id)
        if entry is None or (version is not None and entry.version != version):
            raise ValueError(f"adapter not found: {adapter_id}")
        self.entries.move_to_end(adapter_id)
        self.hits += 1
        return entry

    def acquire(self, adapter_id: str, version: str | None = None) -> RWKVAdapter:
        entry = self.get(adapter_id, version)
        self.references[adapter_id] = self.references.get(adapter_id, 0) + 1
        return entry

    def release(self, adapter_id: str) -> None:
        count = self.references[adapter_id]
        if count == 1:
            del self.references[adapter_id]
        else:
            self.references[adapter_id] = count - 1

    def erase(self, adapter_id: str) -> bool:
        if self.references.get(adapter_id, 0):
            raise ValueError("cannot delete an adapter used by an active request")
        return self.entries.pop(adapter_id, None) is not None

    def info(self) -> dict[str, Any]:
        return {
            "data": [entry.info() for entry in self.entries.values()],
            "capacity": MAX_ADAPTERS,
            "eviction_policy": "lru",
            "ram_hits": self.hits,
            "evictions": self.evictions,
            "ram_bytes": sum(
                entry.info()["ram_bytes"] for entry in self.entries.values()
            ),
        }
