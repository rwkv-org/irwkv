# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, bounded RWKV prefix snapshots and their portable binary format."""

import hashlib
import json
import struct
import sys
import uuid
import zlib
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path

import torch
from safetensors.torch import load, save

_MAGIC = b"RWKVST1\x00"
_HEADER = struct.Struct("<8sQ32s")


def compress_state(raw: bytes) -> bytes:
    return _HEADER.pack(_MAGIC, len(raw), hashlib.sha256(raw).digest()) + zlib.compress(
        raw, level=1
    )


def context_sha1(tokens) -> str:
    """Hash token IDs as unsigned little-endian 32-bit integers."""
    return context_prefix_sha1(tokens, [len(tokens)])[len(tokens)]


def context_prefix_sha1(tokens, lengths) -> dict[int, str]:
    """Hash candidate boundaries in one pass over the longest prefix."""
    boundaries = sorted(set(lengths))
    if not boundaries:
        return {}
    values = array("I", islice(tokens, boundaries[-1]))
    if sys.byteorder != "little":
        values.byteswap()
    data = memoryview(values).cast("B")
    digest = hashlib.sha1(usedforsecurity=False)
    result = {}
    offset = 0
    for length in boundaries:
        end = length * 4
        digest.update(data[offset:end])
        result[length] = digest.hexdigest()
        offset = end
    return result


@dataclass(frozen=True)
class RwkvState:
    model_id: str
    tokens: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    cache_salt: str | None = None
    automatic: bool = True
    request_id: str | None = None

    @cached_property
    def nbytes(self) -> int:
        return (
            sum(t.numel() * t.element_size() for t in self.tensors.values())
            + sys.getsizeof(self.tokens)
            + len(self.tokens) * sys.getsizeof(0)
        )

    @cached_property
    def context_sha1(self) -> str:
        return context_sha1(self.tokens)

    @cached_property
    def context_key(self) -> tuple[str, str | None, tuple[int, ...]]:
        return self.model_id, self.cache_salt, self.tokens

    def metadata(self, state_id: str) -> dict:
        return {
            "id": state_id,
            "model_id": self.model_id,
            "num_tokens": len(self.tokens),
            "context_sha1": self.context_sha1,
            "cache_salt": self.cache_salt,
            "size_bytes": self.nbytes,
            "automatic": self.automatic,
            "request_id": self.request_id,
        }

    def to_bytes(self) -> bytes:
        return compress_state(self.to_safetensors())

    def to_safetensors(self) -> bytes:
        metadata = {
            "version": "1",
            "model_id": self.model_id,
            "context_sha1": self.context_sha1,
            "cache_salt": json.dumps(self.cache_salt),
        }
        tensors = self.tensors | {
            "context_tokens": torch.tensor(self.tokens, dtype=torch.int32)
        }
        return save(tensors, metadata=metadata)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        model_id: str,
        shapes: dict,
        max_bytes: int,
        max_tokens: int,
    ):
        if len(data) < _HEADER.size or len(data) > max_bytes + 1024 * 1024:
            raise ValueError("Invalid RWKV state file size")
        magic, size, digest = _HEADER.unpack_from(data)
        if magic != _MAGIC or size > max_bytes or size < 8:
            raise ValueError("Invalid RWKV state header or decompressed size")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(data[_HEADER.size :], size + 1)
        except zlib.error as error:
            raise ValueError("Invalid compressed RWKV state") from error
        if (
            len(raw) != size
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ValueError("RWKV state length does not match its header")
        if hashlib.sha256(raw).digest() != digest:
            raise ValueError("RWKV state checksum mismatch")
        header_size = struct.unpack_from("<Q", raw)[0]
        if header_size > 65536 or header_size + 8 > len(raw):
            raise ValueError("Invalid safetensors header")
        header = json.loads(raw[8 : 8 + header_size])
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors metadata")
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            raise ValueError("Invalid safetensors metadata")
        if metadata.get("version") != "1" or metadata.get("model_id") != model_id:
            raise ValueError(
                "RWKV state belongs to a different model or format version"
            )
        try:
            tensors = load(raw)
        except Exception as error:
            raise ValueError("Invalid RWKV state tensors") from error
        tokens = tensors.pop("context_tokens", None)
        if (
            tokens is None
            or tokens.dtype != torch.int32
            or tokens.ndim != 1
            or not 0 < len(tokens) <= max_tokens
        ):
            raise ValueError("Invalid RWKV state context tokens")
        values = tuple(tokens.tolist())
        if any(t < 0 for t in values) or context_sha1(values) != metadata.get(
            "context_sha1"
        ):
            raise ValueError("RWKV state context hash mismatch")
        if tensors.keys() != shapes.keys():
            raise ValueError("RWKV state tensor names do not match the model")
        for name, tensor in tensors.items():
            shape, dtype = shapes[name]
            if (
                tuple(tensor.shape) != tuple(shape)
                or tensor.dtype != dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid RWKV state tensor {name}")
        salt = json.loads(metadata.get("cache_salt", "null"))
        if salt is not None and not isinstance(salt, str):
            raise ValueError("Invalid RWKV state cache salt")
        return cls(model_id, values, tensors, salt, automatic=False)


class RwkvStateCache:
    """Keep immutable CPU snapshots under a byte budget; compress only for export."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("RWKV state cache size must be positive")
        self.max_bytes = max_bytes
        self.nbytes = 0
        self.generation = 0
        self.entries: OrderedDict[str, RwkvState] = OrderedDict()
        self._contexts: dict[tuple[str, str | None, tuple[int, ...]], str] = {}

    def find(
        self, model_id: str, tokens: tuple[int, ...], cache_salt: str | None
    ) -> str | None:
        """Find an automatic snapshot by its exact context and refresh its LRU."""
        state_id = self._contexts.get((model_id, cache_salt, tokens))
        if state_id is not None:
            self.entries.move_to_end(state_id)
        return state_id

    def add(self, state: RwkvState) -> str:
        if state.nbytes > self.max_bytes:
            raise ValueError("RWKV state exceeds the host cache budget")
        if state.automatic:
            existing = self.find(state.model_id, state.tokens, state.cache_salt)
            if existing is not None:
                return existing
        while self.nbytes + state.nbytes > self.max_bytes:
            self.remove(next(iter(self.entries)))
        state_id = uuid.uuid4().hex
        self.entries[state_id] = state
        if state.automatic:
            self._contexts[state.context_key] = state_id
        self.nbytes += state.nbytes
        self.generation += 1
        return state_id

    def get(self, state_id: str) -> RwkvState:
        state = self.entries[state_id]
        self.entries.move_to_end(state_id)
        return state

    def remove(self, state_id: str) -> None:
        state = self.entries.pop(state_id)
        if state.automatic:
            del self._contexts[state.context_key]
        self.nbytes -= state.nbytes
        self.generation += 1

    def metadata(self) -> list[dict]:
        return [state.metadata(state_id) for state_id, state in self.entries.items()]


def model_fingerprint(model_config, cache_dir: str | None = None) -> str:
    """Identify an artifact by its configuration, tokenizer, and actual weights."""
    folder = Path(model_config.model)
    if not folder.is_dir():
        from huggingface_hub import snapshot_download

        folder = Path(
            snapshot_download(
                model_config.model,
                revision=model_config.revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
    weights = sorted(folder.glob("*.safetensors"))
    if not weights:
        raise ValueError("RWKV state caching requires a cached safetensors artifact")
    from irwkv.tokenizers.rwkv import RWKV_VOCAB_PATH

    digest = hashlib.sha256()
    config = model_config.hf_config.to_dict()
    config.pop("_name_or_path", None)
    digest.update(json.dumps(config, sort_keys=True).encode())
    for path in [RWKV_VOCAB_PATH, *weights]:
        digest.update(path.name.encode())
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
