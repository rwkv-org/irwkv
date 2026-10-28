# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

from pydantic import field_validator

from irwkv.config.utils import config
from irwkv.v1.attention.backends.registry import AttentionBackendEnum


@config
class AttentionConfig:
    """Configuration for attention mechanisms in irwkv."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    backend_per_kind: dict[str, AttentionBackendEnum] = field(default_factory=dict)
    """Per-KV-cache-group attention backend overrides, keyed by
    `KVCacheSpecKind` (e.g. `{"mla_attention": "FLASHINFER_MLA",
    "sliding_window_mla": "TRITON_MLA"}`). This lets a model that splits its
    layers across multiple KV-cache groups (e.g. interleaved full and
    sliding-window attention) use a different backend per group.

    An entry overrides `backend` for layers of the matching kind; kinds not
    listed fall back to `backend` (or automatic selection). A selected backend
    that is invalid for that kind raises at startup."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force irwkv to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from irwkv.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, set())
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @field_validator("backend_per_kind", mode="before")
    @classmethod
    def validate_backend_per_kind_before(cls, value: Any) -> Any:
        """Parse the `backend_per_kind` map from strings.

        Keys must be valid `KVCacheSpecKind` values; values are parsed like
        `backend` (enum name, case-insensitive).
        """
        from irwkv.v1.kv_cache_interface import KVCacheSpecKind

        if not isinstance(value, dict):
            return value
        valid_kinds = {kind.value for kind in KVCacheSpecKind}
        parsed: dict[str, AttentionBackendEnum] = {}
        for kind, backend in value.items():
            if kind not in valid_kinds:
                raise ValueError(
                    f"Unknown KV cache group kind '{kind}' in "
                    f"backend_per_kind. Valid kinds are: "
                    f"{', '.join(sorted(valid_kinds))}."
                )
            if isinstance(backend, str):
                backend = AttentionBackendEnum[backend.upper()]
            parsed[kind] = backend
        return parsed
