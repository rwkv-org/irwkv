# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for attention-related v1 tests."""

from dataclasses import dataclass

import pytest
import torch

from irwkv.v1.attention.backend import (
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from irwkv.v1.attention.backends.registry import AttentionBackendEnum
from irwkv.v1.kv_cache_interface import (
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    compute_layout_strides,
    create_kv_cache_views,
)


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = False,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec and ModelParams."""
    # Create query start locations
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    if arange_block_indices:
        num_blocks = batch_spec.batch_size * max_blocks
        block_table_tensor = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(batch_spec.batch_size, max_blocks)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device).view(
            num_tokens
        )
    else:
        block_table_tensor = torch.randint(
            0,
            max_block_idx,
            (batch_spec.batch_size, max_blocks),
            dtype=torch.int32,
            device=device,
        )
        slot_mapping = torch.randint(
            0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device
        )

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def try_get_attention_backend(
    backend: AttentionBackendEnum,
) -> tuple[type[AttentionMetadataBuilder], type[AttentionImpl]]:
    """Try to get the attention backend class, skipping test if not found."""
    try:
        backend_class = backend.get_class()
        return backend_class.get_builder_cls(), backend_class.get_impl_cls()
    except ImportError as e:
        pytest.skip(f"{backend.name} not available: {e}")
        raise AssertionError("unreachable") from None


def try_backend_includes_kv_cache_update(
    backend: AttentionBackendEnum,
) -> bool:
    """Try to get the attention backend class, skipping test if not found."""
    try:
        backend_class = backend.get_class()
        return backend_class.forward_includes_kv_cache_update
    except ImportError as e:
        pytest.skip(f"{backend.name} not available: {e}")
        raise AssertionError("unreachable") from None


@dataclass
class BackendConfig:
    name: str
    attention_config: dict
    comp_config: dict
    specific_gpu_arch: tuple | None = None


# Define all backend configurations of full cudagraph to be tested
full_cg_backend_configs = {
    # FA3 on Hopper
    "FA3": BackendConfig(
        name="FA3",
        attention_config={
            "backend": "FLASH_ATTN",
            "flash_attn_version": 3,
            "flash_attn_max_num_splits_for_cuda_graph": 16,
        },
        comp_config={
            "cudagraph_mode": "FULL",
        },
        specific_gpu_arch=(9, 0),
    ),
    # FlashMLA on Hopper
    "FlashMLA": BackendConfig(
        name="FlashMLA",
        attention_config={"backend": "FLASHMLA"},
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
        specific_gpu_arch=(9, 0),
    ),
    # Cutlass MLA on Blackwell
    "CutlassMLA": BackendConfig(
        name="CutlassMLA",
        attention_config={"backend": "CUTLASS_MLA"},
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
        specific_gpu_arch=(10, 0),
    ),
    # FlashInfer MLA on Blackwell
    "FlashInferMLA": BackendConfig(
        name="FlashInferMLA",
        attention_config={"backend": "FLASHINFER_MLA"},
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
        specific_gpu_arch=(10, 0),
    ),
    # FlashAttention MLA on Hopper
    "FlashAttentionMLA": BackendConfig(
        name="FlashAttentionMLA",
        attention_config={
            "backend": "FLASH_ATTN_MLA",
            "flash_attn_max_num_splits_for_cuda_graph": 16,
        },
        comp_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
        specific_gpu_arch=(9, 0),
    ),
    # FA2
    "FA2": BackendConfig(
        name="FA2",
        attention_config={
            "backend": "FLASH_ATTN",
            "flash_attn_version": 2,
            "flash_attn_max_num_splits_for_cuda_graph": 16,
        },
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
    ),
    # Triton Attention
    "TritonAttn": BackendConfig(
        name="TritonAttn",
        attention_config={"backend": "TRITON_ATTN"},
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
    ),
    # FlashInfer
    "FlashInfer": BackendConfig(
        name="FlashInfer",
        attention_config={"backend": "FLASHINFER"},
        comp_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
        },
    ),
    "RocmAttn": BackendConfig(
        name="RocmAttn",
        attention_config={
            "backend": "ROCM_ATTN",
        },
        comp_config={
            "cudagraph_mode": "FULL",
        },
    ),
}


def dense_kv_cache_tensor(
    raw: torch.Tensor,
    spec: KVCacheSpec,
    num_blocks: int,
    num_layers: int,
    layout: KVCacheLayout,
    layer_names: list[str] | None = None,
) -> KVCacheTensor:
    """The ``KVCacheTensor`` for a dense allocation of ``num_layers`` layers."""
    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        spec, num_blocks, num_layers, layout
    )
    return KVCacheTensor(
        size=raw.numel(),
        layers=layer_names or [str(i) for i in range(num_layers)],
        layer_stride=layer_stride,
        block_stride=block_stride,
    )


def dense_kv_cache_views(
    raw: torch.Tensor,
    spec: KVCacheSpec,
    num_blocks: int,
    num_layers: int,
    layout: KVCacheLayout,
    kernel_block_size: int | None = None,
) -> list[torch.Tensor]:
    """``create_kv_cache_views`` for a dense allocation of ``num_layers`` layers."""
    tensor = dense_kv_cache_tensor(raw, spec, num_blocks, num_layers, layout)
    return create_kv_cache_views(
        raw,
        spec,
        num_blocks,
        layout,
        tensor,
        kernel_block_size=kernel_block_size,
    )
