# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contains replacement functions to fallback Triton usages in CPU backend."""

import ctypes

import torch


def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,  # [num_reqs + 1], int32
    positions: torch.Tensor,  # [num_tokens], int64
    block_table: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride: int,  # max_num_blocks_per_req
    block_size: int,
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: int | None = None,
    BLOCKS_PER_KV_BLOCK: int = 1,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = -1,
    BLOCK_SIZE: int = 1024,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    if BLOCKS_PER_KV_BLOCK != 1:
        assert block_size * BLOCKS_PER_KV_BLOCK == KV_CACHE_BLOCK_SIZE
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        block_size,
    )


def _batch_memcpy_impl(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=None):
    # BLOCK_SIZE is unused; kept for signature parity with the Triton kernel.
    for src, dst, size in zip(src_ptrs.tolist(), dst_ptrs.tolist(), sizes.tolist()):
        ctypes.memmove(dst, src, size)
