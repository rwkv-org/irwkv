# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

import irwkv.envs as envs
from irwkv.logger import init_logger
from irwkv.v1.attention.backends.registry import MambaAttentionBackendEnum

logger = init_logger(__name__)

ConvStateLayoutType = Literal["SD", "DS"]


@functools.lru_cache
def get_conv_state_layout() -> ConvStateLayoutType:
    """Return the SSM conv state layout.

    SD = (state_len, dim) — dim is the innermost contiguous dimension.
    DS = (dim, state_len) — TP-sharded dim is on dim-1 (like HND for KV
         cache), consistent with SSM temporal state layout.
    """
    layout: ConvStateLayoutType | None = envs.IRWKV_SSM_CONV_STATE_LAYOUT
    if layout is not None:
        logger.info_once(
            "IRWKV_SSM_CONV_STATE_LAYOUT env detected. "
            "Setting SSM conv state layout to %s.",
            layout,
        )
        return layout

    return "SD"


def is_conv_state_dim_first() -> bool:
    """True when the conv state is stored as (dim, state_len) per block."""
    return get_conv_state_layout() == "DS"


@dataclass
class MambaCopySpec:
    """Data class specifying the memory-copy parameters for Mamba states used for
    prefix caching in align mode.

    Attributes:
        start_addr (int): Starting address for the memory copy operation.
        num_elements (int): Number of elements to copy from the starting address.

    """

    start_addr: int
    num_elements: int


MambaStateCopyFunc: TypeAlias = Callable[
    [torch.Tensor, list[int], int, int], MambaCopySpec
]
"""
Type alias for a function that computes a MambaCopySpec for copying state slices.
Parameters:
  state: torch.Tensor - the Mamba state tensor (e.g., conv or temporal states).
  block_ids: list[int] - the list of block indices for the state to copy.
  cur_block_idx: int - current block index within `block_ids` to copy from.
  num_accepted_tokens: int - number of accepted tokens used to compute the copy offset.
"""
MambaStateCopyFuncs: TypeAlias = tuple[MambaStateCopyFunc, ...]
MambaStateCopyFuncsByType: TypeAlias = dict[
    MambaAttentionBackendEnum, MambaStateCopyFuncs
]


def get_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a convolutional state slice.

    Works for both SD layout ``(num_blocks, state_len, dim)`` and
    DS layout ``(num_blocks, dim, state_len)``.
    """
    src_block_id = block_ids[cur_block_idx]
    offset = num_accepted_tokens - 1
    if is_conv_state_dim_first():
        # DS offset > 0 is handled by the fused postprocess kernel.
        assert offset == 0, (
            "DS conv state with num_accepted_tokens > 1 must be handled by "
            "the fused postprocess kernel, not get_conv_copy_spec"
        )
        src_state = state[src_block_id]
    else:
        # SD layout: (num_blocks, state_len, dim), with dim contiguous.
        src_state = state[src_block_id, offset:]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )


def get_temporal_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a temporal state slice."""
    src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]
    src_state = state[src_block_id]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )
