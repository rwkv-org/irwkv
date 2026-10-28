# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from irwkv.logger import init_logger
from irwkv.platforms import current_platform

logger = init_logger(__name__)

current_platform.import_kernels()


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


# Fused vocab-parallel embedding lookup
# (see csrc/.../vocab_parallel_embedding_kernels.cu).
def vocab_parallel_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> torch.Tensor:
    """Gather the embedding rows owned by this rank, zeros for the others.

    Fuses the range mask, id shift, gather and output masking of the TP > 1
    embedding path into a single kernel. Out-of-shard tokens produce zero rows,
    so an all-reduce over the ranks yields the full embedding.

    Args:
        input_ids: ``[num_tokens]`` int32 or int64 token ids.
        weight: ``[num_embeddings_per_partition, embedding_dim]`` shard.
        org_vocab_start_index: First original vocab id owned by this rank.
        org_vocab_end_index: One past the last original vocab id owned by
            this rank.
        num_org_vocab_padding: Padding rows between this rank's original and
            added vocab shards.
        added_vocab_start_index: First added vocab id owned by this rank.
        added_vocab_end_index: One past the last added vocab id owned by
            this rank.

    Returns:
        ``[num_tokens, embedding_dim]`` partial embeddings.

    """
    out = torch.empty(
        input_ids.shape[0],
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    )
    torch.ops._C.vocab_parallel_embedding(
        out,
        input_ids,
        weight,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    return out


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    # Note: this func is batch invariant
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    torch.ops._C.apply_repetition_penalties_(
        logits, prompt_mask, output_mask, repetition_penalties
    )


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).

    """
    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# fused silu_and_mul + block quant


# gptq


# cutlass


# gptq_marlin


# machete


# CUTLASS W4A8


# fp4


# fp8


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device
    )


# custom ar
def init_custom_ar(
    ipc_tensors: list[torch.Tensor],
    rank_data: torch.Tensor,
    rank: int,
    fully_connected: bool,
) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(
        ipc_tensors, rank_data, rank, fully_connected
    )


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def custom_all_gather(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.custom_all_gather(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def mnnvl_lamport_all_gather(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    local_buffer: int,
    multicast_buffer: int,
    epoch_buffer: int,
    stage_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.mnnvl_lamport_all_gather(
        fa,
        inp,
        out,
        local_buffer,
        multicast_buffer,
        epoch_buffer,
        stage_sz_bytes,
    )


def custom_reduce_scatter(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.custom_reduce_scatter(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def mnnvl_lamport_reduce_scatter(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    local_buffer: int,
    epoch_buffer: int,
    stage_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.mnnvl_lamport_reduce_scatter(
        fa, inp, out, local_buffer, epoch_buffer, stage_sz_bytes
    )


def mnnvl_multimem_reduce_scatter(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    local_buffer: int,
    multicast_buffer: int,
    stage_sz_bytes: int,
    block_limit: int,
) -> None:
    torch.ops._C_custom_ar.mnnvl_multimem_reduce_scatter(
        fa,
        inp,
        out,
        local_buffer,
        multicast_buffer,
        stage_sz_bytes,
        block_limit,
    )


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(
    fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce
def init_custom_qr(rank: int, world_size: int, qr_max_size: int | None = None) -> int:
    return torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)


def qr_destroy(fa: int) -> None:
    torch.ops._C_custom_ar.qr_destroy(fa)


def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool = False,
) -> None:
    torch.ops._C_custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)


def qr_get_handle(fa: int) -> torch.Tensor:
    return torch.ops._C_custom_ar.qr_get_handle(fa)


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    return torch.ops._C_custom_ar.qr_open_handles(fa, handles)


def qr_max_size() -> int:
    return torch.ops._C_custom_ar.qr_max_size()


_supports_cpu_fp8_w8a16 = bool(hasattr(torch.ops._C, "fp8_scaled_mm_cpu"))


# FP8 W8A8 CPU kernels


_supports_onednn = bool(hasattr(torch.ops._C, "create_onednn_mm_handler"))
