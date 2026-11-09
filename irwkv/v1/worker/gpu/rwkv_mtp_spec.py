# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy MTP speculative decoding for RWKV-7 inside the MRV2 model runner.

One round is a single scheduled verify step:

1. the previous decode step drafted ``K`` tokens with the light-MTP head,
2. the verify chunk runs ``[sampled, draft_1 .. draft_K]`` in one forward,
3. the matching draft prefix plus the bonus token are emitted, and the
   recurrent state is rewound to exactly that prefix using the per-token
   checkpoints the FlashRWKV2 forward writes for this step.

No extra commit forward is needed, so a round costs one chunk forward.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from irwkv.logger import init_logger

logger = init_logger(__name__)


class _DraftGraph:
    """CUDA-graph the ``depth``-step greedy draft chain for a batch of one.

    Each draft step is a small block forward plus the full 65k-vocab LM head;
    eager execution is launch-bound (~0.55 ms/step), so replaying a captured
    graph cuts most of that.  Inputs are static buffers (hidden state + the
    sampled token), the outputs are the graph-pool draft ids.
    """

    def __init__(self, head, embed, head_fn, hidden: torch.Tensor, depth: int):
        self.head = head
        self.embed = embed
        self.head_fn = head_fn
        self.depth = int(depth)
        hidden_size = int(hidden.shape[-1])
        self.hidden = torch.zeros(
            1, hidden_size, device=hidden.device, dtype=hidden.dtype)
        self.tokens = torch.zeros(1, dtype=torch.long, device=hidden.device)
        self.graph = torch.cuda.CUDAGraph()
        # warm up outside the capture (required by torch.cuda.graph)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                head.draft(self.hidden, self.tokens, embed, head_fn, self.depth)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(self.graph):
            self.drafts, self.logits = head.draft(
                self.hidden, self.tokens, embed, head_fn, self.depth)

    def run(self, hidden: torch.Tensor, sampled: torch.Tensor) -> list[list[int]]:
        self.hidden.copy_(hidden.reshape(self.hidden.shape))
        self.tokens.copy_(sampled.reshape(self.tokens.shape))
        self.graph.replay()
        return self.drafts.tolist()


class RwkvMTPSpeculator:
    """Greedy MTP drafter plus recurrent-state checkpoint rewind."""

    def __init__(
        self,
        model: torch.nn.Module,
        irwkv_config: Any,
        device: torch.device,
        depth: int,
    ) -> None:
        self.model = model
        self.irwkv_config = irwkv_config
        self.device = device
        self.depth = depth
        self.state_layer = model.get_rwkv_state_layer()
        self.head = model.mtp_head
        if self.head is None:
            raise ValueError("RWKV MTP depth is set but the model has no MTP head")
        self._metadata: Any = None
        self._layer_inputs: list[torch.Tensor | None] | None = None
        self._ffn_inputs: list[torch.Tensor | None] | None = None
        self.accepted = 0
        self.rounds = 0
        self._draft_graph: _DraftGraph | None = None
        self._draft_graph_failed = False
        self._draft_graph_enabled = (
            os.environ.get("IRWKV_RWKV_MTP_GRAPH", "1") != "0")

    @property
    def num_layers(self) -> int:
        return int(self.state_layer.num_local_layers)

    # -- checkpoints -------------------------------------------------------- #
    def prepare_verify_batch(self, input_batch: Any, attn_metadata: Any) -> None:
        """Remember the rewind buffers attached by the state layer."""
        metadata = attn_metadata
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        self._metadata = metadata

    def _rewind(
        self,
        cu_seqlens: list[int],
        state_indices: torch.Tensor,
        accepted: list[int],
        chunk_len: int,
    ) -> None:
        """Restore the recurrent state at ``accepted + 1`` consumed tokens."""
        metadata = self._metadata
        checkpoints = None if metadata is None else metadata.spec_checkpoints
        tmix_norm = None if metadata is None else metadata.spec_normalized_tmix
        cmix_norm = None if metadata is None else metadata.spec_normalized_cmix
        if checkpoints is None or tmix_norm is None or cmix_norm is None:
            if os.environ.get("IRWKV_RWKV_MTP_NOSAVE") == "1":
                return
            raise RuntimeError("MTP verify needs recurrent-state checkpoints")
        tmix_pool = self.state_layer._tmix_shift_pool
        cmix_pool = self.state_layer._cmix_shift_pool
        wkv_pool = self.state_layer._wkv_state_pool
        assert tmix_pool is not None and cmix_pool is not None and wkv_pool is not None
        for row, count in enumerate(accepted):
            prefix = count + 1
            if prefix >= chunk_len:
                continue  # the whole chunk is accepted; the state is final
            slot = int(state_indices[row])
            base = cu_seqlens[row] + prefix - 1
            # The kernel lays checkpoints out as CU[seq] + seq + token_offset.
            checkpoint_index = cu_seqlens[row] + row + prefix
            for layer in range(self.num_layers):
                wkv_pool[layer, slot].copy_(checkpoints[layer][checkpoint_index])
                tmix_pool[layer, slot].copy_(tmix_norm[layer][base])
                cmix_pool[layer, slot].copy_(cmix_norm[layer][base])

    # -- drafting ----------------------------------------------------------- #
    @torch.no_grad()
    def propose(
        self,
        hidden_states: torch.Tensor,
        sampled: torch.Tensor,
        depth: int,
    ) -> list[list[int]]:
        """Draft ``depth`` greedy tokens for every request in the batch."""
        embed = self.model.embed_for_mtp
        head_fn = self.model.compute_logits
        if (self._draft_graph_enabled and not self._draft_graph_failed
                and hidden_states.shape[0] == 1):
            if self._draft_graph is None:
                try:
                    self._draft_graph = _DraftGraph(
                        self.head, embed, head_fn, hidden_states, depth)
                    logger.info("RWKV MTP draft chain captured in a CUDA graph")
                except Exception as exc:  # noqa: BLE001 - fall back to eager
                    self._draft_graph_failed = True
                    logger.warning(
                        "RWKV MTP draft CUDA graph disabled: %r", exc)
            if self._draft_graph is not None and self._draft_graph.depth == depth:
                return self._draft_graph.run(hidden_states, sampled)
        drafts, _ = self.head.draft(hidden_states, sampled, embed, head_fn, depth)
        return drafts.tolist()

    # -- verification ------------------------------------------------------- #
    @torch.no_grad()
    def verify(
        self,
        chunk_hidden_states: torch.Tensor,
        cu_seqlens: list[int],
        drafts: list[list[int]],
        state_indices: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Accept the matching draft prefix, rewind the state, return the bonus.

        Returns ``(accepted_counts, bonus_tokens)`` per request.
        """
        logits = self.model.compute_logits(chunk_hidden_states)
        predictions = logits.argmax(-1)
        accepted: list[int] = []
        bonus: list[int] = []
        chunk_len = int(chunk_hidden_states.shape[0]) // len(drafts)
        for row, draft in enumerate(drafts):
            start = cu_seqlens[row]
            hit = 0
            for index, token in enumerate(draft):
                if int(predictions[start + index]) != int(token):
                    break
                hit += 1
            accepted.append(hit)
            bonus.append(int(predictions[start + hit]))
        self._rewind(cu_seqlens, state_indices, accepted, chunk_len)
        self.rounds += len(drafts)
        self.accepted += sum(accepted)
        if os.environ.get("IRWKV_RWKV_MTP_STATS") and self.rounds % 32 == 0:
            import sys

            print(
                f"[mtp-stats] rounds={self.rounds} accepted={self.accepted} "
                f"accept_len={1.0 + self.accepted / self.rounds:.2f}",
                file=sys.stderr,
                flush=True,
            )
        return accepted, bonus
