# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Light multi-token-prediction drafter for RWKV-7.

The head uses the DeepSpec ``lightmlp`` layout: a fused
``proj([norm_e(embedding), norm_h(hidden)])`` residual followed by a gated-SiLU
MLP, and reuses the base model's LM head with its own output norm.  One block is
shared by every draft depth (``mtp_loop``), matching the converted
``mtp.safetensors`` sidecar.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

MTP_CONFIG_NAME = "mtp_config.json"
MTP_WEIGHTS_NAME = "mtp.safetensors"


@dataclass(frozen=True)
class RwkvMTPConfig:
    mtp_type: str
    mtp_depth: int
    mtp_loop: bool
    hidden_size: int
    intermediate_size: int
    norm_eps: float
    weight_file: str
    num_blocks: int
    shared_lm_head: bool

    @classmethod
    def from_model_path(cls, model_path: str) -> "RwkvMTPConfig | None":
        path = Path(model_path) / MTP_CONFIG_NAME
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        config = cls(
            mtp_type=raw["mtp_type"],
            mtp_depth=int(raw["mtp_depth"]),
            mtp_loop=bool(raw["mtp_loop"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            norm_eps=float(raw["norm_eps"]),
            weight_file=raw["weight_file"],
            num_blocks=int(raw.get("num_blocks", 1)),
            shared_lm_head=bool(raw.get("shared_lm_head", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mtp_type != "lightmlp":
            raise ValueError(f"Unsupported RWKV MTP type {self.mtp_type!r}")
        if not self.shared_lm_head:
            raise ValueError("RWKV MTP requires the base model's LM head")
        if self.num_blocks != 1:
            raise ValueError("RWKV MTP supports exactly one shared block")
        if self.mtp_depth < 1:
            raise ValueError("RWKV MTP depth must be >= 1")


class RwkvMTPBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.norm_h = nn.LayerNorm(hidden_size, eps=eps)
        self.norm_e = nn.LayerNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.norm_mlp = nn.LayerNorm(hidden_size, eps=eps)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm_out = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden: torch.Tensor, token_emb: torch.Tensor) -> torch.Tensor:
        fused = self.proj(
            torch.cat([self.norm_e(token_emb), self.norm_h(hidden)], dim=-1)
        )
        x = hidden + fused
        mlp_in = self.norm_mlp(x)
        return x + self.down(torch.nn.functional.silu(self.gate(mlp_in)) * self.up(mlp_in))

    def head_logits(self, hidden: torch.Tensor, head_fn: Callable) -> torch.Tensor:
        normalized = self.norm_out(hidden)
        flat = normalized.reshape(-1, normalized.shape[-1])
        logits = head_fn(flat)
        return logits.view(*hidden.shape[:-1], logits.shape[-1])


class RwkvMTPHead(nn.Module):
    """Drafting head loaded from ``mtp.safetensors``."""

    def __init__(self, config: RwkvMTPConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.block = RwkvMTPBlock(
            config.hidden_size, config.intermediate_size, config.norm_eps
        ).to(dtype)

    @property
    def depth(self) -> int:
        return self.config.mtp_depth

    @classmethod
    def from_model_path(
        cls, model_path: str, dtype: torch.dtype = torch.bfloat16
    ) -> "RwkvMTPHead | None":
        config = RwkvMTPConfig.from_model_path(model_path)
        if config is None:
            return None
        from safetensors.torch import load_file

        head = cls(config, dtype)
        state = load_file(str(Path(model_path) / config.weight_file))
        remapped = {}
        for key, value in state.items():
            prefix = "mtp.blocks.0."
            if not key.startswith(prefix):
                raise ValueError(f"Unexpected RWKV MTP tensor {key!r}")
            remapped[f"block.{key[len(prefix):]}"] = value
        head.load_state_dict(remapped, strict=True)
        return head

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        embed: Callable[[torch.Tensor], torch.Tensor],
        head_fn: Callable[[torch.Tensor], torch.Tensor],
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft ``depth`` greedy tokens from one hidden state.

        Args:
            hidden: ``[B, C]`` hidden state of the last accepted position.
            token_ids: ``[B]`` next token ids the drafts condition on.
            embed: base model token embedding.
            head_fn: base model logits projection (fused LM head).
            depth: number of drafts, at most ``config.mtp_depth``.

        Returns:
            ``[B, depth]`` draft token ids and ``[B, depth, V]`` logits.
        """
        depth = min(depth, self.config.mtp_depth)
        h = hidden.unsqueeze(1)
        drafted: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for _ in range(depth):
            h = self.block(h, embed(token_ids).unsqueeze(1))
            step_logits = self.block.head_logits(h, head_fn)[:, -1]
            token_ids = step_logits.argmax(-1)
            drafted.append(token_ids)
            logits.append(step_logits)
        return torch.stack(drafted, dim=1), torch.stack(logits, dim=1)
