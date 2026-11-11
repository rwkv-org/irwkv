# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from irwkv.distributed import get_pp_group
from irwkv.exceptions import LoRAAdapterNotFoundError
from irwkv.lora.lora_model import LoRAModel
from irwkv.lora.peft_helper import PEFTHelper
from irwkv.lora.request import LoRARequest
from irwkv.utils.torch_utils import PIN_MEMORY


class WorkerLoRAManager:
    """Manage RWKV LoRA snapshots and the model's captured adapter buffers."""

    def __init__(self, model: torch.nn.Module, max_cpu_loras: int) -> None:
        self.model = model
        self.max_cpu_loras = max_cpu_loras
        config = getattr(model.config, "rwkv_lora", None)
        if config is None:
            raise ValueError("RWKV LoRA requires the rwkv_lora model configuration")
        self.config: dict[str, Any] = config
        params = dict(model.named_parameters())
        self.layers = {
            name.removesuffix(".lora_A.weight"): (
                param,
                params[name.replace(".lora_A.weight", ".lora_B.weight")],
            )
            for name, param in params.items()
            if name.endswith(".lora_A.weight")
        }
        self.adapters: dict[int, LoRAModel] = {}
        self.active: int | None = None
        self.refresh_base_weights()

    def refresh_base_weights(self) -> None:
        self.base = {
            name: (
                a.detach().cpu().pin_memory()
                if PIN_MEMORY
                else a.detach().cpu().clone(),
                b.detach().cpu().pin_memory()
                if PIN_MEMORY
                else b.detach().cpu().clone(),
            )
            for name, (a, b) in self.layers.items()
        }

    def _load_adapter(self, request: LoRARequest) -> LoRAModel:
        path = Path(request.lora_path).expanduser()
        try:
            with (path / "adapter_config.json").open() as file:
                config = json.load(file)
        except FileNotFoundError as exc:
            raise LoRAAdapterNotFoundError(
                request.lora_name, request.lora_path
            ) from exc
        peft = PEFTHelper.from_dict(config)
        peft.validate_legal(SimpleNamespace(max_lora_rank=self.config["r"]))
        if (
            peft.r != self.config["r"]
            or peft.vllm_lora_scaling_factor
            != self.config["lora_alpha"] / self.config["r"]
            or config.get("use_nora", False) != self.config["use_nora"]
            or config.get("rank_pattern")
            or config.get("alpha_pattern")
            or config.get("peft_type", "LORA") != "LORA"
            or peft.modules_to_save
        ):
            raise ValueError("LoRA configuration does not match the RWKV model")
        expected = {name.rsplit(".", 1)[-1] for name in self.layers}
        expected.update(("r_proj", "k_proj", "v_proj", "o_proj", "key", "value"))
        expected.add("lm_head")
        adapter = LoRAModel.from_local_checkpoint(
            str(path),
            expected,
            peft,
            lora_model_id=request.lora_int_id,
            device="cpu",
            dtype=torch.float32,
        )
        local_loras = {}
        for name, weights in adapter.loras.items():
            if name not in self.layers:
                parts = name.split(".")
                if (
                    len(parts) > 3
                    and parts[:2] == ["model", "layers"]
                    and parts[2].isdigit()
                    and not (
                        self.model.model.start_layer
                        <= int(parts[2])
                        < self.model.model.end_layer
                    )
                ) or (name == "lm_head" and not get_pp_group().is_last_rank):
                    continue
                raise ValueError(f"LoRA target {name!r} is not enabled for RWKV")
            a, b = self.layers[name]
            if (
                weights.lora_a is None
                or weights.lora_b is None
                or weights.lora_a.shape != a.shape
                or weights.lora_b.shape != b.shape
            ):
                raise ValueError(f"Invalid LoRA tensor shapes for {name!r}")
            if PIN_MEMORY:
                weights.lora_a = weights.lora_a.pin_memory()
                weights.lora_b = weights.lora_b.pin_memory()
            local_loras[name] = weights
        adapter.loras = local_loras
        return adapter

    def add_adapter(self, request: LoRARequest) -> bool:
        if request.lora_int_id in self.adapters:
            return False
        if len(self.adapters) >= self.max_cpu_loras:
            raise ValueError(
                f"LoRA capacity reached (--max-cpu-loras={self.max_cpu_loras})"
            )
        self.adapters[request.lora_int_id] = self._load_adapter(request)
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        return self.adapters.pop(adapter_id, None) is not None

    @torch.no_grad()
    def set_active_adapter(self, adapter_id: int | None) -> None:
        if self.active == adapter_id:
            return
        adapter = self.adapters[adapter_id] if adapter_id is not None else None
        for name, (a, b) in self.layers.items():
            weights: Any = adapter.get_lora(name) if adapter is not None else None
            if weights is None:
                if adapter is None:
                    a.copy_(self.base[name][0], non_blocking=True)
                    b.copy_(self.base[name][1], non_blocking=True)
                else:
                    b.zero_()
            else:
                a.copy_(weights.lora_a, non_blocking=True)
                b.copy_(weights.lora_b, non_blocking=True)
        self.active = adapter_id
