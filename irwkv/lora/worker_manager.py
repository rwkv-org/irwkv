# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from irwkv.lora.lora_model import LoRAModel
from irwkv.lora.peft_helper import PEFTHelper
from irwkv.lora.request import LoRARequest
from irwkv.utils.torch_utils import PIN_MEMORY


class WorkerLoRAManager:
    """Manage RWKV LoRA snapshots and the model's captured adapter buffers."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
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
        self.adapters: dict[int, LoRAModel] = {}
        self.pins: Counter[int] = Counter()
        self.retired: set[int] = set()
        self.active: int | None = None

    def _load_adapter(self, request: LoRARequest) -> LoRAModel:
        path = Path(request.lora_path).expanduser()
        with (path / "adapter_config.json").open() as file:
            config = json.load(file)
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
                ) or (name == "lm_head" and "lm_head" not in self.layers):
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
        return adapter

    def add_adapter(self, request: LoRARequest) -> bool:
        if request.lora_int_id in self.adapters:
            return False
        self.adapters[request.lora_int_id] = self._load_adapter(request)
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self.adapters or adapter_id in self.retired:
            return False
        self.pins[adapter_id] += 1
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        if adapter_id not in self.adapters:
            return False
        self.retired.add(adapter_id)
        if not self.pins[adapter_id]:
            del self.adapters[adapter_id]
            self.retired.remove(adapter_id)
        return True

    def unpin_adapter(self, adapter_id: int) -> bool:
        self.pins[adapter_id] -= 1
        if self.pins[adapter_id] == 0:
            del self.pins[adapter_id]
            if adapter_id in self.retired:
                del self.adapters[adapter_id]
                self.retired.remove(adapter_id)
        return True

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
