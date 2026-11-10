# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import safetensors
import torch

from irwkv.lora.lora_weights import LoRALayerWeights
from irwkv.lora.peft_helper import PEFTHelper
from irwkv.lora.utils import is_base_embedding_weights, parse_fine_tuned_lora_name
from irwkv.utils.torch_utils import PIN_MEMORY


class LoRAModel:
    """A LoRA fine-tuned model."""

    def __init__(
        self, lora_model_id: int, rank: int, loras: dict[str, LoRALayerWeights]
    ) -> None:
        assert lora_model_id > 0
        self.id = lora_model_id
        self.rank = rank
        self.loras = loras

    def get_lora(self, module_name: str) -> LoRALayerWeights | None:
        return self.loras.get(module_name)

    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> "LoRAModel":
        pin_memory = device == "cpu" and PIN_MEMORY
        loras: dict[str, LoRALayerWeights] = {}
        for tensor_name, tensor in tensors.items():
            if is_base_embedding_weights(tensor_name):
                continue
            module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
            if module_name not in loras:
                loras[module_name] = LoRALayerWeights.from_config(
                    module_name, peft_helper
                )
            tensor = tensor.to(device=device, dtype=dtype)
            if pin_memory:
                tensor = tensor.pin_memory()
            if is_lora_a:
                loras[module_name].lora_a = tensor
            else:
                loras[module_name].lora_b = tensor
        return cls(lora_model_id, peft_helper.r, loras)

    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ) -> "LoRAModel":
        """Create a LoRAModel from a local checkpoint."""
        lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
        lora_pt_file_path = os.path.join(lora_dir, "adapter_model.pt")

        if os.path.isfile(lora_tensor_path):
            with safetensors.safe_open(lora_tensor_path, framework="pt") as f:
                names = f.keys()
                tensors = {name: f.get_tensor(name) for name in names}
        elif os.path.isfile(lora_bin_file_path) or os.path.isfile(lora_pt_file_path):
            path = (
                lora_bin_file_path
                if os.path.isfile(lora_bin_file_path)
                else lora_pt_file_path
            )
            tensors = torch.load(path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"{lora_dir} doesn't contain tensors")

        unexpected = []
        for tensor_name in tensors:
            if is_base_embedding_weights(tensor_name):
                continue
            module_name, _ = parse_fine_tuned_lora_name(tensor_name)
            if module_name.rsplit(".", 1)[-1] not in expected_lora_modules:
                unexpected.append(module_name)
        if unexpected:
            raise ValueError(
                f"While loading {lora_dir}, expected target modules in "
                f"{expected_lora_modules} but received {unexpected}."
            )

        return cls.from_lora_tensors(
            lora_model_id, tensors, peft_helper, device=device, dtype=dtype
        )
