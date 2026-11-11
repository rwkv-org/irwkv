# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Request for a LoRA adapter.

    lora_int_id identifies an immutable adapter version within the engine.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    load_inplace: bool = False
    is_3d_lora_weight: bool = False

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        """Compare adapter versions by their unique IDs."""
        return (
            isinstance(value, self.__class__) and self.lora_int_id == value.lora_int_id
        )

    def __hash__(self) -> int:
        """Hash adapter versions by their unique IDs."""
        return hash(self.lora_int_id)
