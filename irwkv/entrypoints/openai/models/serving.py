# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from itertools import count

from irwkv.config import ModelConfig
from irwkv.engine.protocol import EngineClient
from irwkv.entrypoints.openai.models.protocol import BaseModelPath
from irwkv.entrypoints.serve import create_error_response
from irwkv.entrypoints.serve.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from irwkv.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from irwkv.exceptions import IRWKVNotFoundError, LoRAAdapterNotFoundError
from irwkv.logger import init_logger
from irwkv.lora.request import LoRARequest

logger = init_logger(__name__)


class OpenAIModelRegistry:
    """Read-only view of the loaded base models with no engine dependency.

    Suitable for rendering contexts that have no engine client.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        base_model_paths: list[BaseModelPath],
    ) -> None:
        self.model_config = model_config
        self.base_model_paths = base_model_paths

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        return lora_request.lora_name if lora_request else self.base_model_paths[0].name

    def get_lora(self, model_name: str | None) -> LoRARequest | None:
        return None

    def is_base_model(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        """Return an ErrorResponse if model_name is not served, else None."""
        if (
            not model_name
            or self.is_base_model(model_name)
            or self.get_lora(model_name)
        ):
            return None
        return create_error_response(
            message=f"The model `{model_name}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self.base_model_paths
            ]
        )


class OpenAIServingModels(OpenAIModelRegistry):
    """Loaded base models exposed through /v1/models."""

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
    ):
        super().__init__(engine_client.model_config, base_model_paths)
        self.engine_client = engine_client
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        self.lora_requests: dict[str, LoRARequest] = {}
        self._lora_id = count(1)
        self._lora_lock = asyncio.Lock()
        self._lora_refs: Counter[int] = Counter()
        self._retired_loras: set[int] = set()

    def get_lora(self, model_name: str | None) -> LoRARequest | None:
        return self.lora_requests.get(model_name) if model_name else None

    @asynccontextmanager
    async def use_lora(self, lora: LoRARequest) -> AsyncIterator[None]:
        if self.get_lora(lora.lora_name) != lora:
            raise IRWKVNotFoundError(f"The model `{lora.lora_name}` does not exist.")
        lora_id = lora.lora_int_id
        self._lora_refs[lora_id] += 1
        try:
            yield
        finally:
            self._lora_refs[lora_id] -= 1
            if not self._lora_refs[lora_id]:
                del self._lora_refs[lora_id]
                if lora_id in self._retired_loras:
                    self._retired_loras.remove(lora_id)
                    await asyncio.shield(self.engine_client.remove_lora(lora_id))

    async def _retire_lora(self, lora_id: int) -> None:
        if self._lora_refs[lora_id]:
            self._retired_loras.add(lora_id)
        else:
            await self.engine_client.remove_lora(lora_id)

    async def show_available_models(self) -> ModelList:
        models = await super().show_available_models()
        models.data.extend(
            ModelCard(
                id=lora.lora_name,
                root=lora.lora_path,
                parent=self.base_model_paths[0].name,
                max_model_len=self.model_config.max_model_len,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        )
        return models

    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        return await asyncio.shield(self._load_lora_adapter(request))

    async def _load_lora_adapter(
        self, request: LoadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        async with self._lora_lock:
            name, path = request.lora_name, request.lora_path
            if (
                not name
                or not path
                or self.is_base_model(name)
                or request.is_3d_lora_weight
            ):
                return create_error_response(
                    "Invalid LoRA adapter name, path, or format"
                )
            previous = self.lora_requests.get(name)
            if previous is not None and not request.load_inplace:
                return create_error_response(f"LoRA adapter {name!r} is already loaded")
            lora = LoRARequest(name, next(self._lora_id), path)
            try:
                if not await self.engine_client.add_lora(lora):
                    raise RuntimeError(f"Could not load LoRA adapter {name!r}")
            except Exception as error:
                not_found = LoRAAdapterNotFoundError(name, path)
                if str(not_found) in str(error):
                    return create_error_response(not_found)
                return create_error_response(
                    str(error),
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            self.lora_requests[name] = lora
            if previous is not None:
                await self._retire_lora(previous.lora_int_id)
            return f"Success: LoRA adapter '{name}' added successfully."

    async def unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        return await asyncio.shield(self._unload_lora_adapter(request))

    async def _unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        async with self._lora_lock:
            lora = self.lora_requests.pop(request.lora_name, None)
            if lora is None:
                return create_error_response(
                    f"LoRA adapter {request.lora_name!r} was not found",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )
            await self._retire_lora(lora.lora_int_id)
            return f"Success: LoRA adapter '{request.lora_name}' removed successfully."
