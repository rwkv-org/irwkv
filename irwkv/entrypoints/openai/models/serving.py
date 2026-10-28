# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

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
from irwkv.logger import init_logger

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

    def model_name(self) -> str:
        return self.base_model_paths[0].name

    def is_base_model(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        """Return an ErrorResponse if model_name is not served, else None."""
        if not model_name or self.is_base_model(model_name):
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
