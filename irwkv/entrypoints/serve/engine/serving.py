# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from fastapi import Request

from irwkv import PromptType, SamplingParams, envs
from irwkv.config import ModelConfig
from irwkv.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from irwkv.entrypoints.serve import create_error_response
from irwkv.entrypoints.serve.engine.protocol import ErrorResponse
from irwkv.entrypoints.serve.engine.typing import AnyRequest
from irwkv.entrypoints.serve.utils.request_logger import RequestLogger
from irwkv.inputs import EngineInput
from irwkv.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from irwkv.sampling_params import BeamSearchParams
from irwkv.utils import random_uuid


class BaseServing:
    def __init__(
        self,
        models: OpenAIServingModels | OpenAIModelRegistry,
        model_config: ModelConfig,
        request_logger: RequestLogger | None = None,
    ):
        self.models = models
        self.model_config = model_config
        self.request_logger = request_logger

    async def _check_model(
        self,
        request: AnyRequest,
    ) -> ErrorResponse | None:
        if self.request_logger is not None:
            self.request_logger.log_request_body(request)

        if self._is_model_supported(request.model):
            return None

        served_names = ", ".join(model.name for model in self.models.base_model_paths)
        return self.create_error_response(
            message=(
                f"The model `{request.model}` does not exist. "
                f"Valid aliases: {served_names}."
            ),
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    def _is_model_supported(self, model_name: str | None) -> bool:
        if not model_name:
            return True
        if envs.IRWKV_SKIP_MODEL_NAME_VALIDATION:
            return True
        return self.models.is_base_model(model_name)

    @staticmethod
    def create_error_response(
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def _extract_prompt_components(self, prompt: PromptType | EngineInput):
        return extract_prompt_components(self.model_config, prompt)

    def _extract_prompt_text(self, prompt: PromptType | EngineInput):
        return self._extract_prompt_components(prompt).text

    def _extract_prompt_len(self, prompt: EngineInput):
        return extract_prompt_len(self.model_config, prompt)

    def _log_inputs(
        self,
        request_id: str,
        inputs: PromptType | EngineInput,
        params: SamplingParams | BeamSearchParams | None,
    ) -> None:
        if self.request_logger is None:
            return

        components = self._extract_prompt_components(inputs)

        self.request_logger.log_inputs(
            request_id,
            components.text,
            components.token_ids,
            components.embeds,
            params=params,
        )

    @staticmethod
    def _base_request_id(
        raw_request: Request | None, default: str | None = None
    ) -> str | None:
        """Pulls the request id to use from a header, if provided."""
        if raw_request is not None and (
            (req_id := raw_request.headers.get("X-Request-Id")) is not None
        ):
            return req_id

        return random_uuid() if default is None else default
