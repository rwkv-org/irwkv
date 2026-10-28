# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol, TypeAlias

from irwkv.config import ModelConfig
from irwkv.entrypoints.chat_utils import ChatTemplateContentFormatOption
from irwkv.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from irwkv.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest
from irwkv.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderChatRequest,
    DerenderChatStreamRequest,
    DerenderCompletionRequest,
    DerenderCompletionStreamRequest,
    GenerateRequest,
    GenerateResponse,
)
from irwkv.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from irwkv.renderers import ChatParams, TokenizeParams


class RendererRequest(Protocol):
    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        raise NotImplementedError


class RendererChatRequest(RendererRequest, Protocol):
    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        raise NotImplementedError


CompletionLikeRequest: TypeAlias = (
    CompletionRequest
    | TokenizeCompletionRequest
    | DetokenizeRequest
    | DerenderCompletionRequest
    | DerenderCompletionStreamRequest
)

ChatLikeRequest: TypeAlias = (
    ChatCompletionRequest
    | BatchChatCompletionRequest
    | TokenizeChatRequest
    | DerenderChatRequest
    | DerenderChatStreamRequest
)

AnyRequest: TypeAlias = (
    CompletionLikeRequest | ChatLikeRequest | ResponsesRequest | GenerateRequest
)

AnyResponse: TypeAlias = (
    CompletionResponse | ChatCompletionResponse | TokenizeResponse | GenerateResponse
)
