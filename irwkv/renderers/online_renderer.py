# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from openai.types.responses import ResponseOutputItem
from openai.types.responses.tool import Mcp, Tool

from irwkv.config import ModelConfig
from irwkv.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
    ConversationMessage,
    has_non_text_content,
)
from irwkv.entrypoints.generate.base.protocol import validate_request_mm_kwargs
from irwkv.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from irwkv.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)
from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest
from irwkv.entrypoints.openai.responses.utils import (
    construct_input_messages,
    construct_tool_dicts,
)
from irwkv.entrypoints.serve import create_error_response
from irwkv.entrypoints.serve.engine.protocol import ErrorResponse
from irwkv.entrypoints.serve.utils.request_logger import RequestLogger
from irwkv.exceptions import IRWKVValidationError
from irwkv.inputs import (
    EngineInput,
    PromptType,
    SingletonPrompt,
    tokens_input,
)
from irwkv.logger import init_logger
from irwkv.parser import Parser, ParserManager
from irwkv.renderers import BaseRenderer, ChatParams, merge_kwargs
from irwkv.renderers.inputs.preprocess import (
    parse_model_prompt,
    prompt_to_seq,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from irwkv.entrypoints.mcp.tool_server import ToolServer


ResponsesPreviousMessages: TypeAlias = list[ChatCompletionMessageParam]


@dataclass(frozen=True)
class ResponsesRenderResult:
    messages: ResponsesPreviousMessages
    engine_input: EngineInput


def _extract_allowed_tools_from_mcp_requests(
    tools: list[Tool],
) -> dict[str, list[str] | None]:
    allowed_tools_map: dict[str, list[str] | None] = {}
    for tool in tools:
        if not isinstance(tool, Mcp):
            continue

        allowed_tools = None
        if isinstance(tool.allowed_tools, list):
            allowed_tools = tool.allowed_tools
        elif tool.allowed_tools is not None:
            allowed_tools = tool.allowed_tools.tool_names
        if allowed_tools is not None and "*" in allowed_tools:
            allowed_tools = None
        allowed_tools_map[tool.server_label] = allowed_tools
    return allowed_tools_map


def _reused_prompt_token_ids(
    request: Any, renderer: BaseRenderer, messages: list[Any] | None = None
) -> list[int] | None:
    """Pop prompt token ids forwarded for decode-side reuse, if any.

    Disaggregated serving carries the prefill stage's ids in
    ``kv_transfer_params`` so the decode stage can skip re-tokenizing. Removing
    the key keeps the id list out of the engine's sampling metadata.

    Returns None without checking the ids when ``echo`` is set or ``messages``
    has non-text content, since both need ``messages`` to be rendered.
    Otherwise raises IRWKVValidationError if the ids are malformed or out of
    vocabulary.
    """
    kv = getattr(request, "kv_transfer_params", None)
    if not isinstance(kv, dict):
        return None
    ids = kv.pop("prompt_token_ids", None)
    if ids is None:
        return None
    if getattr(request, "echo", False):
        logger.debug(
            "Ignoring kv_transfer_params['prompt_token_ids']: "
            "echo is set, so messages are rendered instead."
        )
        return None
    if has_non_text_content(messages):
        logger.debug(
            "Ignoring kv_transfer_params['prompt_token_ids']: "
            "messages have non-text content and are rendered instead."
        )
        return None
    # bool is an int subclass, hence the exact type check.
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or x < 0 for x in ids)
    ):
        raise IRWKVValidationError(
            "`kv_transfer_params['prompt_token_ids']` must be a non-empty list "
            "of non-negative integers.",
            parameter="kv_transfer_params.prompt_token_ids",
        )
    # The engine checks this too, but only after a streamed response starts.
    renderer.validate_token_ids(ids, parameter="kv_transfer_params.prompt_token_ids")
    return ids


class OnlineRenderer:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        trust_request_mm_kwargs: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        tool_strict_level: str = "auto",
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.request_logger = request_logger

        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.parser: type[Parser] | None = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_strict_level=tool_strict_level,
            model_name=model_config.model,
            tokenizer=renderer.tokenizer,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: ChatTemplateContentFormatOption = (
            chat_template_content_format
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.trust_request_chat_template = trust_request_chat_template
        self.trust_request_mm_kwargs = trust_request_mm_kwargs

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def effective_chat_template_kwargs(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def render_chat(
        self,
        request: ChatCompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.

        Decode-side token reuse (ids forwarded in ``kv_transfer_params``) is
        handled deeper, in ``preprocess_chat``,
        so it skips only templating and tokenization while tool-choice
        validation and ``adjust_request`` still run and the output is
        detokenized (text-out).
        """
        tool_parser = self.parser.tool_parser_cls if self.parser is not None else None

        tool_parsing_unavailable = tool_parser is None

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (
            None,
            "none",
        ):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                # "required" or named tool requires tool parser
                if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
                    tool_choice_desc = f'function "{request.tool_choice.function.name}"'
                else:
                    tool_choice_desc = f'"{request.tool_choice}"'
                return self.create_error_response(
                    f"tool_choice={tool_choice_desc} requires "
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        error_check_ret = self.validate_chat_template(
            request_chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=self.trust_request_chat_template,
        )
        if error_check_ret is not None:
            return error_check_ret

        conversation, engine_inputs = await self.preprocess_chat(
            request,
            request.messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=self.default_chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=self.parser,
            skip_mm_cache=skip_mm_cache,
        )

        return conversation, engine_inputs

    async def render_responses(
        self,
        request: ResponsesRequest,
        *,
        previous_messages: ResponsesPreviousMessages | None = None,
        previous_response_outputs: list[ResponseOutputItem] | None = None,
        tool_server: "ToolServer | None" = None,
        skip_mm_cache: bool = False,
    ) -> ResponsesRenderResult | ErrorResponse:
        """Render a Responses request using only explicitly supplied history."""
        template_error = self.validate_chat_template(
            request_chat_template=None,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=self.trust_request_chat_template,
        )
        if template_error is not None:
            return template_error

        tool_dicts = construct_tool_dicts(
            request.tools,
            request.tool_choice,
            exclude_tools_when_tool_choice_none=(
                self.exclude_tools_when_tool_choice_none
            ),
        )
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=list(previous_messages) if previous_messages is not None else None,
            prev_response_output=(
                list(previous_response_outputs)
                if previous_response_outputs is not None
                else None
            ),
        )
        chat_template_kwargs = self.effective_chat_template_kwargs(request)
        _, engine_inputs = await self.preprocess_chat(
            request,
            messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=self.parser,
            skip_mm_cache=skip_mm_cache,
        )
        return self._responses_render_result(messages, engine_inputs)

    def _responses_render_result(
        self,
        messages: list[ChatCompletionMessageParam],
        engine_inputs: list[EngineInput],
    ) -> ResponsesRenderResult | ErrorResponse:
        if len(engine_inputs) != 1:
            return self.create_error_response(
                f"Expected exactly 1 engine prompt, got {len(engine_inputs)}"
            )
        return ResponsesRenderResult(
            messages=messages,
            engine_input=engine_inputs[0],
        )

    async def render_completion(
        self,
        request: CompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput] | ErrorResponse:
        """Core preprocessing logic for completion requests (no model/engine check).

        Called directly by render_completion_request and delegated to by
        OpenAIServingCompletion.render_completion_request after its engine-aware checks.
        """
        prompt_input = request.prompt
        if request.suffix is not None:
            if request.echo:
                return self.create_error_response(
                    "Echo is unsupported with suffix.",
                    param="suffix",
                )

            if request.prompt_embeds is not None:
                return self.create_error_response(
                    "suffix is not supported with prompt_embeds",
                    param="suffix",
                )

            if request.truncate_prompt_tokens is not None:
                return self.create_error_response(
                    "suffix is not supported with truncate_prompt_tokens",
                    param="suffix",
                )

            if isinstance(request.prompt, str):
                rendered_prompt = self.renderer.render_completion_suffix(
                    request.prompt, request.suffix
                )
                if rendered_prompt is None:
                    return self.create_error_response(
                        "suffix is only supported for models with FIM completion "
                        "rendering",
                        param="suffix",
                    )
                prompt_input = rendered_prompt
            elif isinstance(request.prompt, list) and all(
                isinstance(prompt, str) for prompt in request.prompt
            ):
                rendered_prompts = []
                for prompt in cast(list[str], request.prompt):
                    rendered_prompt = self.renderer.render_completion_suffix(
                        prompt, request.suffix
                    )
                    if rendered_prompt is None:
                        return self.create_error_response(
                            "suffix is only supported for models with FIM completion "
                            "rendering",
                            param="suffix",
                        )
                    rendered_prompts.append(rendered_prompt)
                prompt_input = rendered_prompts
            else:
                return self.create_error_response(
                    "suffix requires text prompt input for FIM completion rendering",
                    param="suffix",
                )

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=prompt_input,
            prompt_embeds=request.prompt_embeds,
            skip_mm_cache=skip_mm_cache,
        )

        return engine_inputs

    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    def validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        """Copied from GenerateBaseServing._validate_chat_template."""
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    async def preprocess_completion(
        self,
        request: Any,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput]:
        """Copied from GenerateBaseServing._preprocess_completion."""
        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))
        return await self.preprocess_cmpl(request, prompts, skip_mm_cache=skip_mm_cache)

    async def preprocess_cmpl(
        self,
        request: Any,
        prompts: Sequence[PromptType | bytes],
        *,
        skip_mm_cache: bool = False,
    ) -> list[EngineInput]:
        """Copied from GenerateBaseServing._preprocess_cmpl."""
        renderer = self.renderer
        model_config = self.model_config
        validate_request_mm_kwargs(
            mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
            media_io_kwargs=getattr(request, "media_io_kwargs", None),
            trust_request_mm_kwargs=self.trust_request_mm_kwargs,
        )

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

    async def preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: type[Parser] | None = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        """Copied from GenerateBaseServing._preprocess_chat."""
        renderer = self.renderer
        validate_request_mm_kwargs(
            mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
            media_io_kwargs=getattr(request, "media_io_kwargs", None),
            trust_request_mm_kwargs=self.trust_request_mm_kwargs,
        )
        mm_config = self.model_config.multimodal_config

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=self.model_config.enable_prompt_embeds,
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        reuse_ids = _reused_prompt_token_ids(request, renderer, messages)
        if reuse_ids:
            # Decode-side token reuse: feed the forwarded ids straight to the
            # engine, skipping templating and tokenization. ``messages`` are not
            # tokenized, so conversation is empty. The adjust_request tail below
            # still runs.
            conversation: list[ConversationMessage] = []
            engine_input = tokens_input(
                reuse_ids, cache_salt=getattr(request, "cache_salt", None)
            )
        else:
            (conversation,), (engine_input,) = await renderer.render_chat_async(
                [messages],
                chat_params,
                tok_params,
                prompt_extras={
                    k: v
                    for k in ("mm_processor_kwargs", "cache_salt")
                    if (v := getattr(request, k, None)) is not None
                },
                skip_mm_cache=skip_mm_cache,
            )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        #
        if parser is not None:
            tokenizer = renderer.get_tokenizer()
            tool_choice = getattr(request, "tool_choice", "none")
            should_adjust_request = (
                parser.always_adjust_request
                or parser.reasoning_parser_cls is not None
                or tool_choice != "none"
            )
            if should_adjust_request:
                if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                    msg = (
                        "Tool usage is only supported "
                        "for Chat Completions API or Responses API requests, "
                        f"but got {type(request).__name__}"
                    )
                    raise NotImplementedError(msg)
                request = parser(
                    tokenizer,
                    request.tools,
                    model_config=self.model_config,
                    chat_template_kwargs=chat_params.chat_template_kwargs,
                ).adjust_request(
                    request=request,
                )

        return conversation, [engine_input]
