# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from irwkv.engine.protocol import EngineClient
    from irwkv.entrypoints.serve.utils.request_logger import RequestLogger
    from irwkv.tasks import SupportedTask
else:
    RequestLogger = object


def register_generate_api_routers(app: FastAPI):
    from irwkv.entrypoints.openai.chat_completion.api_router import (
        attach_router as register_chat_api_router,
    )

    register_chat_api_router(app)

    from irwkv.entrypoints.openai.responses.api_router import (
        attach_router as register_responses_api_router,
    )

    register_responses_api_router(app)

    from irwkv.entrypoints.openai.completion.api_router import (
        attach_router as register_completion_api_router,
    )

    register_completion_api_router(app)


async def init_generate_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
    default_chat_template_kwargs: dict[str, Any],
):
    from irwkv.entrypoints.chat_utils import load_chat_template
    from irwkv.entrypoints.openai.chat_completion.batch_serving import (
        OpenAIServingChatBatch,
    )
    from irwkv.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from irwkv.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from irwkv.entrypoints.openai.responses.serving import OpenAIServingResponses
    from irwkv.entrypoints.serve.utils.fingerprint import set_default_fingerprint_mode

    # Applied before any serving class is constructed so that each one picks
    # up the chosen mode on its first cache miss.
    set_default_fingerprint_mode(
        getattr(args, "fingerprint_mode", "full"),
        getattr(args, "fingerprint_value", None),
    )

    resolved_chat_template = load_chat_template(args.chat_template)

    # Render endpoints are always backed by OnlineRenderer so that chat,
    # completion, and Responses rendering work on both generate-mode and
    # render-only servers. Created in init_app_state.

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.online_renderer,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_strict_level=args.tool_strict_level,
            tool_server=state.tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_per_request_metrics=args.enable_per_request_metrics,
            enable_log_outputs=args.enable_log_outputs,
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        online_renderer=state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
        enable_per_request_metrics=args.enable_per_request_metrics,
    )
    state.openai_serving_chat = (
        OpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    )
    state.openai_serving_chat_batch = (
        OpenAIServingChatBatch(**_chat_kwargs)
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            online_renderer=state.online_renderer,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_per_request_metrics=args.enable_per_request_metrics,
        )
        if "generate" in supported_tasks
        else None
    )
