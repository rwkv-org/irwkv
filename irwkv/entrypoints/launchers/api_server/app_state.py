# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from argparse import Namespace
from typing import cast

from starlette.datastructures import State

from irwkv.engine.protocol import EngineClient
from irwkv.entrypoints.chat_utils import load_chat_template
from irwkv.entrypoints.mcp.tool_server import init_tool_server
from irwkv.entrypoints.openai.models.protocol import BaseModelPath
from irwkv.entrypoints.openai.models.serving import OpenAIServingModels
from irwkv.entrypoints.serve.tokenize.serving import ServingTokenization
from irwkv.entrypoints.serve.utils.request_logger import RequestLogger
from irwkv.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from irwkv.renderers.online_derenderer import OnlineDerenderer
from irwkv.renderers.online_renderer import OnlineRenderer
from irwkv.tasks import FALLBACK_SUPPORTED_TASKS, SupportedTask


async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None:
    irwkv_config = engine_client.irwkv_config

    if args.tool_call_parser is not None:
        from irwkv.parser.metrics import init_parser_metrics

        init_parser_metrics(
            model_name=cast(str, irwkv_config.model_config.served_model_name)
        )

    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "init_app_state and will be required in a future version. "
            "Please pass 'supported_tasks' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = FALLBACK_SUPPORTED_TASKS

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.irwkv_config = irwkv_config
    state.args = args
    resolved_chat_template = load_chat_template(args.chat_template)
    default_chat_template_kwargs = dict(args.default_chat_template_kwargs or {})
    state.tool_server = (
        await init_tool_server(args) if "generate" in supported_tasks else None
    )

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
    )

    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        trust_request_mm_kwargs=args.trust_request_mm_kwargs,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    state.online_derenderer = OnlineDerenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.serving_tokenization = ServingTokenization(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    if "generate" in supported_tasks:
        from irwkv.entrypoints.generate.api_router import init_generate_state

        await init_generate_state(
            engine_client,
            state,
            args,
            request_logger,
            supported_tasks,
            default_chat_template_kwargs,
        )

        from irwkv.entrypoints.scale_out.factories import init_scale_out_state

        init_scale_out_state(state, args, engine_client, request_logger)

    await init_endpoint_plugins_state(engine_client, state, args)

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0
