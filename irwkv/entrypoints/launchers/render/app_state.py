# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from starlette.datastructures import State

from irwkv.config import IrwkvConfig
from irwkv.entrypoints.chat_utils import load_chat_template
from irwkv.entrypoints.mcp.tool_server import init_tool_server
from irwkv.entrypoints.openai.models.protocol import BaseModelPath
from irwkv.entrypoints.openai.models.serving import OpenAIModelRegistry
from irwkv.entrypoints.scale_out.factories import init_render_state
from irwkv.entrypoints.serve.tokenize.serving import ServingTokenization
from irwkv.entrypoints.serve.utils.request_logger import RequestLogger
from irwkv.plugins.endpoint_plugins.interface import init_endpoint_plugins_state
from irwkv.renderers import renderer_from_config
from irwkv.renderers.online_derenderer import OnlineDerenderer
from irwkv.renderers.online_renderer import OnlineRenderer


async def init_render_app_state(
    irwkv_config: IrwkvConfig,
    state: State,
    args: Namespace,
) -> None:
    """Initialise FastAPI app state for a CPU-only render server.

    Unlike :func:`init_app_state` this function does not require an
    :class:`~irwkv.engine.protocol.EngineClient`; it bootstraps the
    preprocessing pipeline (renderer, input_processor)
    directly from the :class:`~irwkv.config.IrwkvConfig`.
    """
    served_model_names = args.served_model_name or [args.model]
    model_registry = OpenAIModelRegistry(
        model_config=irwkv_config.model_config,
        base_model_paths=[
            BaseModelPath(name=name, model_path=args.model)
            for name in served_model_names
        ],
    )

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    renderer = renderer_from_config(irwkv_config)
    resolved_chat_template = load_chat_template(args.chat_template)
    default_chat_template_kwargs = dict(args.default_chat_template_kwargs or {})
    state.tool_server = await init_tool_server(args)

    # The config-resolved reasoning parser carries both the CLI flag (merged
    # in the entrypoint via `create_structured_outputs_config`) and any
    # model-specific default applied by `verify_and_update_config`
    # Keep `args.reasoning_parser` as the first source so callers that build
    # a IrwkvConfig without that merge still honor an explicit flag.
    reasoning_parser = (
        args.reasoning_parser or irwkv_config.structured_outputs_config.reasoning_parser
    )

    state.online_renderer = OnlineRenderer(
        model_config=irwkv_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        trust_request_mm_kwargs=args.trust_request_mm_kwargs,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )
    state.online_renderer.warmup()

    state.online_derenderer = OnlineDerenderer(
        model_config=irwkv_config.model_config,
        renderer=renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        tool_strict_level=args.tool_strict_level,
        reasoning_parser=reasoning_parser,
        default_chat_template_kwargs=default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_models = model_registry
    state.serving_tokenization = ServingTokenization(
        model_registry,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    init_render_state(state, request_logger)

    state.irwkv_config = irwkv_config
    # Disable stats logging — there is no engine to poll.
    state.log_stats = False
    state.engine_client = None
    state.args = args
    state.enable_server_load_tracking = False
    state.server_load_metrics = 0

    # No `EngineClient` exists for the render server, so plugins get `None` and
    # must handle it themselves (see `EndpointPlugin.init_state`).
    await init_endpoint_plugins_state(None, state, args)
