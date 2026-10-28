# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This file contains the command line arguments for the irwkv's online server.
It is kept in a separate file for documentation purposes.
"""

import argparse
import json
import ssl
from dataclasses import field
from typing import Any, Literal

import irwkv.envs as envs
from irwkv.config import config
from irwkv.engine.arg_utils import AsyncEngineArgs
from irwkv.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    validate_chat_template,
)
from irwkv.logger import init_logger
from irwkv.tool_parsers import ToolParserManager
from irwkv.tool_parsers.tool_strict_level import ToolStrictLevelName
from irwkv.utils.argparse_utils import FlexibleArgumentParser

from .utils.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)

logger = init_logger(__name__)


@config
class BaseFrontendArgs:
    """Base arguments for the OpenAI-compatible frontend server.

    This base class does not include host, port, and server-specific arguments
    like SSL, CORS, and HTTP server settings. Those arguments are added by
    the subclasses.
    """

    chat_template: str | None = None
    """The file path to the chat template, or the template in single-line form
    for the specified model."""
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    """The format to render message content within a chat template.

    * "string" will render the content as a string. Example: `"Hello World"`
    * "openai" will render the content as a list of dictionaries, similar to
      OpenAI schema. Example: `[{"type": "text", "text": "Hello world!"}]`"""
    trust_request_chat_template: bool = False
    """Whether to trust the chat template provided in the request. If False,
    the server will always use the chat template specified by `--chat-template`
    or the ones from tokenizer."""
    trust_request_mm_kwargs: bool = False
    """Whether to trust per-request multimodal kwargs (`mm_processor_kwargs`
    and `media_io_kwargs`). If False, the server rejects non-empty values
    because they can change multimodal preprocessing resource usage. Only
    enable this when API clients are trusted."""
    default_chat_template_kwargs: dict[str, Any] | None = None
    """Default keyword arguments to pass to the chat template renderer.
    These will be merged with request-level chat_template_kwargs,
    with request values taking precedence. Useful for setting default
    behavior for reasoning models. Example: '{"enable_thinking": false}'
    to disable thinking mode by default for Qwen3/DeepSeek models."""
    response_role: str = "assistant"
    """The role name to return if `request.add_generation_prompt=true`."""
    return_tokens_as_token_ids: bool = False
    """When `--max-logprobs` is specified, represents single tokens as
    strings of the form 'token_id:{token_id}' so that tokens that are not
    JSON-encodable can be identified."""
    enable_auto_tool_choice: bool = False
    """Enable auto tool choice for supported models. Use `--tool-call-parser`
    to specify which parser to use."""
    exclude_tools_when_tool_choice_none: bool = False
    """If specified, exclude tool definitions in prompts when
    tool_choice='none'."""
    tool_call_parser: str | None = None
    """Select the tool call parser depending on the model that you're using.
    This is used to parse the model-generated tool call into OpenAI API format.
    Required for `--enable-auto-tool-choice`. You can choose any option from
    the built-in parsers or register a plugin via `--tool-parser-plugin`."""
    tool_parser_plugin: str = ""
    """Special the tool parser plugin write to parse the model-generated tool
    into OpenAI API format, the name register in this plugin can be used in
    `--tool-call-parser`."""
    tool_strict_level: ToolStrictLevelName = "auto"
    """Server-side floor for structural-tag based tool calling, applied on top
    of the per-tool `strict` field. `auto` follows the request's tool choice
    and per-tool strictness; `function` constrains the
    tool-call envelope (markup and function name) for every request with
    tools; `parameter` additionally pins argument schemas, as if every tool
    were `strict: true`."""
    tool_server: str | None = None
    """Comma-separated list of host:port pairs (IPv4, IPv6, or hostname).
    Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for
    built-in demo tools (browser and Python code interpreter). WARNING:
    The `demo` Python tool executes model-generated code in Docker without
    network isolation by default. See the security guide for more
    information."""
    max_log_len: int | None = None
    """Max number of prompt characters or prompt ID numbers being printed in
    log. The default of None means unlimited."""
    enable_prompt_tokens_details: bool = False
    """If set to True, enable prompt_tokens_details in usage."""
    enable_per_request_metrics: bool = False
    """If set to True, include per-request timing metrics in API responses."""
    enable_server_load_tracking: bool = False
    """If set to True, enable tracking server_load_metrics in the app state."""
    enable_force_include_usage: bool = False
    """If set to True, including usage on every request."""
    sse_keep_alive_interval: int = 0
    """Send an SSE keep-alive comment line every this many seconds when a
    `/v1/chat/completions` or `/v1/completions` streaming response is idle
    (queued, prefill, or between tokens), to prevent reverse proxies/tunnels
    with read timeouts from closing the connection. Defaults to 0, which
    disables keep-alive comments entirely.
    """
    enable_tokenizer_info_endpoint: bool = False
    """Enable the `/tokenizer_info` endpoint. May expose chat
    templates and other tokenizer configuration."""
    enable_log_outputs: bool = False
    """If set to True, log model outputs (generations). Requires
    `--enable-log-requests`. Output text and finish reasons are logged at INFO,
    while output token IDs are logged at DEBUG."""
    enable_log_deltas: bool = True
    """If set to False, output deltas will not be logged. Relevant only if 
    --enable-log-outputs is set.
    """
    log_error_stack: bool = envs.IRWKV_SERVER_DEV_MODE
    """If set to True, log the stack trace of error responses"""
    tokens_only: bool = False
    """
    If set to True, only enable the Tokens In<>Out endpoint.
    This is intended for use in a Disaggregated Everything setup.
    """
    enable_scale_out: bool = False
    """
    If set to True, register the scale-out endpoints (`/render`, `/derender`,
    `/inference/v1/generate` and `/inference/v1/abort_requests`) on `irwkv serve`.
    Has no effect on `irwkv serve --tokens-only`, which
    always register their required endpoints regardless of this flag.
    """
    fingerprint_mode: Literal["full", "hash", "custom", "none"] = "full"
    """Controls the ``system_fingerprint`` field on responses.

    - ``full`` (default): ``irwkv-<version>[-<parallelism>]-<hash8>``. Encodes
      server version, non-trivial parallelism degrees (tp/pp/dp/ep), and an
      8-char config hash.
    - ``hash``: ``irwkv-<version>-<hash8>``. Parallelism stripped.
    - ``custom``: emits the literal string from ``--fingerprint-value``.
    - ``none``: the field is omitted (serialized as ``null``).
    """
    fingerprint_value: str | None = None
    """Literal fingerprint string used when ``--fingerprint-mode=custom``."""

    @classmethod
    def _customize_cli_kwargs(
        cls,
        frontend_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Customize argparse kwargs before arguments are registered.

        Subclasses should override this and call
        ``super()._customize_cli_kwargs(frontend_kwargs)`` first.
        """
        # Special case: default_chat_template_kwargs needs json.loads type
        frontend_kwargs["default_chat_template_kwargs"]["type"] = json.loads

        # Special case: Tool call parser shows built-in options.
        valid_tool_parsers = list(ToolParserManager.list_registered())
        parsers_str = ",".join(valid_tool_parsers)
        frontend_kwargs["tool_call_parser"]["metavar"] = (
            f"{{{parsers_str}}} or name registered in --tool-parser-plugin"
        )
        return frontend_kwargs

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Register CLI arguments for this frontend class.

        Subclasses should override ``_customize_cli_kwargs`` instead of
        this method so that base-class postprocessing is always applied.
        """
        from irwkv.engine.arg_utils import get_kwargs

        frontend_kwargs = get_kwargs(cls)
        frontend_kwargs = cls._customize_cli_kwargs(frontend_kwargs)

        group_name = cls.__name__.replace("Args", "")
        frontend_group = parser.add_argument_group(
            title=group_name,
            description=cls.__doc__,
        )
        for key, value in frontend_kwargs.items():
            extra_flags = value.pop("flags", [])
            frontend_group.add_argument(
                *extra_flags, f"--{key.replace('_', '-')}", **value
            )

        return parser


@config
class FrontendArgs(BaseFrontendArgs):
    """Arguments for the OpenAI-compatible frontend server."""

    host: str | None = None
    """Host name."""
    port: int = 8000
    """Port number."""
    grpc_port: int | None = None
    """Enable the Rust frontend's additional gRPC Inference and Control services
    on this port. Requires `IRWKV_USE_RUST_FRONTEND=1 irwkv serve`; HTTP remains on
    `--port`. Binds on `--host`, or 127.0.0.1 with `--uds`."""
    data_parallel_supervisor_port: int = 9256
    """HTTP port for aggregated health endpoints in multi-port external LB
    mode."""
    dp_supervisor_probe_interval_s: float = 5.0
    """Seconds between aggregated health probes in multi-port external LB mode."""
    dp_supervisor_probe_timeout_s: float = 5.0
    """Seconds to wait between retries when a child health probe fails with a
    connection error in multi-port external LB mode."""
    dp_supervisor_probe_failure_threshold: int = 3
    """Number of consecutive connection-error retries before a child health
    probe is declared failed in multi-port external LB mode."""
    uds: str | None = None
    """Unix domain socket path. If set, host and port arguments are ignored."""
    uvicorn_log_level: Literal[
        "critical", "error", "warning", "info", "debug", "trace"
    ] = "info"
    """Log level for uvicorn."""
    disable_uvicorn_access_log: bool = False
    """Disable uvicorn access log."""
    disable_access_log_for_endpoints: str | None = None
    """Comma-separated list of endpoint paths to exclude from uvicorn access
    logs. This is useful to reduce log noise from high-frequency endpoints
    like health checks. Example: "/health,/metrics,/ping".
    When set, access logs for requests to these paths will be suppressed
    while keeping logs for other endpoints."""
    allow_credentials: bool = False
    """Allow credentials."""
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    """Allowed origins."""
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    """Allowed methods."""
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    """Allowed headers."""
    api_key: list[str] | None = None
    """If provided, the server will require one of these keys to be presented in
    the header.

    Warning: this only authenticates endpoints under the `/v1`, `/v2`, and
    `/inference` path prefixes. Other endpoints on the same server, including
    `/invocations` (which exposes the same inference capabilities as `/v1`),
    remain unauthenticated. Do not rely on `--api-key` alone to secure irwkv;
    see
    https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations
    for what it does and does not protect."""
    ssl_keyfile: str | None = None
    """The file path to the SSL key file."""
    ssl_certfile: str | None = None
    """The file path to the SSL cert file."""
    ssl_ca_certs: str | None = None
    """The CA certificates file."""
    enable_ssl_refresh: bool = False
    """Refresh SSL Context when SSL certificate files change"""
    ssl_cert_reqs: int = int(ssl.CERT_NONE)
    """Whether client certificate is required (see stdlib ssl module's)."""
    ssl_ciphers: str | None = None
    """SSL cipher suites for HTTPS (TLS 1.2 and below only).
    Example: 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305'"""
    root_path: str | None = None
    """FastAPI root_path when app is behind a path based routing proxy."""
    middleware: list[str] = field(default_factory=lambda: [])
    """Additional ASGI middleware to apply to the app. We accept multiple
    --middleware arguments. The value should be an import path. If a function
    is provided, irwkv will add it to the server using
    `@app.middleware('http')`. If a class is provided, irwkv will
    add it to the server using `app.add_middleware()`."""
    enable_request_id_headers: bool = False
    """If specified, API server will add X-Request-Id header to responses."""
    disable_fastapi_docs: bool = False
    """Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."""
    h11_max_incomplete_event_size: int = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    """Maximum size (bytes) of an incomplete HTTP event (header or body) for
    h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB)."""
    h11_max_header_count: int = H11_MAX_HEADER_COUNT_DEFAULT
    """Maximum number of HTTP headers allowed in a request for h11 parser.
    Helps mitigate header abuse. Default: 256."""
    enable_offline_docs: bool = False
    """
    Enable offline FastAPI documentation for air-gapped environments.
    Uses vendored static assets bundled with irwkv.
    """

    @classmethod
    def _customize_cli_kwargs(
        cls,
        frontend_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        frontend_kwargs = super()._customize_cli_kwargs(frontend_kwargs)

        # Special case: allowed_origins, allowed_methods, allowed_headers all
        # need json.loads type
        # Should also remove nargs
        frontend_kwargs["allowed_origins"]["type"] = json.loads
        frontend_kwargs["allowed_methods"]["type"] = json.loads
        frontend_kwargs["allowed_headers"]["type"] = json.loads
        del frontend_kwargs["allowed_origins"]["nargs"]
        del frontend_kwargs["allowed_methods"]["nargs"]
        del frontend_kwargs["allowed_headers"]["nargs"]

        # Special case: Middleware needs to append action
        frontend_kwargs["middleware"]["action"] = "append"
        frontend_kwargs["middleware"]["type"] = str
        if "nargs" in frontend_kwargs["middleware"]:
            del frontend_kwargs["middleware"]["nargs"]
        frontend_kwargs["middleware"]["default"] = []

        # Special case: disable_access_log_for_endpoints is a single
        # comma-separated string, not a list
        if "nargs" in frontend_kwargs["disable_access_log_for_endpoints"]:
            del frontend_kwargs["disable_access_log_for_endpoints"]["nargs"]

        return frontend_kwargs


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server.

    We rely on the helper methods of `FrontendArgs` and `AsyncEngineArgs` to
    register all arguments instead of manually enumerating them here. This
    avoids code duplication and keeps the argument definitions in one place.
    """
    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=None,
        help="How many API server processes to run. "
        "Defaults to data_parallel_size if not specified.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "The YAML keys correspond to the serve command options.",
    )
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


def validate_grpc_port_arg(args: argparse.Namespace) -> None:
    """Validate the Rust frontend's optional gRPC listener."""
    if getattr(args, "grpc_port", None) is None:
        return

    if not envs.IRWKV_USE_RUST_FRONTEND or getattr(args, "subparser", None) != "serve":
        raise ValueError("--grpc-port requires IRWKV_USE_RUST_FRONTEND=1 irwkv serve")
    if args.headless or (
        args.api_server_count is not None and args.api_server_count <= 0
    ):
        raise ValueError(
            "--grpc-port requires a Rust frontend; remove --headless "
            "and non-positive --api-server-count"
        )
    if args.data_parallel_multi_port_external_lb:
        raise ValueError(
            "--grpc-port is incompatible with "
            "--data-parallel-multi-port-external-lb: "
            "its frontends would share the same gRPC port"
        )
    if not 0 <= args.grpc_port <= 65535:
        raise ValueError("--grpc-port must be between 0 and 65535")
    if args.grpc_port != 0 and args.grpc_port == args.port and not args.uds:
        raise ValueError("--grpc-port must differ from --port when HTTP uses TCP")


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "serve":
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)
    if args.chat_template is not None and "hf" in (
        args.tool_call_parser,
        getattr(args, "reasoning_parser", None),
    ):
        logger.warning(
            "--chat-template is set; the hf parser still expects "
            "the checkpoint's output format."
        )

    validate_grpc_port_arg(args)

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
    if args.enable_log_outputs and not args.enable_log_requests:
        raise TypeError("Error: --enable-log-outputs requires --enable-log-requests")

    # SSE keep-alive interval must be zero (disabled) or a positive integer.
    if getattr(args, "sse_keep_alive_interval", 0) < 0:
        raise ValueError(
            "Error: --sse-keep-alive-interval must be a non-negative integer "
            "(0 disables keep-alive comments)."
        )

    if getattr(args, "enable_per_request_metrics", False) and getattr(
        args, "disable_log_stats", False
    ):
        raise ValueError(
            "Error: --enable-per-request-metrics requires engine statistics "
            "logging; remove --disable-log-stats to enable per-request metrics."
        )

    if args.data_parallel_multi_port_external_lb:
        from irwkv.entrypoints.launchers.dp_supervisor import (
            validate_multi_port_external_lb_args,
        )

        validate_multi_port_external_lb_args(args)


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m irwkv.entrypoints.launchers.api_server.entry"
    )
    return make_arg_parser(parser_for_docs)
