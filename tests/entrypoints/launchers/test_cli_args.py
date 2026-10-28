# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.engine.arg_utils import AsyncEngineArgs
from irwkv.entrypoints.launchers.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from irwkv.exceptions import IRWKVValidationError
from irwkv.utils.argparse_utils import FlexibleArgumentParser


def _build_irwkv_parsers():
    irwkv_parser = FlexibleArgumentParser()
    subparsers = irwkv_parser.add_subparsers(dest="subparser")
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return {"irwkv": irwkv_parser, "irwkv serve": serve_parser}


@pytest.fixture
def irwkv_parser():
    return _build_irwkv_parsers()["irwkv"]


@pytest.fixture
def serve_parser():
    return _build_irwkv_parsers()["irwkv serve"]


### Test config parsing
def test_config_arg_parsing(serve_parser, cli_config_file):
    args = serve_parser.parse_args([])
    assert args.port == 8000
    args = serve_parser.parse_args(["--config", cli_config_file])
    assert args.port == 12312
    args = serve_parser.parse_args(
        [
            "--config",
            cli_config_file,
            "--port",
            "9000",
        ]
    )
    assert args.port == 9000
    args = serve_parser.parse_args(
        [
            "--port",
            "9000",
            "--config",
            cli_config_file,
        ]
    )
    assert args.port == 9000


def test_logging_config_cli_args(serve_parser):
    with pytest.warns(UserWarning, match="--log-config-file is deprecated"):
        args = serve_parser.parse_args(
            [
                "--logging-config",
                '{"log_level":"WARNING","pylogging_config_file":"/tmp/json.json"}',
                "--logging-config.formatter",
                "json",
                "--log-level",
                "DEBUG",
                "--log-config-file",
                "/tmp/flat.json",
            ]
        )

    config = AsyncEngineArgs.from_cli_args(args).create_logging_config()
    assert config.log_level == "DEBUG"
    assert config.formatter == "json"
    assert config.pylogging_config_file == "/tmp/flat.json"


### Tests for serve argument validation that run prior to loading
@pytest.mark.parametrize(
    "port, extra_args",
    [
        (50051, []),
        (0, ["--port", "0"]),
        (50051, ["--port", "50051", "--uds", "/tmp/irwkv.sock"]),
    ],
)
def test_rust_grpc_port_accepts_frontend_launch(
    irwkv_parser, monkeypatch, port, extra_args
):
    monkeypatch.setenv("IRWKV_USE_RUST_FRONTEND", "1")
    args = irwkv_parser.parse_args(
        [
            "serve",
            "--grpc-port",
            str(port),
            "--data-parallel-size",
            "8",
            "--data-parallel-size-local",
            "4",
            "--data-parallel-start-rank",
            "4",
            "--data-parallel-hybrid-lb",
            *extra_args,
        ]
    )
    validate_parsed_serve_args(args)
    assert args.grpc_port == port


@pytest.mark.parametrize(
    "rust_enabled, extra_args, port",
    [
        (False, [], 50051),
        (True, ["--headless"], 50051),
        (True, ["--api-server-count", "0"], 50051),
        (True, ["--api-server-count", "-1"], 50051),
        (True, ["--data-parallel-multi-port-external-lb"], 50051),
        (True, ["--port", "50051"], 50051),
        (True, [], -1),
        (True, [], 65536),
    ],
)
def test_rust_grpc_port_rejects_incompatible_launch(
    irwkv_parser, monkeypatch, rust_enabled, extra_args, port
):
    monkeypatch.setenv("IRWKV_USE_RUST_FRONTEND", "1" if rust_enabled else "0")
    args = irwkv_parser.parse_args(["serve", "--grpc-port", str(port), *extra_args])
    with pytest.raises(ValueError, match="--grpc-port"):
        validate_parsed_serve_args(args)


def test_rust_grpc_port_requires_serve_subcommand(monkeypatch):
    """Direct Python render/API entrypoints must reject a Rust-only listener."""
    monkeypatch.setenv("IRWKV_USE_RUST_FRONTEND", "1")
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(["--grpc-port", "50051"])
    with pytest.raises(ValueError, match="--grpc-port requires"):
        validate_parsed_serve_args(args)


def test_enable_auto_choice_fails_without_tool_call_parser(serve_parser):
    """Ensure validation fails if tool choice is enabled with no call parser."""
    # If we enable-auto-tool-choice, explode with no tool-call-parser
    args = serve_parser.parse_args(args=["--enable-auto-tool-choice"])
    with pytest.raises(TypeError):
        validate_parsed_serve_args(args)


def test_enable_auto_choice_passes_with_tool_call_parser(serve_parser):
    """Ensure validation passes with tool choice enabled with a call parser."""
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "rwkv",
        ]
    )
    validate_parsed_serve_args(args)


def test_auto_tool_choice_with_reasoning_parser_passes(serve_parser):
    """Auto tool choice combined with a reasoning parser is supported.

    Both parsers can be enabled together.
    """
    args = serve_parser.parse_args(
        args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "rwkv",
            "--reasoning-parser",
            "rwkv",
        ]
    )
    validate_parsed_serve_args(args)


def test_passes_with_reasoning_parser(serve_parser):
    """Ensure validation passes if reasoning is enabled
    with a reasoning parser"""
    args = serve_parser.parse_args(
        args=[
            "--reasoning-parser",
            "rwkv",
        ]
    )
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_happy_paths(serve_parser, tmp_path):
    """Ensure validation passes if the chat template exists."""
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}")
    args = serve_parser.parse_args(args=["--chat-template", template.as_posix()])
    validate_parsed_serve_args(args)


def test_chat_template_validation_for_sad_paths(serve_parser):
    """Ensure validation fails if the chat template doesn't exist."""
    args = serve_parser.parse_args(args=["--chat-template", "does/not/exist"])
    with pytest.raises(IRWKVValidationError):
        validate_parsed_serve_args(args)


def test_per_request_metrics_requires_log_stats(serve_parser):
    args = serve_parser.parse_args(
        args=["--enable-per-request-metrics", "--disable-log-stats"]
    )
    with pytest.raises(ValueError):
        validate_parsed_serve_args(args)


def test_subcommand_without_serve_args_skips_validation():
    """A subcommand that does not use the serve parser is still skipped"""
    irwkv_parser = FlexibleArgumentParser()
    subparsers = irwkv_parser.add_subparsers(required=False, dest="subparser")
    subparsers.add_parser("chat")
    args = irwkv_parser.parse_args(args=["chat"])

    validate_parsed_serve_args(args)


@pytest.mark.parametrize(
    "cli_args, expected",
    [
        ([], 0),
        (["--sse-keep-alive-interval", "0"], 0),
        (["--sse-keep-alive-interval", "15"], 15),
    ],
)
def test_sse_keep_alive_interval_valid(serve_parser, cli_args, expected):
    """The default is 0 and zero or positive integers parse exactly."""
    args = serve_parser.parse_args(args=cli_args)
    assert args.sse_keep_alive_interval == expected
    validate_parsed_serve_args(args)


@pytest.mark.parametrize("value", ["-1", "-5"])
def test_sse_keep_alive_interval_negative(serve_parser, value):
    """Negative intervals are rejected by validation."""
    args = serve_parser.parse_args(args=["--sse-keep-alive-interval", value])
    with pytest.raises(ValueError):
        validate_parsed_serve_args(args)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0.5", "abc"])
def test_sse_keep_alive_interval_non_integer(serve_parser, value):
    """Non-integer values fail argparse's int parsing with SystemExit."""
    with pytest.raises(SystemExit):
        serve_parser.parse_args(args=["--sse-keep-alive-interval", value])


@pytest.mark.parametrize(
    "cli_args, expected_middleware",
    [
        (
            ["--middleware", "middleware1", "--middleware", "middleware2"],
            ["middleware1", "middleware2"],
        ),
        ([], []),
    ],
)
def test_middleware(serve_parser, cli_args, expected_middleware):
    """Ensure multiple middleware args are parsed properly."""
    args = serve_parser.parse_args(args=cli_args)
    assert args.middleware == expected_middleware


def test_default_chat_template_kwargs_parsing(serve_parser):
    """Ensure default_chat_template_kwargs JSON is parsed correctly."""
    args = serve_parser.parse_args(
        args=["--default-chat-template-kwargs", '{"enable_thinking": false}']
    )
    assert args.default_chat_template_kwargs == {"enable_thinking": False}


def test_default_chat_template_kwargs_complex(serve_parser):
    """Ensure complex default_chat_template_kwargs JSON is parsed correctly."""
    kwargs_json = '{"enable_thinking": false, "custom_param": "value", "num": 42}'
    args = serve_parser.parse_args(args=["--default-chat-template-kwargs", kwargs_json])
    assert args.default_chat_template_kwargs == {
        "enable_thinking": False,
        "custom_param": "value",
        "num": 42,
    }


def test_default_chat_template_kwargs_default_none(serve_parser):
    """Ensure default_chat_template_kwargs defaults to None."""
    args = serve_parser.parse_args(args=[])
    assert args.default_chat_template_kwargs is None


def test_default_chat_template_kwargs_invalid_json(serve_parser):
    """Ensure invalid JSON raises an error."""
    with pytest.raises(SystemExit):
        serve_parser.parse_args(
            args=["--default-chat-template-kwargs", "not valid json"]
        )


@pytest.mark.parametrize(
    "args, raises",
    [
        (["user/model"], None),
        (["user/model", "--served-model-name", "model"], None),
        (["--served-model-name", "model", "user/model"], ValueError),
        (["--served-model-name", "model", "--config", "config.yaml"], None),
        (["--served-model-name", "model", "--config", "config.yaml"], ValueError),
    ],
    ids=[
        "model_tag_only",
        "model_tag_with_served_model_name",
        "served_model_name_before_model_tag",
        "served_model_name_with_model_in_config",
        "served_model_name_with_no_model_in_config",
    ],
)
def test_served_model_name_parsing(tmp_path, irwkv_parser, args, raises):
    """Ensure that users don't misuse --served-model-name and end up with the default
    model tag instead of the one they intended to serve."""
    # Call the serve subparser
    args.insert(0, "serve")
    # Create a dummy config file if the test case includes it
    if "config.yaml" in args:
        # Create a dummy config file if the test case includes it
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: user/model" if raises is None else "port: 8000")
        args[args.index("config.yaml")] = config_path.as_posix()
    # Do the parsing and check for expected exceptions or values
    if raises is None:
        parsed_args = irwkv_parser.parse_args(args=args)
        expected = "user/model"
        assert parsed_args.model_tag == expected or parsed_args.model == expected
    else:
        with pytest.raises(raises):
            irwkv_parser.parse_args(args=args)
