# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of irwkv.

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from irwkv.logger import configure_logging_from_args, init_logger

logger = init_logger(__name__)


def main():
    import irwkv.entrypoints.cli.benchmark.main
    import irwkv.entrypoints.cli.serve
    from irwkv.entrypoints.serve.utils.api_utils import (
        IRWKV_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from irwkv.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        irwkv.entrypoints.cli.serve,
        irwkv.entrypoints.cli.benchmark.main,
    ]

    cli_env_setup()

    irwkv.entrypoints.cli.benchmark.main.maybe_exec_rust_bench()

    # For 'irwkv bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from irwkv import platforms

        if platforms.current_platform.is_unspecified():
            from irwkv.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="irwkv CLI",
        epilog=IRWKV_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("irwkv"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmd = cmds[args.subparser]
        configure_logging_from_args(args)
        cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
