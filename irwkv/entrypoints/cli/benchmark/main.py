# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import sys
import typing

from irwkv import envs
from irwkv.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from irwkv.entrypoints.cli.types import CLISubcommand
from irwkv.entrypoints.serve.utils.api_utils import IRWKV_SUBCMD_PARSER_EPILOG
from irwkv.logger import init_logger

if typing.TYPE_CHECKING:
    from irwkv.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser

logger = init_logger(__name__)


def maybe_exec_rust_bench() -> None:
    if sys.argv[1:3] != ["bench", "serve"] or not envs.IRWKV_USE_RUST_BENCH:
        return

    rust_cli = envs.IRWKV_RUST_FRONTEND_PATH
    if rust_cli is None:
        raise RuntimeError(
            "IRWKV_USE_RUST_BENCH=1 requires IRWKV_RUST_FRONTEND_PATH "
            "to resolve to the irwkv-rs binary."
        )

    logger.info("Delegating `irwkv bench serve` to Rust binary at %s.", rust_cli)
    os.execv(rust_cli, [rust_cli, "bench", "serve", *sys.argv[3:]])


def _import_bench_subcommand_modules() -> None:
    # Imported lazily so `BenchmarkSubcommandBase` subclasses register only
    # when `irwkv bench` is actually invoked.
    import irwkv.entrypoints.cli.benchmark.latency  # noqa: F401
    import irwkv.entrypoints.cli.benchmark.serve  # noqa: F401
    import irwkv.entrypoints.cli.benchmark.startup  # noqa: F401
    import irwkv.entrypoints.cli.benchmark.sweep  # noqa: F401
    import irwkv.entrypoints.cli.benchmark.throughput  # noqa: F401


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the irwkv CLI."""

    name = "bench"
    help = "irwkv bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.help,
            usage=f"irwkv {self.name} <bench_type> [options]",
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        # Only build the nested bench subparsers when the user is actually
        # invoking `bench`; otherwise we'd drag in imports
        # unnecessarily on every `irwkv --help` and `irwkv serve`.
        # Scan for the first positional arg so global flags (e.g. `-v`)
        # before the subcommand don't break detection.
        first_positional = next(
            (arg for arg in sys.argv[1:] if not arg.startswith("-")), None
        )
        if first_positional == self.name:
            _import_bench_subcommand_modules()
            for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
                cmd_subparser = bench_subparsers.add_parser(
                    cmd_cls.name,
                    help=cmd_cls.help,
                    description=cmd_cls.help,
                    usage=f"irwkv {self.name} {cmd_cls.name} [options]",
                )
                cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
                cmd_cls.add_cli_args(cmd_subparser)
                cmd_subparser.epilog = IRWKV_SUBCMD_PARSER_EPILOG.format(
                    subcmd=f"{self.name} {cmd_cls.name}"
                )
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
