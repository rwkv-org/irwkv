# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from irwkv.benchmarks.throughput import add_cli_args, main
from irwkv.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from irwkv.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkThroughputSubcommand(BenchmarkSubcommandBase):
    """The `throughput` subcommand for `irwkv bench`."""

    name = "throughput"
    help = "Benchmark offline inference throughput."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
