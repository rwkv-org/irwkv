# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from irwkv.benchmarks.latency import add_cli_args, main
from irwkv.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from irwkv.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    """The `latency` subcommand for `irwkv bench`."""

    name = "latency"
    help = "Benchmark the latency of a single batch of requests."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
