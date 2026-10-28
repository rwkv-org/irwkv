# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from irwkv.engine.arg_utils import EngineArgs
from irwkv.usage.usage_lib import UsageContext
from irwkv.utils.argparse_utils import FlexibleArgumentParser
from irwkv.utils.hashing import _xxhash
from tests.v1.core.utils import _scheduler_model_dir


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.set_defaults(model=_scheduler_model_dir().name, skip_tokenizer_init=True)
    args = parser.parse_args([])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not irwkv_config.cache_config.enable_prefix_caching
    assert irwkv_config.cache_config.prefix_cache_retention_interval == 0

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not irwkv_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert irwkv_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])

    args = parser.parse_args(["--prefix-cache-retention-interval", "64"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.prefix_cache_retention_interval == 64


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.set_defaults(model=_scheduler_model_dir().name, skip_tokenizer_init=True)

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_custom_histogram_buckets_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.set_defaults(model=_scheduler_model_dir().name, skip_tokenizer_init=True)

    # Unset: byte-identical default behavior.
    args = parser.parse_args([])
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.observability_config.custom_histogram_buckets is None

    # A JSON mapping round-trips into the config verbatim.
    args = parser.parse_args(
        [
            "--custom-histogram-buckets",
            '{"request_latency": [0.01, 0.05, 0.1, 0.5]}',
        ]
    )
    irwkv_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert irwkv_config.observability_config.custom_histogram_buckets == {
        "request_latency": [0.01, 0.05, 0.1, 0.5]
    }

    # Malformed JSON is rejected at argument-parsing time.
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        parser.parse_args(["--custom-histogram-buckets", "{not json"])

    # An unknown family key parses as JSON but fails config validation.
    args = parser.parse_args(["--custom-histogram-buckets", '{"bogus": [1.0, 2.0]}'])
    with pytest.raises(ValueError, match="unknown bucket family"):
        EngineArgs.from_cli_args(args=args).create_engine_config()


def test_data_parallel_start_rank_zero_infers_hybrid_lb():
    """An explicit --data-parallel-start-rank 0 must be treated the same as
    any other explicit start rank when inferring hybrid LB mode, not as
    "unset" (regression test for a truthiness-vs-`is not None` bug).
    """
    engine_args = EngineArgs(
        model=_scheduler_model_dir().name,
        skip_tokenizer_init=True,
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_start_rank=0,
    )
    irwkv_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert irwkv_config.parallel_config.data_parallel_hybrid_lb is True
    assert irwkv_config.parallel_config.data_parallel_rank == 0
