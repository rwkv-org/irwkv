# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""These envs only work for a small part of the tests, fix what you need!"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from irwkv.envs import maybe_convert_bool

if TYPE_CHECKING:
    IRWKV_CI_NO_SKIP: bool = False
    IRWKV_CI_DTYPE: str | None = None
    IRWKV_CI_HEAD_DTYPE: str | None = None
    IRWKV_CI_HF_DTYPE: str | None = None

environment_variables: dict[str, Callable[[], Any]] = {
    # A model family has many models with the same architecture.
    # By default, a model family tests only one model.
    # Through this flag, all models can be tested.
    "IRWKV_CI_NO_SKIP": lambda: bool(int(os.getenv("IRWKV_CI_NO_SKIP", "0"))),
    # Allow changing the dtype used by irwkv in tests
    "IRWKV_CI_DTYPE": lambda: os.getenv("IRWKV_CI_DTYPE", None),
    # Allow changing the head dtype used by irwkv in tests
    "IRWKV_CI_HEAD_DTYPE": lambda: os.getenv("IRWKV_CI_HEAD_DTYPE", None),
    # Allow changing the head dtype used by transformers in tests
    "IRWKV_CI_HF_DTYPE": lambda: os.getenv("IRWKV_CI_HF_DTYPE", None),
    # Allow control over whether tests use enforce_eager
    "IRWKV_CI_ENFORCE_EAGER": lambda: maybe_convert_bool(
        os.getenv("IRWKV_CI_ENFORCE_EAGER", None)
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
