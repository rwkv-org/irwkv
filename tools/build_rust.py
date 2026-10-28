# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension
from setuptools_scm import get_version

ROOT_DIR = Path(__file__).resolve().parents[1]
IRWKV_RS_BUILD_VERSION = "IRWKV_RS_BUILD_VERSION"

# Select irwkv release tags, excluding crate tags such as "proto-v0.3.0".
IRWKV_GIT_DESCRIBE_COMMAND = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--abbrev=40",
    "--match",
    "v[0-9]*",
]


def prepare_build_environment() -> str | None:
    """Set the device-independent irwkv source version for Rust artifacts."""
    version = os.getenv(IRWKV_RS_BUILD_VERSION) or None
    if version is None:
        try:
            version = get_version(
                root=ROOT_DIR,
                git_describe_command=IRWKV_GIT_DESCRIBE_COMMAND,
            )
        except LookupError:
            return None

    os.environ[IRWKV_RS_BUILD_VERSION] = version
    return version


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    return [
        RustExtension(
            target="irwkv.irwkv-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "irwkv-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
    ]


def build_binary(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "irwkv").mkdir(exist_ok=True)
    setup(
        name="irwkv-rust-frontend-build",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    prepare_build_environment()
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
