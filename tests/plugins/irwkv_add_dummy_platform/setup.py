# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="irwkv_add_dummy_platform",
    version="0.1",
    packages=["irwkv_add_dummy_platform"],
    entry_points={
        "irwkv.platform_plugins": [
            "dummy_platform_plugin = irwkv_add_dummy_platform:dummy_platform_plugin"  # noqa
        ],
        "irwkv.general_plugins": [
            "dummy_custom_ops = irwkv_add_dummy_platform:register_ops"
        ],
    },
)
