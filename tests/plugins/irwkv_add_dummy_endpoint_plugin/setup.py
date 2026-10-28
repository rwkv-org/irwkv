# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="irwkv_add_dummy_endpoint_plugin",
    version="0.1",
    packages=["irwkv_add_dummy_endpoint_plugin"],
    entry_points={
        "irwkv.endpoint_plugins": [
            "dummy_admin_endpoint_plugin = irwkv_add_dummy_endpoint_plugin:DummyAdminEndpointPlugin"  # noqa
        ]
    },
)
