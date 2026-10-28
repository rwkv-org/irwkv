# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.plugins import load_general_plugins


def test_platform_plugins():
    # simulate workload by running an example
    import runpy

    current_file = __file__
    import os

    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples",
        "basic/offline_inference/basic.py",
    )
    runpy.run_path(example_file)

    # check if the plugin is loaded correctly
    from irwkv.platforms import _init_trace, current_platform

    assert current_platform.device_name == "DummyDevice", (
        f"Expected DummyDevice, got {current_platform.device_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}"
    )


def test_oot_custom_op(default_irwkv_config, monkeypatch: pytest.MonkeyPatch):
    # simulate workload by running an example
    load_general_plugins()
    from irwkv.model_executor.layers.layernorm import RMSNorm

    layer = RMSNorm(16)
    assert layer.__class__.__name__ == "DummyRMSNorm", (
        f"Expected DummyRMSNorm, got {layer.__class__.__name__}, "
        "possibly because the custom op is not registered correctly."
    )
    assert hasattr(layer, "addition_config"), (
        "Expected DummyRMSNorm to have an 'addition_config' attribute, "
        "which is set by the custom op."
    )
