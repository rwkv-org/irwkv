# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the apply-time Triton kernel override registry.

The registry must let out-of-tree platforms replace kernels defined as
plain ``@triton.jit`` functions, patching every hold-site (defining
module, from-import copies, and JIT warmup owners) so that existing call
sites dispatch to the override without any core-side decoration.
"""

import sys
import types
from typing import Any

import pytest

from irwkv.model_executor.warmup.jit_warmup_triton_helper import (
    _DecoratedTritonJitKernel,
)
from irwkv.triton_utils import tl, triton
from irwkv.triton_utils.dispatcher import (
    KernelOverride,
    register_kernels,
)


class _FakeTritonKernel:
    """Stands in for a Triton JITFunction (works without Triton installed)."""

    arg_names = ("output_ptr", "input_ptr", "replace_from", "MAX_NUM_TOKENS")

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> Any:
            return ("default", grid, kwargs)

        return launch


@pytest.fixture
def _fake_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Expose a fake defining module holding the kernel and its warmup owner."""
    import types

    kernel = _FakeTritonKernel()
    owner = _DecoratedTritonJitKernel(
        kernel,
        lambda: {},
        lambda output, input, replace_from: (  # noqa: ARG005
            (1,),
            dict(MAX_NUM_TOKENS=8),
        ),
    )
    module = types.ModuleType("irwkv.tests.fake_kernel_home")
    module.expand_kernel = kernel  # type: ignore[attr-defined]
    module._expand = owner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_override_patches_defining_module_and_warmup_owner(
    _fake_module: Any,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def cpu_impl(output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS=None):
        calls.append((output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS))
        return "cpu"

    register_kernels(
        {
            "irwkv.tests.fake_kernel_home.expand_kernel": cpu_impl,
        }
    )

    overridden = _fake_module.expand_kernel
    assert isinstance(overridden, KernelOverride)
    # Launch with kernel-arg names like the warmup launch path does.
    assert overridden[(4,)](output_ptr=1, input_ptr=2, replace_from=0) == "cpu"
    assert calls == [(1, 2, 0, None)]

    # The warmup owner's captured kernel is patched too, and its cached
    # argument-name introspection is invalidated.
    assert _fake_module._expand.kernel is overridden


def test_override_forwards_positionally_for_mismatched_names(
    _fake_module: Any,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def cpu_impl(output, input_val, replace_from, MAX_NUM_TOKENS=None):
        calls.append((output, input_val, replace_from, MAX_NUM_TOKENS))
        return "cpu"

    register_kernels(
        {
            "irwkv.tests.fake_kernel_home.expand_kernel": cpu_impl,
        }
    )
    overridden = _fake_module.expand_kernel
    assert overridden[(4,)](output_ptr=1, input_ptr=2, replace_from=0) == "cpu"
    assert calls == [(1, 2, 0, None)]


def test_override_covers_from_import_copies(_fake_module: Any) -> None:
    import types

    importer = types.ModuleType("irwkv.tests.fake_kernel_importer")
    importer.expand_kernel = _fake_module.expand_kernel  # type: ignore[attr-defined]
    sys.modules["irwkv.tests.fake_kernel_importer"] = importer

    def cpu_impl(output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS=None):
        return "cpu"

    register_kernels(
        {
            "irwkv.tests.fake_kernel_home.expand_kernel": cpu_impl,
        }
    )
    assert isinstance(importer.expand_kernel, KernelOverride)
    del sys.modules["irwkv.tests.fake_kernel_importer"]


def test_override_owner_launch_goes_through_impl(_fake_module: Any) -> None:
    launched: list[Any] = []

    def cpu_impl(output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS=None):
        launched.append((output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS))
        return "cpu"

    register_kernels(
        {
            "irwkv.tests.fake_kernel_home.expand_kernel": cpu_impl,
        }
    )
    owner = _fake_module._expand
    # In-place kernels return None from launch; assert on the recorded call.
    owner(1, 2, 0)
    assert launched == [(1, 2, 0, 8)]


def test_register_kernel_auto_imports_defining_module(
    monkeypatch: pytest.MonkeyPatch, _fake_module: Any
) -> None:
    """Load the defining module before applying a platform kernel override."""
    module_name = _fake_module.__name__
    kernel_name = f"{module_name}.expand_kernel"
    monkeypatch.delitem(sys.modules, module_name)
    imports = []

    def import_module(name):
        imports.append(name)
        monkeypatch.setitem(sys.modules, name, _fake_module)
        return _fake_module

    def cpu_impl(output_ptr, input_ptr, replace_from, MAX_NUM_TOKENS=None):
        return "cpu"

    monkeypatch.setattr(
        "irwkv.triton_utils.dispatcher.importlib.import_module", import_module
    )
    register_kernels({kernel_name: cpu_impl})

    assert imports == [module_name]
    overridden = _fake_module.expand_kernel
    assert isinstance(overridden, KernelOverride)
    assert overridden[(2,)](1, 2, 0, MAX_NUM_TOKENS=8) == "cpu"
    assert _fake_module._expand.kernel is overridden


def test_real_triton_kernel_override_end_to_end() -> None:
    """Override a real @triton.jit kernel through the registry."""
    if not triton.__class__.__module__.startswith("triton"):
        pytest.skip("Triton is not installed")

    @triton.jit
    def _add_one_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):  # pragma: no cover
        offsets = tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) + 1)

    module = types.ModuleType("irwkv.tests.real_jit_kernel_home")
    module._add_one_kernel = _add_one_kernel  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module

    def cpu_impl(x_ptr, out_ptr, BLOCK=None):  # noqa: ARG001
        return "cpu"

    register_kernels({"irwkv.tests.real_jit_kernel_home._add_one_kernel": cpu_impl})
    assert module._add_one_kernel.arg_names == (  # type: ignore[attr-defined]
        "x_ptr",
        "out_ptr",
        "BLOCK",
    )
