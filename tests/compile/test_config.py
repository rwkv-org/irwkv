# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from irwkv.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from irwkv.config import (
    CompilationConfig,
    CUDAGraphMode,
    IrwkvConfig,
    ParallelConfig,
    SchedulerConfig,
)
from irwkv.config.compilation import CompilationMode, PassConfig
from irwkv.platforms import current_platform
from irwkv.utils.torch_utils import (
    _is_torch_equal_or_newer,
    is_torch_equal,
)
from irwkv.v1.cudagraph_dispatcher import CudagraphDispatcher

# This import automatically registers `torch.ops.silly.attention`
from . import silly_attention  # noqa: F401

DEVICE_TYPE = current_platform.device_type


def test_version():
    # Test the version comparison logic using the private function
    assert _is_torch_equal_or_newer("2.8.0.dev20250624+cu128", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0a0+gitc82a174", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.0", "2.8.0.dev")
    assert _is_torch_equal_or_newer("2.8.1", "2.8.0.dev")
    assert not _is_torch_equal_or_newer("2.7.1", "2.8.0.dev")


def test_get_raw_stream_patch():
    """Test that get_raw_stream patch is applied only for torch 2.9.0 or 2.9.1."""
    import builtins

    # Check if get_raw_stream exists in builtins
    has_patch = hasattr(builtins, "get_raw_stream")

    # Import torch to get actual version

    is_torch_2_9 = is_torch_equal("2.9.0") or is_torch_equal("2.9.1")

    if is_torch_2_9:
        # For torch 2.9.x, the patch should be applied
        assert has_patch, "get_raw_stream should be patched for torch 2.9.x"
        # Verify it's callable (it should be the _cuda_getCurrentRawStream function)
        get_raw_stream = builtins.get_raw_stream  # type: ignore[attr-defined]
        assert callable(get_raw_stream)
        # Verify it's the correct function from torch._C
        from torch._C import _cuda_getCurrentRawStream

        assert get_raw_stream is _cuda_getCurrentRawStream


# Inductor can initialize CUDA even for CPU inputs.
@pytest.mark.forked
@pytest.mark.parametrize("use_v2", [False, True])
def test_e8m0_custom_op_fullgraph(use_v2, use_fresh_inductor_cache):
    """E8M0 inputs must not prevent mutable custom-op decomposition."""
    with torch.library._scoped_library("test_e8m0", "FRAGMENT") as lib:
        lib.define("scale_(Tensor(a!) out, Tensor scale) -> ()")

        @torch.library.impl(lib, "scale_", "CPU")
        def scale_impl(out, scale):
            out.mul_(scale.float())

        @torch.library.register_fake("test_e8m0::scale_", lib=lib)
        def scale_fake(out, scale):
            return None

        def forward(x, scale):
            out = x.clone()
            torch.ops.test_e8m0.scale_(out, scale)
            return out

        x = torch.randn(4, 4)
        scale = torch.full((4,), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        compiled = torch.compile(
            forward,
            fullgraph=True,
            options={"enable_auto_functionalized_v2": use_v2},
        )
        torch.testing.assert_close(compiled(x, scale), x * 2)


def test_copy_pass():
    irwkv_config = IrwkvConfig()
    inductor_pass = NoOpEliminationPass(irwkv_config)
    copied_inductor_pass = copy.deepcopy(inductor_pass)
    assert (
        copied_inductor_pass.compilation_config.use_inductor_graph_partition
        == irwkv_config.compilation_config.use_inductor_graph_partition
    )
    assert (
        copied_inductor_pass.compilation_config.splitting_ops
        == irwkv_config.compilation_config.splitting_ops
    )


def test_custom_op():
    # proper syntax
    _ = CompilationConfig(custom_ops=["+quant_fp8", "-silu_and_mul"])
    _ = CompilationConfig(custom_ops=["none", "+rms_norm"])
    _ = CompilationConfig(custom_ops=["+rms_norm", "+rms_norm"])

    for custom_ops in (["quant_fp8"], ["+"], ["-"]):
        with pytest.raises(ValueError, match="Invalid syntax '"):
            CompilationConfig(custom_ops=custom_ops)


@pytest.mark.parametrize(
    ("custom_ops", "config_kwargs", "match"),
    [
        (["all", "none"], {}, "can contain only one base mode"),
        (
            ["none", "+rms_norm", "-rms_norm"],
            {},
            "cannot both enable and disable.*rms_norm",
        ),
    ],
)
def test_reject_contradictory_custom_ops(custom_ops, config_kwargs, match):
    with pytest.raises(ValueError, match=match):
        CompilationConfig(custom_ops=custom_ops, **config_kwargs)


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073


# forked needed to workaround https://github.com/vllm-project/vllm/issues/21073


def test_splitting_ops_dynamic():
    # Default config
    config = IrwkvConfig()
    # Default V1 config leaves cudagraph mode unset; splitting ops are only
    # populated when the engine decides to use piecewise compilation.
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
    assert config.compilation_config.splitting_ops_contain_attention()
    splitting_ops = config.compilation_config.splitting_ops
    assert splitting_ops is not None
    assert {
        "irwkv::unified_attention_with_output",
    } <= set(splitting_ops)

    # When use_inductor_graph_partition=True
    config = IrwkvConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.IRWKV_COMPILE,
            use_inductor_graph_partition=True,
            splitting_ops=["irwkv::unified_attention_with_output"],
        )
    )
    # with inductor partition we use splitting_ops directly for
    # partition rules
    assert config.compilation_config.splitting_ops == [
        "irwkv::unified_attention_with_output"
    ]


def test_moe_splitting_ops_deepep_ht_inductor_partition():
    # Inductor partition case: user-provided splitting_ops should be
    # preserved and MoE ops should be appended for DeepEP HT with dp>1.
    config = IrwkvConfig(
        parallel_config=ParallelConfig(
            all2all_backend="deepep_high_throughput",
            data_parallel_size=8,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.IRWKV_COMPILE,
            use_inductor_graph_partition=True,
            splitting_ops=[
                "irwkv::unified_attention_with_output",
                "irwkv::moe_forward",
                "irwkv::moe_forward_shared",
            ],
        ),
    )
    splitting_ops = config.compilation_config.splitting_ops
    assert splitting_ops == [
        "irwkv::unified_attention_with_output",
        "irwkv::moe_forward",
        "irwkv::moe_forward_shared",
    ]


def test_should_split():
    import torch

    from irwkv.compilation.partition_rules import should_split

    graph = torch.fx.Graph()
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.aten.add.default,
        args=(),
        kwargs={},
    )

    # supports OpOverloadPacket
    splitting_ops = ["aten::add"]
    assert should_split(node, splitting_ops)

    # supports OpOverload
    splitting_ops = ["aten::add.default"]
    assert should_split(node, splitting_ops)

    # supports OpOverload
    splitting_ops = ["aten::add.Tensor"]
    assert not should_split(node, splitting_ops)

    q, k, v, out = [torch.randn(1)] * 4

    # supports custom ops as OpOverloadPacket
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.silly.attention,
        args=(q, k, v, out),
        kwargs={},
    )

    splitting_ops = ["silly::attention"]
    assert should_split(node, splitting_ops)

    # supports custom ops as OpOverload
    node = torch.fx.Node(
        graph=graph,
        name="dummy_node",
        op="call_function",
        target=torch.ops.silly.attention.default,
        args=(q, k, v, out),
        kwargs={},
    )

    splitting_ops = ["silly::attention"]
    assert should_split(node, splitting_ops)

    splitting_ops = ["silly::attention.default"]
    assert should_split(node, splitting_ops)


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if not cudagraph mode supported",
)
@pytest.mark.parametrize(
    (
        "cudagraph_capture_sizes",
        "max_cudagraph_capture_size",
        "max_num_batched_tokens",
        "cudagraph_mode",
        "expected_max_size",
    ),
    [
        (None, None, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        ([1, 2, 4], 4, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        (
            [1, 2, 4],
            8,
            2048,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            ValueError,
        ),
        ([1, 256], None, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        ([], None, 2048, CUDAGraphMode.NONE, 0),
        (None, 0, 2048, CUDAGraphMode.NONE, 0),
        # truncated to nearest multiple of 8 or 16
        (None, 257, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 256),
        # max_num_batched_tokens <= max_cudagraph_capture_size should always be
        # captured even if not landing on a 16-stride step
        (None, 2048, 257, CUDAGraphMode.FULL_AND_PIECEWISE, 257),
        # max from list
        ([1, 2, 4, 15], None, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, 15),
        # limited by the max_tokens
        ([1, 2, 4, 15], None, 8, CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        # the list should contain at least 1 element when use cudagraph
        ([], None, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, AssertionError),
        # the max capturing size should be >= 1 when use cudagraph
        (None, 0, 2048, CUDAGraphMode.FULL_AND_PIECEWISE, AssertionError),
    ],
)
def test_cudagraph_sizes_post_init(
    cudagraph_capture_sizes,
    max_cudagraph_capture_size,
    max_num_batched_tokens,
    cudagraph_mode,
    expected_max_size,
):
    ctx = nullcontext()
    if expected_max_size in (ValueError, AssertionError):
        ctx = pytest.raises(expected_max_size)

    with ctx:
        kwargs = {}
        if cudagraph_capture_sizes is not None:
            kwargs["cudagraph_capture_sizes"] = cudagraph_capture_sizes
        if max_cudagraph_capture_size is not None:
            kwargs["max_cudagraph_capture_size"] = max_cudagraph_capture_size
        compilation_config = CompilationConfig(
            pass_config=PassConfig(eliminate_noops=True),
            cudagraph_mode=cudagraph_mode,
            **kwargs,
        )
        irwkv_config = _mock_config_for_cudagraph_sizes(
            max_num_seqs=min(max_num_batched_tokens, 128),
            max_num_batched_tokens=max_num_batched_tokens,
            compilation_config=compilation_config,
        )
        IrwkvConfig._set_cudagraph_sizes(irwkv_config)

        assert (
            irwkv_config.compilation_config.max_cudagraph_capture_size
            == expected_max_size
        )


def _mock_config_for_cudagraph_sizes(
    max_num_seqs: int,
    max_num_batched_tokens: int,
    compilation_config: CompilationConfig,
) -> MagicMock:
    """Mock IrwkvConfig wired up enough to run `_set_cudagraph_sizes`."""
    config = MagicMock(spec=IrwkvConfig)
    config.compilation_config = compilation_config
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    config.parallel_config = ParallelConfig()
    config.model_config = MagicMock()
    config.model_config.enforce_eager = False
    config.performance_mode = None
    config.uniform_decode_query_len = IrwkvConfig.uniform_decode_query_len.fget(config)
    return config


@pytest.mark.parametrize("max_num_seqs", [100, 101])
def test_default_cudagraph_capture_sizes_cover_off_stride_max_num_seqs(
    max_num_seqs: int,
) -> None:
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    config = _mock_config_for_cudagraph_sizes(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=32768,
        compilation_config=compilation_config,
    )

    IrwkvConfig._set_cudagraph_sizes(config)

    assert max_num_seqs in compilation_config.cudagraph_capture_sizes


@pytest.mark.parametrize(
    ("max_num_seqs", "widest_is_captured"),
    [
        (8, True),
        (32, True),
        (64, True),
        (256, True),
        (512, True),
        (1025, False),
    ],
)
def test_default_cudagraph_capture_size_respects_platform_ceiling(
    max_num_seqs, widest_is_captured
):
    """Uniform decode sizes stay within the platform capture ceiling."""
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    config = _mock_config_for_cudagraph_sizes(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=32768,
        compilation_config=compilation_config,
    )

    IrwkvConfig._set_cudagraph_sizes(config)

    decode_query_len = 1
    default_max_graph_size = (
        1024 if current_platform.is_device_capability_family(100) else 512
    )
    token_grid_max = min(
        max_num_seqs * decode_query_len * 2,
        default_max_graph_size,
    )
    widest_uniform_decode = max_num_seqs * decode_query_len
    assert compilation_config.max_cudagraph_capture_size == token_grid_max
    assert (
        compilation_config.max_cudagraph_capture_size
        == compilation_config.cudagraph_capture_sizes[-1]
    )
    assert (
        widest_uniform_decode in compilation_config.cudagraph_capture_sizes
    ) is widest_is_captured


def test_default_cudagraph_capture_size_caps_tokens():
    """The default capture range stays under the platform memory ceiling."""
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    config = _mock_config_for_cudagraph_sizes(
        max_num_seqs=1024,
        max_num_batched_tokens=1_000_000,
        compilation_config=compilation_config,
    )

    IrwkvConfig._set_cudagraph_sizes(config)

    default_max_graph_size = (
        1024 if current_platform.is_device_capability_family(100) else 512
    )
    assert compilation_config.max_cudagraph_capture_size == default_max_graph_size
    assert (
        compilation_config.max_cudagraph_capture_size
        == compilation_config.cudagraph_capture_sizes[-1]
    )
    assert all(
        size <= default_max_graph_size
        for size in compilation_config.cudagraph_capture_sizes
    )


def test_default_cudagraph_capture_size_still_clamped_by_token_budget():
    """Decode coverage does not override the `max_num_batched_tokens` clamp.

    A batch wider than the token budget cannot be scheduled in the first place,
    so there is no decode step of that size to capture a graph for.
    """
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    config = _mock_config_for_cudagraph_sizes(
        max_num_seqs=128,
        max_num_batched_tokens=192,
        compilation_config=compilation_config,
    )

    IrwkvConfig._set_cudagraph_sizes(config)

    assert compilation_config.max_cudagraph_capture_size == 192
    assert (
        compilation_config.max_cudagraph_capture_size
        == compilation_config.cudagraph_capture_sizes[-1]
    )
    assert 544 not in compilation_config.cudagraph_capture_sizes


@pytest.mark.skipif(
    not current_platform.support_static_graph_mode(),
    reason="Skip if static graph mode is not supported",
)
@pytest.mark.parametrize(
    ("is_blackwell", "expected_max_size"), [(False, 512), (True, 1024)]
)
def test_blackwell_cudagraph_default(is_blackwell, expected_max_size):
    irwkv_config = IrwkvConfig()
    irwkv_config.model_config = MagicMock(enforce_eager=False)
    irwkv_config.scheduler_config = SchedulerConfig(
        max_num_seqs=512,
        max_num_batched_tokens=2048,
        max_model_len=2048,
        is_encoder_decoder=False,
    )
    irwkv_config.compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    with patch.object(
        current_platform,
        "is_device_capability_family",
        return_value=is_blackwell,
    ):
        irwkv_config._set_cudagraph_sizes()

    assert (
        irwkv_config.compilation_config.max_cudagraph_capture_size == expected_max_size
    )


def _create_irwkv_config_for_validation(
    compilation_config: CompilationConfig,
) -> MagicMock:
    """Helper to create a mock IrwkvConfig for padding validation testing."""
    mock_config = MagicMock(spec=IrwkvConfig)
    mock_config.compilation_config = compilation_config
    mock_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
    mock_config.parallel_config = ParallelConfig()
    return mock_config


def test_compile_sizes_padding_validation():
    """Test that compile_sizes with values that would be padded raises an error."""
    # cudagraph_capture_sizes=[1, 2, 4, 8] means:
    # - size 1 -> padded to 1
    # - size 2 -> padded to 2
    # - size 3 -> padded to 4
    # - size 4 -> padded to 4
    # - size 5 -> padded to 8
    # etc.
    # So compile_sizes=[3] should fail because 3 would be padded to 4

    with pytest.raises(ValueError, match="would be padded to"):
        config = CompilationConfig(
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_cudagraph_capture_size=8,
            compile_sizes=[3],
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        config.post_init_cudagraph_sizes()
        dispatcher = CudagraphDispatcher(_create_irwkv_config_for_validation(config))
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)

    with pytest.raises(ValueError, match="would be padded to"):
        config = CompilationConfig(
            cudagraph_capture_sizes=[1, 2, 4, 8],
            max_cudagraph_capture_size=8,
            compile_sizes=[5],
            cudagraph_mode=CUDAGraphMode.FULL,
        )
        config.post_init_cudagraph_sizes()
        dispatcher = CudagraphDispatcher(_create_irwkv_config_for_validation(config))
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)

    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        compile_sizes=[1, 2, 4, 8],
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [1, 2, 4, 8]
    dispatcher = CudagraphDispatcher(_create_irwkv_config_for_validation(config))
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL)  # Should not raise

    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        compile_sizes=["cudagraph_capture_sizes"],
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [1, 2, 4, 8]

    # When cudagraphs are disabled (max_cudagraph_capture_size=0),
    # padding validation should be skipped
    config = CompilationConfig(
        cudagraph_capture_sizes=[],
        max_cudagraph_capture_size=0,
        compile_sizes=[3, 5, 7],  # would be invalid with cudagraphs
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [3, 5, 7]

    # When cudagraph_mode is NONE but capture_sizes is non-empty,
    # padding validation should still be skipped
    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_cudagraph_capture_size=8,
        cudagraph_mode=CUDAGraphMode.NONE,
        compile_sizes=[3, 5, 7],  # would be invalid if cudagraphs were enabled
    )
    config.post_init_cudagraph_sizes()
    assert sorted(config.compile_sizes) == [3, 5, 7]
    dispatcher = CudagraphDispatcher(_create_irwkv_config_for_validation(config))
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)  # Should not raise


def test_inductor_asserts_default_disabled(monkeypatch):
    """Test that inductor runtime asserts are disabled by default
    (INFO logging level) on torch < 2.12."""
    monkeypatch.setenv("IRWKV_LOGGING_LEVEL", "INFO")

    import importlib

    import irwkv.envs

    importlib.reload(irwkv.envs)

    config = CompilationConfig()
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("size_asserts") is False
        assert config.inductor_compile_config.get("alignment_asserts") is False
        assert config.inductor_compile_config.get("scalar_asserts") is False


def test_inductor_asserts_enabled_in_debug(monkeypatch):
    """Test that IRWKV_LOGGING_LEVEL=DEBUG enables inductor runtime asserts
    on torch < 2.12."""
    monkeypatch.setenv("IRWKV_LOGGING_LEVEL", "DEBUG")

    import importlib

    import irwkv.envs

    importlib.reload(irwkv.envs)

    config = CompilationConfig()
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("size_asserts") is True
        assert config.inductor_compile_config.get("alignment_asserts") is True
        assert config.inductor_compile_config.get("scalar_asserts") is True


def test_get_inductor_factors_includes_configs():
    """Changing inductor or functorch config must change the cache key factors."""
    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config

    from irwkv.compilation.compiler_interface import get_inductor_factors

    baseline = get_inductor_factors()

    with inductor_config.patch("max_autotune", not inductor_config.max_autotune):
        patched = get_inductor_factors()
    assert baseline != patched, "inductor config change was not reflected"

    with functorch_config.patch("donated_buffer", not functorch_config.donated_buffer):
        patched = get_inductor_factors()
    assert baseline != patched, "functorch config change was not reflected"


def test_inductor_asserts_user_override(monkeypatch):
    """Test that explicit inductor_compile_config overrides the
    debug-logging default."""
    monkeypatch.setenv("IRWKV_LOGGING_LEVEL", "INFO")

    import importlib

    import irwkv.envs

    importlib.reload(irwkv.envs)

    config = CompilationConfig(
        inductor_compile_config={"size_asserts": True},
    )
    assert config.inductor_compile_config.get("size_asserts") is True
    if not _is_torch_equal_or_newer(torch.__version__, "2.12.0.dev"):
        assert config.inductor_compile_config.get("alignment_asserts") is False


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_combo_kernel_benchmarking_respects_deterministic(deterministic, override):
    from torch._inductor import config as inductor_config

    overrides = {} if override is None else {"deterministic": override}
    with (
        inductor_config.patch(deterministic=deterministic),
        patch("irwkv.config.compilation.current_platform.is_cpu", return_value=False),
    ):
        config = CompilationConfig(inductor_compile_config=overrides)

    assert config.inductor_compile_config["combo_kernels"] is True
    effective_deterministic = deterministic if override is None else override
    assert config.inductor_compile_config["benchmark_combo_kernel"] is (
        not effective_deterministic
    )


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"combo_kernels": False},
        {"combo_kernels": True},
        {"benchmark_combo_kernel": False},
        {"benchmark_combo_kernel": True},
    ],
)
def test_combo_kernel_explicit_settings_preserved(deterministic, overrides):
    from torch._inductor import config as inductor_config

    with (
        inductor_config.patch(deterministic=deterministic),
        patch("irwkv.config.compilation.current_platform.is_cpu", return_value=False),
    ):
        config = CompilationConfig(inductor_compile_config=overrides.copy())

    for key in ("combo_kernels", "benchmark_combo_kernel"):
        assert config.inductor_compile_config.get(key) == overrides.get(key)


@pytest.mark.parametrize("is_cpu,torch_version", [(True, "2.13.0"), (False, "2.8.0")])
def test_combo_kernel_defaults_require_supported_platform_and_torch(
    is_cpu, torch_version
):
    with (
        patch("irwkv.config.compilation.current_platform.is_cpu", return_value=is_cpu),
        patch("torch.__version__", torch_version),
    ):
        config = CompilationConfig()

    assert "combo_kernels" not in config.inductor_compile_config
    assert "benchmark_combo_kernel" not in config.inductor_compile_config


def test_combo_kernel_defaults_without_inductor_deterministic_setting():
    with (
        patch("torch._inductor.config", SimpleNamespace()),
        patch("irwkv.config.compilation.current_platform.is_cpu", return_value=False),
    ):
        config = CompilationConfig()

    assert config.inductor_compile_config["combo_kernels"] is True
    assert config.inductor_compile_config["benchmark_combo_kernel"] is True
