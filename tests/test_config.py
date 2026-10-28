# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import MISSING, Field, dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pydantic
import pytest
import torch
from huggingface_hub import ResolvedRevision
from pydantic import ValidationError

import irwkv.config.irwkv as irwkv_config_module
import irwkv.envs as envs
from irwkv.compilation.backends import IrwkvBackend
from irwkv.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    IrwkvConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    WatermarkConfig,
    update_config,
)
from irwkv.config.compilation import CompilationMode, CUDAGraphMode
from irwkv.config.irwkv import OPTIMIZATION_LEVEL_TO_CONFIG, OptimizationLevel
from irwkv.config.kernel import IrOpPriorityConfig
from irwkv.config.load import LoadConfig
from irwkv.config.utils import get_field
from irwkv.platforms import current_platform
from irwkv.sampling_params import BeamSearchParams
from irwkv.v1.attention.backend import AttentionCGSupport

DEVICE_TYPE = current_platform.device_type


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific test")
@pytest.mark.parametrize(
    ("is_mm_prefix_lm", "is_multimodal_model", "expected"),
    [
        pytest.param(True, True, True, id="multimodal-prefix-lm"),
        pytest.param(False, True, False, id="multimodal-causal"),
        pytest.param(True, False, False, id="text-prefix-lm"),
        pytest.param(None, True, False, id="missing-model-config"),
    ],
)
def test_rocm_mm_prefix_lm_disables_chunked_mm_input(
    is_mm_prefix_lm: bool | None,
    is_multimodal_model: bool,
    expected: bool,
) -> None:
    from irwkv.platforms.rocm import RocmPlatform

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            worker_cls="test-worker",
        ),
        model_config=(
            None
            if is_mm_prefix_lm is None
            else SimpleNamespace(is_mm_prefix_lm=is_mm_prefix_lm)
        ),
        scheduler_config=SimpleNamespace(
            is_multimodal_model=is_multimodal_model,
            disable_chunked_mm_input=False,
        ),
    )

    RocmPlatform.check_and_update_config(config)

    assert config.scheduler_config.disable_chunked_mm_input is expected


def _sampling_replay_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            return_sampling_mask=True,
            logprobs_mode=overrides.pop("logprobs_mode", "processed_logprobs"),
            logits_processors=overrides.pop("logits_processors", None),
        )
    )
    return config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_sampling_replay_config(), None),
        (
            _sampling_replay_config(logits_processors=["custom"]),
            "does not support custom logits processors",
        ),
        (
            _sampling_replay_config(logprobs_mode="raw_logprobs"),
            "requires logprobs_mode='processed_logprobs'",
        ),
    ],
)
def test_sampling_replay_config(config, message):
    if message is None:
        IrwkvConfig._verify_sampling_replay_config(config)
    else:
        with pytest.raises(ValueError, match=message):
            IrwkvConfig._verify_sampling_replay_config(config)


def test_mamba_cache_mode_all_is_rejected():
    """The removed 'all' mode must fail validation instead of being ignored."""
    with pytest.raises(ValidationError, match="mamba_cache_mode"):
        CacheConfig(mamba_cache_mode="all")


def test_compile_config_repr_succeeds():
    # setup: IrwkvBackend mutates the config object
    config = IrwkvConfig()
    backend = IrwkvBackend(config)
    backend.configure_post_pass()

    # test that repr(config) succeeds
    val = repr(config)
    assert "IrwkvConfig" in val
    assert "inductor_passes" in val


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, None),
        ("0", False),
        ("1", True),
    ],
)
def test_v2_model_runner_env_tri_state(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("IRWKV_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("IRWKV_USE_V2_MODEL_RUNNER", env_value)

    assert envs.IRWKV_USE_V2_MODEL_RUNNER is expected


def test_v2_model_runner_supports_custom_logits_processors():
    config = IrwkvConfig()
    config.model_config = cast(
        ModelConfig, SimpleNamespace(logits_processors=["a.b:C"])
    )

    assert config._get_v2_model_runner_unsupported_features() == []


@pytest.mark.parametrize(
    ("mode", "piecewise_capture_available", "attention_support", "expected"),
    [
        ("PIECEWISE", False, "ALWAYS", "NONE"),
        ("FULL_AND_PIECEWISE", False, "ALWAYS", "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "ALWAYS", "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "NEVER", "NONE"),
        ("FULL_AND_PIECEWISE", True, "UNIFORM_BATCH", "FULL_AND_PIECEWISE"),
        ("FULL", True, "UNIFORM_BATCH", "FULL_DECODE_ONLY"),
    ],
)
def test_resolve_cudagraph_mode_uses_loaded_piecewise_provider(
    mode, piecewise_capture_available, attention_support, expected
):
    compilation_config = CompilationConfig(
        mode=CompilationMode.IRWKV_COMPILE,
        cudagraph_mode=CUDAGraphMode[mode],
        use_inductor_graph_partition=True,
    )

    resolved = compilation_config.resolve_cudagraph_mode_and_sizes(
        AttentionCGSupport[attention_support],
        "FakeAttentionBackend",
        piecewise_capture_available=piecewise_capture_available,
    )

    assert resolved.name == expected
    assert compilation_config.cudagraph_mode == resolved


def test_resolve_cudagraph_mode_skips_mamba_block_check_while_profiling():
    """Cudagraph memory profiling uses a minimal KV cache, so the Mamba
    block-count guard must only fire for the real cache sizing."""
    kv_cache_config = SimpleNamespace(has_mamba_layers=True, num_blocks=4)

    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    with pytest.raises(ValueError, match="exceeds available Mamba cache blocks"):
        compilation_config.resolve_cudagraph_mode_and_sizes(
            AttentionCGSupport.ALWAYS,
            "FakeAttentionBackend",
            kv_cache_config=kv_cache_config,
            max_num_reqs=256,
        )

    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    cudagraph_mode = compilation_config.resolve_cudagraph_mode_and_sizes(
        AttentionCGSupport.ALWAYS,
        "FakeAttentionBackend",
        kv_cache_config=kv_cache_config,
        max_num_reqs=256,
        is_profiling=True,
    )
    assert cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_batch_sharded_sampling_rejects_return_sampling_mask():
    """The batch-sharded gather drops sampling masks, so the combination must
    fail loudly instead of returning ``sampling_mask=None``."""
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_batch_sharded_sampling=True, tensor_parallel_size=2
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(max_logprobs=20, return_sampling_mask=True),
    )

    with pytest.raises(ValueError, match="sampling masks"):
        IrwkvConfig._validate_batch_sharded_sampling(config)


@pytest.mark.skip_global_cleanup
def test_with_hf_config_populates_missing_architectures_from_causal_lm_mapping(
    monkeypatch,
):
    monkeypatch.setattr(
        irwkv_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(model_type="mistral", architectures=None)

    updated = IrwkvConfig.with_hf_config(cfg, hf_config)

    assert updated.model_config.hf_config.architectures == ["MistralForCausalLM"]
    assert hf_config.architectures is None


@pytest.mark.skip_global_cleanup
def test_with_hf_config_preserves_explicit_architectures_override(monkeypatch):
    monkeypatch.setattr(
        irwkv_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(model_type="mistral", architectures=None)

    updated = IrwkvConfig.with_hf_config(
        cfg,
        hf_config,
        architectures=["Ministral3ForCausalLM"],
    )

    assert updated.model_config.hf_config.architectures == ["Ministral3ForCausalLM"]


@pytest.mark.skip_global_cleanup
def test_with_hf_config_leaves_unknown_model_type_without_architectures(
    monkeypatch,
):
    monkeypatch.setattr(
        irwkv_config_module,
        "replace",
        lambda self, **kwargs: SimpleNamespace(**kwargs),
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(),
            get_model_arch_config=lambda: "arch-config",
        )
    )
    hf_config = SimpleNamespace(
        model_type="not_a_real_model",
        architectures=None,
    )

    updated = IrwkvConfig.with_hf_config(cfg, hf_config)

    assert updated.model_config.hf_config.architectures is None


@pytest.mark.parametrize(
    "checkpoint_tensors,tied",
    [
        # The checkpoint has an lm_head of its own, so it must win over the config
        (["model.embed_tokens.weight", "lm_head.weight"], False),
        (["model.embed_tokens.weight"], True),
        # Contents unknown (not safetensors), so the config must be left alone
        ([], True),
    ],
)
def test_maybe_untie_word_embeddings(tmp_path, checkpoint_tensors, tied):
    import torch
    from safetensors.torch import save_file

    if checkpoint_tensors:
        save_file(
            {name: torch.zeros(2, 2) for name in checkpoint_tensors},
            tmp_path / "model.safetensors",
        )

    text_config = SimpleNamespace(tie_word_embeddings=True)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        hf_config=SimpleNamespace(
            tie_word_embeddings=True,
            get_text_config=lambda: text_config,
        ),
        word_embeddings_untied_by_checkpoint=False,
    )

    ModelConfig.maybe_untie_word_embeddings(model_config)

    # Both levels must agree, since different callers read different ones
    assert model_config.hf_config.tie_word_embeddings is tied
    assert text_config.tie_word_embeddings is tied
    assert model_config.word_embeddings_untied_by_checkpoint is not tied


def test_async_scheduling_with_pipeline_parallelism_is_allowed():
    cfg = IrwkvConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            async_scheduling=True,
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            nnodes=2,
        ),
    )
    assert cfg.scheduler_config.async_scheduling is True


def test_data_parallel_rpc_port_has_fixed_default():
    assert ParallelConfig().data_parallel_rpc_port == 29550


def test_dp_group_uses_configured_timeout_without_current_config(monkeypatch):
    monkeypatch.setattr(irwkv_config_module, "_current_irwkv_config", None)
    config = ParallelConfig(cpu_distributed_timeout_seconds=30)
    with (
        patch(
            "irwkv.distributed.utils.rendezvous",
            return_value=iter([(torch.distributed.HashStore(), 0, 1)]),
        ),
        patch("irwkv.distributed.utils.init_gloo_process_group") as init_group,
    ):
        config.stateless_init_dp_group()
    assert init_group.call_args.kwargs["timeout"] == timedelta(seconds=30)


@pytest.mark.parametrize("port", [1, 29550, 65535])
def test_data_parallel_rpc_port_accepts_valid_ports(port: int):
    assert ParallelConfig(data_parallel_rpc_port=port).data_parallel_rpc_port == port


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_data_parallel_rpc_port_rejects_invalid_ports(port: int):
    with pytest.raises(ValidationError):
        ParallelConfig(data_parallel_rpc_port=port)


def test_reconfigure_for_independent_dp_rank_on_multinode_dense_model():
    parallel_config = ParallelConfig(
        tensor_parallel_size=8,
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=1,
        distributed_executor_backend="mp",
        nnodes=2,
        node_rank=1,
    )

    assert parallel_config.nnodes_within_dp == 1
    assert parallel_config.node_rank_within_dp == 0

    parallel_config.reconfigure_for_independent_dp_rank()

    assert parallel_config.data_parallel_size == 1
    assert parallel_config.data_parallel_size_local == 1
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_index == 1
    assert parallel_config.nnodes == 1
    assert parallel_config.node_rank == 0
    assert parallel_config.world_size == 8


@pytest.mark.parametrize("data_parallel_size", [4, 8])
def test_nnodes_within_dp_when_replicas_outnumber_nodes(data_parallel_size):
    """A replica that fits on one node spans one node, never zero.

    External LB pins ``data_parallel_size_local`` to 1, so the ratio rounds
    down to 0 as soon as there are more DP replicas than nodes. Anything
    dividing by ``nnodes_within_dp`` then raises ZeroDivisionError.
    """
    parallel_config = ParallelConfig(
        data_parallel_size=data_parallel_size,
        data_parallel_size_local=1,
        data_parallel_external_lb=True,
        distributed_executor_backend="mp",
        nnodes=2,
        node_rank=1,
    )

    assert parallel_config.nnodes_within_dp == 1
    assert parallel_config.node_rank_within_dp == 0
    assert parallel_config.local_world_size == parallel_config.world_size


@pytest.mark.parametrize("nnodes", [4, 9])
def test_nnodes_within_dp_rejects_uneven_internal_lb(nnodes):
    parallel_config = ParallelConfig(
        data_parallel_size=8,
        data_parallel_size_local=1,
        distributed_executor_backend="mp",
        nnodes=nnodes,
    )

    with pytest.raises(ValueError, match="Invalid data parallel configuration"):
        _ = parallel_config.nnodes_within_dp


@dataclass
class _TestConfigFields:
    a: int
    b: dict = field(default_factory=dict)
    c: str = "default"


def test_get_field():
    b = get_field(_TestConfigFields, "b")
    assert isinstance(b, Field)
    assert b.default is MISSING
    assert b.default_factory is dict

    c = get_field(_TestConfigFields, "c")
    assert isinstance(c, Field)
    assert c.default == "default"
    assert c.default_factory is MISSING


@dataclass
class _TestNestedConfig:
    a: _TestConfigFields = field(default_factory=lambda: _TestConfigFields(a=0))


@dataclass
class _TestDerivedConfigFields(_TestConfigFields):
    pass


def test_update_config():
    # Simple update
    config1 = _TestConfigFields(a=0)
    new_config1 = update_config(config1, {"a": 42})
    assert new_config1.a == 42
    # Nonexistent field
    with pytest.raises(ValueError, match=r"_TestConfigFields\.nonexistent"):
        new_config1 = update_config(config1, {"nonexistent": 1})
    # Nested update with dataclass
    config2 = _TestNestedConfig()
    new_inner_config = _TestConfigFields(a=1, c="new_value")
    new_config2 = update_config(config2, {"a": new_inner_config})
    assert new_config2.a == new_inner_config
    # Declared field type, not the live value's subtype, defines valid overrides
    config_with_derived = _TestNestedConfig(a=_TestDerivedConfigFields(a=0))
    new_config2 = update_config(config_with_derived, {"a": new_inner_config})
    assert new_config2.a is new_inner_config
    # Nested update with unrelated dataclass
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a"):
        update_config(config2, {"a": _TestNestedConfig()})
    # Nested update with dict
    config3 = _TestNestedConfig()
    new_config3 = update_config(config3, {"a": {"c": "new_value"}})
    assert new_config3.a.c == "new_value"
    # Nested update with invalid type
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a"):
        update_config(config3, {"a": "new_value"})
    # Invalid nested field preserves its full path
    with pytest.raises(ValueError, match=r"_TestNestedConfig\.a\.nonexistent"):
        update_config(config3, {"a": {"nonexistent": 1}})


MODEL_IDS_EXPECTED = [
    ("Qwen/Qwen1.5-7B", 32768),
    ("mistralai/Mistral-7B-v0.1", 4096),
    ("mistralai/Mistral-7B-Instruct-v0.2", 32768),
]


@pytest.mark.parametrize(
    "pt_load_map_location",
    [
        DEVICE_TYPE,
        {"": DEVICE_TYPE},
    ],
)
def test_load_config_pt_load_map_location(pt_load_map_location):
    load_config = LoadConfig(pt_load_map_location=pt_load_map_location)
    config = IrwkvConfig(load_config=load_config)

    assert config.load_config.pt_load_map_location == pt_load_map_location


class MockConfig:
    """Simple mock object for testing maybe_pull_model_tokenizer_for_runai."""

    def __init__(self, model: str, tokenizer: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_weights = None


@pytest.mark.parametrize(
    "s3_url",
    [
        "s3://example-bucket-1/model/",
        "s3://example-bucket-2/model/",
    ],
)
@patch("irwkv.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_model_tokenizer_paths(mock_pull_files, s3_url):
    """Test that S3 URLs create deterministic local directories for model and
    tokenizer."""
    # Mock pull_files to avoid actually downloading files during tests
    mock_pull_files.return_value = None

    # Create first mock and run the method
    config1 = MockConfig(model=s3_url, tokenizer=s3_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config1, s3_url, s3_url)

    # Check that model and tokenizer point to existing directories
    assert os.path.exists(config1.model), (
        f"Model directory does not exist: {config1.model}"
    )
    assert os.path.isdir(config1.model), (
        f"Model path is not a directory: {config1.model}"
    )
    assert os.path.exists(config1.tokenizer), (
        f"Tokenizer directory does not exist: {config1.tokenizer}"
    )
    assert os.path.isdir(config1.tokenizer), (
        f"Tokenizer path is not a directory: {config1.tokenizer}"
    )

    # Verify that the paths are different from the original S3 URL
    assert config1.model != s3_url, "Model path should be converted to local directory"
    assert config1.tokenizer != s3_url, (
        "Tokenizer path should be converted to local directory"
    )

    # Store the original paths
    created_model_dir = config1.model
    create_tokenizer_dir = config1.tokenizer

    # Create a new mock and run the method with the same S3 URL
    config2 = MockConfig(model=s3_url, tokenizer=s3_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config2, s3_url, s3_url)

    # Check that the new directories exist
    assert os.path.exists(config2.model), (
        f"Model directory does not exist: {config2.model}"
    )
    assert os.path.isdir(config2.model), (
        f"Model path is not a directory: {config2.model}"
    )
    assert os.path.exists(config2.tokenizer), (
        f"Tokenizer directory does not exist: {config2.tokenizer}"
    )
    assert os.path.isdir(config2.tokenizer), (
        f"Tokenizer path is not a directory: {config2.tokenizer}"
    )

    # Verify that the paths are deterministic (same as before)
    assert config2.model == created_model_dir, (
        f"Model paths are not deterministic. "
        f"Original: {created_model_dir}, New: {config2.model}"
    )
    assert config2.tokenizer == create_tokenizer_dir, (
        f"Tokenizer paths are not deterministic. "
        f"Original: {create_tokenizer_dir}, New: {config2.tokenizer}"
    )


@patch("irwkv.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_different_models_create_different_directories(mock_pull_files):
    """Test that different S3 URLs create different local directories."""
    # Mock pull_files to avoid actually downloading files during tests
    mock_pull_files.return_value = None

    s3_url1 = "s3://example-bucket-1/model/"
    s3_url2 = "s3://example-bucket-2/model/"

    # Create mocks with different S3 URLs and run the method
    config1 = MockConfig(model=s3_url1, tokenizer=s3_url1)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config1, s3_url1, s3_url1)

    config2 = MockConfig(model=s3_url2, tokenizer=s3_url2)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config2, s3_url2, s3_url2)

    # Verify that different URLs produce different directories
    assert config1.model != config2.model, (
        f"Different S3 URLs should create different model directories. "
        f"URL1 model: {config1.model}, URL2 model: {config2.model}"
    )
    assert config1.tokenizer != config2.tokenizer, (
        f"Different S3 URLs should create different tokenizer directories. "
        f"URL1 tokenizer: {config1.tokenizer}, "
        f"URL2 tokenizer: {config2.tokenizer}"
    )

    # Verify that both sets of directories exist
    assert os.path.exists(config1.model) and os.path.isdir(config1.model)
    assert os.path.exists(config1.tokenizer) and os.path.isdir(config1.tokenizer)
    assert os.path.exists(config2.model) and os.path.isdir(config2.model)
    assert os.path.exists(config2.tokenizer) and os.path.isdir(config2.tokenizer)


@patch("irwkv.transformers_utils.runai_utils.ObjectStorageModel.pull_files")
def test_s3_url_different_model_and_tokenizer(mock_pull_files):
    """Test that when model and tokenizer are different cloud URIs,
    pull_files receives the correct URI for each."""
    mock_pull_files.return_value = None

    model_url = "s3://bucket/model/"
    tokenizer_url = "s3://bucket/tokenizer/"

    config = MockConfig(model=model_url, tokenizer=tokenizer_url)
    ModelConfig.maybe_pull_model_tokenizer_for_runai(config, model_url, tokenizer_url)

    # pull_files should be called twice: once for model, once for tokenizer
    assert mock_pull_files.call_count == 2
    # First call: model URI with allow_pattern
    assert mock_pull_files.call_args_list[0][0][0] == model_url
    # Second call: tokenizer URI with ignore_pattern
    assert mock_pull_files.call_args_list[1][0][0] == tokenizer_url


@pytest.mark.parametrize(
    ("backend", "custom_ops", "expected"),
    [
        ("eager", [], True),
        ("eager", ["+fused_layernorm"], True),
        ("eager", ["all", "-fused_layernorm"], False),
        ("inductor", [], False),
        ("inductor", ["none", "+fused_layernorm"], True),
        ("inductor", ["none", "-fused_layernorm"], False),
    ],
)
def test_is_custom_op_enabled(backend: str, custom_ops: list[str], expected: bool):
    """Test that is_custom_op_enabled works correctly."""
    config = IrwkvConfig(
        compilation_config=CompilationConfig(backend=backend, custom_ops=custom_ops)
    )
    assert config.compilation_config.is_custom_op_enabled("fused_layernorm") is expected


def test_irwkv_config_defaults_are_none():
    """Verify that optimization-level defaults are None when not set by user."""
    # Test all optimization levels to ensure defaults work correctly
    for opt_level in OptimizationLevel:
        config = object.__new__(IrwkvConfig)
        config.compilation_config = CompilationConfig()
        config.optimization_level = opt_level
        config.model_config = None

        # Use the global optimization level defaults
        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[opt_level]

        # Verify that all pass_config values are None before defaults are applied
        for pass_k in default_config["compilation_config"]["pass_config"]:
            assert getattr(config.compilation_config.pass_config, pass_k) is None

        # Verify that other config values are None before defaults are applied
        for k in default_config["compilation_config"]:
            if k != "pass_config":
                assert getattr(config.compilation_config, k) is None


def test_validate_mamba_align_subblock_prefill():
    """Align mode permits configured prefill chunks smaller than a block."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=11392,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            long_prefill_token_threshold=4096,
            disable_chunked_mm_input=False,
        ),
        kv_transfer_config=None,
    )

    IrwkvConfig.validate_block_size(config)


@pytest.mark.parametrize("optimization_level", list(OptimizationLevel))
@pytest.mark.parametrize("compilation_config", [CompilationConfig()])
@pytest.mark.parametrize("model_id", [None])
def test_irwkv_config_defaults(model_id, compilation_config, optimization_level):
    """Test that optimization-level defaults are correctly applied."""
    model_config = None
    if model_id is not None:
        model_config = ModelConfig(model_id)
        irwkv_config = IrwkvConfig(
            model_config=model_config,
            compilation_config=compilation_config,
            optimization_level=optimization_level,
        )
    else:
        irwkv_config = IrwkvConfig(
            compilation_config=compilation_config,
            optimization_level=optimization_level,
        )
    # Use the global optimization level defaults
    default_config = OPTIMIZATION_LEVEL_TO_CONFIG[optimization_level]

    # Verify pass_config defaults (nested under compilation_config)
    pass_config_dict = default_config["compilation_config"]["pass_config"]
    for pass_k, pass_v in pass_config_dict.items():
        actual = getattr(irwkv_config.compilation_config.pass_config, pass_k)
        expected = pass_v(irwkv_config) if callable(pass_v) else pass_v
        assert actual == expected, (
            f"pass_config.{pass_k}: expected {expected}, got {actual}"
        )

    # Verify other compilation_config defaults
    compilation_config_dict = default_config["compilation_config"]
    for k, v in compilation_config_dict.items():
        if k == "pass_config":
            continue
        actual = getattr(irwkv_config.compilation_config, k)
        expected = v(irwkv_config) if callable(v) else v
        # On platforms without static graph support, __post_init__ forces
        # cudagraph_mode to NONE; expect that instead of the level default.
        if k == "cudagraph_mode" and (
            model_id is None or not current_platform.support_static_graph_mode()
        ):
            expected = CUDAGraphMode.NONE
        assert actual == expected, (
            f"compilation_config.{k}: expected {expected}, got {actual}"
        )


def test_scheduler_config_init():
    with pytest.raises(ValidationError):
        # Positional InitVars missing
        # (InitVars cannot have defaults otherwise they will become attributes)
        SchedulerConfig()

    with pytest.raises(AttributeError):
        # InitVar does not become an attribute
        print(SchedulerConfig.default_factory().max_model_len)


def test_scheduler_config_rejects_zero_max_num_scheduled_tokens():
    """A zero token budget never schedules anything, so generation would hang."""
    with pytest.raises(ValidationError, match="max_num_scheduled_tokens"):
        SchedulerConfig.default_factory(max_num_scheduled_tokens=0)


def _watermarked_irwkv_config() -> IrwkvConfig:
    config = object.__new__(IrwkvConfig)
    config.watermark_config = WatermarkConfig(key=42)
    config.attention_config = AttentionConfig()
    return config


def test_gumbel_watermark_disables_beam_search_watermarking(caplog_irwkv):
    enabled = _watermarked_irwkv_config()._check_supports_watermarking(
        BeamSearchParams(beam_width=2, max_tokens=1)
    )

    assert not enabled
    assert "beam search requests will run without watermarking" in caplog_irwkv.text


def test_gumbel_watermark_allows_unwatermarked_beam_search():
    _watermarked_irwkv_config()._check_supports_watermarking(
        BeamSearchParams(beam_width=2, max_tokens=1, watermarking=False)
    )


def test_gumbel_watermark_rejects_custom_sampler():
    with pytest.raises(ValueError, match="custom samplers are not supported"):
        _watermarked_irwkv_config()._check_supports_watermarking(custom_sampler=True)


def test_watermark_key_must_fit_in_64_bits():
    with pytest.raises(ValueError, match="64 bits"):
        WatermarkConfig(key=2**64)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_watermark_alpha_must_be_a_probability(alpha):
    with pytest.raises(ValidationError):
        WatermarkConfig(key=42, algorithm="dual_key_gumbel", alpha=alpha)


def test_unknown_watermark_prf_is_rejected():
    with pytest.raises(ValidationError):
        pydantic.TypeAdapter(WatermarkConfig).validate_python(
            {"key": 42, "prf": "unsupported"}
        )


def test_watermark_key_is_excluded_from_serialization():
    serialized = pydantic.TypeAdapter(WatermarkConfig).dump_python(
        WatermarkConfig(key=42), mode="json"
    )

    assert "key" not in serialized


def test_ir_op_priority_default():
    """Test that IR op priority defaults are set correctly."""
    # Assert default is applied to ops
    priority_config = IrOpPriorityConfig.with_default(["irwkv_c", "native"])
    assert priority_config.rms_norm == ["irwkv_c", "native"]
    assert priority_config.fused_add_rms_norm == ["irwkv_c", "native"]

    # Assert single ops override the default
    priority_config = IrOpPriorityConfig.with_default(
        ["native"], rms_norm=["oink", "native"]
    )
    assert priority_config.rms_norm == ["oink", "native"]
    assert priority_config.fused_add_rms_norm == ["native"]


def test_ir_op_priority_str():
    """Test that passing a comma-delimited string works."""
    priority_config = IrOpPriorityConfig(rms_norm="irwkv_c")
    assert priority_config.rms_norm == ["irwkv_c"]

    priority_config = IrOpPriorityConfig(rms_norm="irwkv_c,native")
    assert priority_config.rms_norm == ["irwkv_c", "native"]

    priority_config = IrOpPriorityConfig(rms_norm=" native, irwkv_c ")
    assert priority_config.rms_norm == ["native", "irwkv_c"]

    with pytest.raises(pydantic.ValidationError):
        # must be list of only strings
        priority_config = IrOpPriorityConfig(rms_norm=["irwkv_c", 4, "native"])


def test_ir_op_priority_ctx(monkeypatch):
    """Test that the priority-setting context sets priority correctly."""
    from irwkv import ir

    current_platform.import_ir_kernels()
    monkeypatch.setattr(ir.ops.rms_norm.impls["irwkv_c"], "supported", True)
    monkeypatch.setattr(ir.ops.fused_add_rms_norm.impls["irwkv_c"], "supported", True)

    priority = IrOpPriorityConfig.with_default(["native"], rms_norm=["irwkv_c"])
    priority2 = IrOpPriorityConfig.with_default(
        ["native"], fused_add_rms_norm=["irwkv_c"]
    )
    with priority.set_priority():
        assert ir.ops.rms_norm.get_priority() == ["irwkv_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]
        with priority2.set_priority():
            assert ir.ops.rms_norm.get_priority() == ["native"]
            assert ir.ops.fused_add_rms_norm.get_priority() == ["irwkv_c", "native"]

        # context restored
        assert ir.ops.rms_norm.get_priority() == ["irwkv_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]

        with pytest.raises(ValueError), priority2.set_priority():
            assert ir.ops.rms_norm.get_priority() == ["native"]
            assert ir.ops.fused_add_rms_norm.get_priority() == ["irwkv_c", "native"]

            raise ValueError

        # context restored even after exception
        assert ir.ops.rms_norm.get_priority() == ["irwkv_c", "native"]
        assert ir.ops.fused_add_rms_norm.get_priority() == ["native"]


def test_load_config_rejects_invalid_safetensors_load_strategy():
    with pytest.raises(pydantic.ValidationError):
        LoadConfig(safetensors_load_strategy="not_a_real_strategy")


@pytest.mark.parametrize("bad_load_format", [None, 123])
def test_load_config_rejects_non_string_load_format(bad_load_format):
    with pytest.raises(pydantic.ValidationError):
        LoadConfig(load_format=bad_load_format)


# A fixed revision used to verify resolved revision propagation.
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@patch("irwkv.config.model.resolve_revision", return_value=ResolvedRevision(REVISION))
def test_revision_resolved_for_model(mock_resolve):
    model = "rwkv-rs/rwkv7-g1j-1.5b"
    config = ModelConfig(model, hf_config_path=os.environ.get("IRWKV_RWKV_TEST_MODEL"))
    assert isinstance(config.revision, ResolvedRevision)
    assert config.revision.resolved == REVISION
    mock_resolve.assert_any_call(model, None, config.hf_token)
