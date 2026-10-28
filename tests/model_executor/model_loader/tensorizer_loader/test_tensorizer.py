# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import os
import pathlib
import subprocess
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import irwkv.model_executor.model_loader.tensorizer
from irwkv import SamplingParams
from irwkv.engine.arg_utils import EngineArgs
from irwkv.model_executor.model_loader.tensorizer import (
    TensorizerArgs,
    TensorizerConfig,
    TensorSerializer,
    deserialize_tensorizer_model,
    is_irwkv_tensorized,
    open_stream,
    serialize_extra_artifacts,
    tensorize_irwkv_model,
)
from irwkv.model_executor.model_loader.tensorizer_loader import (
    BLACKLISTED_TENSORIZER_ARGS,
)
from irwkv.utils.import_utils import PlaceholderModule
from tests.utils import RemoteOpenAIServer

from .conftest import DummyExecutor, assert_from_collective_rpc

try:
    import tensorizer
    from tensorizer import EncryptionParams
except ImportError:
    tensorizer = PlaceholderModule("tensorizer")
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")


class TensorizerCaughtError(Exception):
    pass


def test_serialize_extra_artifacts_from_local_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "rwkv"}')
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        "irwkv.model_executor.model_loader.tensorizer.hf_api",
        lambda: pytest.fail("local model should not use Hugging Face Hub"),
    )

    args = TensorizerArgs(TensorizerConfig(tensorizer_uri=str(output_dir / "model")))
    serialize_extra_artifacts(args, str(model_dir))

    assert (output_dir / "config.json").read_text() == '{"model_type": "rwkv"}'
    assert not (output_dir / "model.safetensors").exists()


def test_tensorizer_preserves_mixed_weight_dtypes(tmp_path, monkeypatch):
    source = torch.nn.Module()
    source.register_parameter(
        "base", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    )
    source.register_parameter(
        "adapter", torch.nn.Parameter(torch.tensor([1.001, 2.001]))
    )
    source.register_parameter(
        "irwkv_tensorized_marker",
        torch.nn.Parameter(torch.ones(1), requires_grad=False),
    )
    model_path = tmp_path / "model.tensors"
    with open_stream(str(model_path), mode="wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(source)
        serializer.close()

    target = torch.nn.Module()
    target.register_parameter(
        "base", torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    )
    target.register_parameter("adapter", torch.nn.Parameter(torch.zeros(2)))
    config = TensorizerConfig(tensorizer_uri=str(model_path))
    config.dtype = torch.bfloat16
    monkeypatch.setattr(
        "irwkv.model_executor.model_loader.tensorizer.current_platform.device_type",
        "cpu",
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    deserialize_tensorizer_model(target, config)

    assert target.base.dtype == torch.bfloat16
    assert target.adapter.dtype == torch.float32
    assert torch.equal(target.adapter, source.adapter)


pytest_plugins = ("pytest_asyncio",)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def patch_init_and_catch_error(self, obj, method_name, expected_error: type[Exception]):
    original = getattr(obj, method_name, None)
    if original is None:
        raise ValueError("Method '{}' not found.".format(method_name))

    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except expected_error as err:
            raise TensorizerCaughtError from err

    setattr(obj, method_name, wrapper)

    self.load_model()


def assert_specific_tensorizer_error_is_raised(
    executor,
    obj: Any,
    method_name: str,
    expected_error: type[Exception],
):
    with pytest.raises(TensorizerCaughtError):
        executor.collective_rpc(
            patch_init_and_catch_error,
            args=(
                obj,
                method_name,
                expected_error,
            ),
        )


def is_curl_installed():
    try:
        subprocess.check_call(["curl", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def write_keyfile(keyfile_path: str):
    encryption_params = EncryptionParams.random()
    pathlib.Path(keyfile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(keyfile_path, "wb") as f:
        f.write(encryption_params.key)


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_deserialized_encrypted_irwkv_model_has_same_outputs(
    model_ref, irwkv_runner, tmp_path, model_path
):
    args = EngineArgs(model=model_ref)
    with irwkv_runner(model_ref) as irwkv_model:
        key_path = tmp_path / model_ref / "model.key"
        write_keyfile(key_path)

        outputs = irwkv_model.generate(prompts, sampling_params)

    config_for_serializing = TensorizerConfig(
        tensorizer_uri=str(model_path), encryption_keyfile=str(key_path)
    )

    tensorize_irwkv_model(args, config_for_serializing)

    config_for_deserializing = TensorizerConfig(
        tensorizer_uri=str(model_path), encryption_keyfile=str(key_path)
    )

    with irwkv_runner(
        model_ref,
        load_format="tensorizer",
        model_loader_extra_config=config_for_deserializing,
    ) as loaded_irwkv_model:  # noqa: E501
        deserialized_outputs = loaded_irwkv_model.generate(prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs


def test_deserialized_hf_model_has_same_outputs(
    hf_runner, irwkv_runner, tmp_path, model_ref, model_path
):
    with hf_runner(model_ref, dtype="bfloat16") as hf_model:
        max_tokens = 5
        outputs = hf_model.generate_greedy(prompts[:1], max_tokens=max_tokens)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open_stream(model_path, "wb+") as stream:
            serializer = TensorSerializer(stream)
            serializer.write_module(hf_model.model)
            serializer.close()

    with irwkv_runner(
        model_ref,
        dtype="bfloat16",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=2048,
        max_num_seqs=4,
        load_format="tensorizer",
        model_loader_extra_config=TensorizerConfig(
            tensorizer_uri=str(model_path),
            num_readers=1,
        ),
    ) as loaded_hf_model:
        deserialized_outputs = loaded_hf_model.generate_greedy(
            prompts[:1], max_tokens=max_tokens
        )

        assert outputs == deserialized_outputs


def test_load_without_tensorizer_load_format(irwkv_runner, capfd, model_ref):
    model = None
    try:
        model = irwkv_runner(
            model_ref, model_loader_extra_config=TensorizerConfig(tensorizer_uri="test")
        )
        pytest.fail("Expected RuntimeError for extra config keys")
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            "ValueError: Unexpected extra config keys for load format auto"
        ) in combined_output
    finally:
        del model
        gc.collect()
        torch.accelerator.empty_cache()


def test_raise_value_error_on_invalid_load_format(irwkv_runner, capfd, model_ref):
    model = None
    try:
        model = irwkv_runner(
            model_ref,
            load_format="safetensors",
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"),
        )
        pytest.fail("Expected RuntimeError for extra config keys")
    except RuntimeError:
        out, err = capfd.readouterr()

        combined_output = out + err
        assert (
            "ValueError: Unexpected extra config keys for load format safetensors"
        ) in combined_output
    finally:
        del model
        gc.collect()
        torch.accelerator.empty_cache()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Requires 2 GPUs")
def test_tensorizer_with_tp_path_without_template(irwkv_runner, capfd):
    try:
        model_ref = "EleutherAI/pythia-1.4b"
        tensorized_path = f"s3://tensorized/{model_ref}/fp16/model.tensors"

        irwkv_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=TensorizerConfig(
                tensorizer_uri=tensorized_path,
                num_readers=1,
                s3_endpoint="object.ord1.coreweave.com",
            ),
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,
        )
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            "ValueError: For a sharded model, tensorizer_uri "
            "should include a string format template like '%04d' "
            "to be formatted with the rank "
            "of the shard"
        ) in combined_output


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Requires 2 GPUs")
def test_deserialized_encrypted_irwkv_model_with_tp_has_same_outputs(
    irwkv_runner, tmp_path
):
    model_ref = "EleutherAI/pythia-1.4b"
    # record outputs from un-sharded un-tensorized model
    with irwkv_runner(
        model_ref,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    ) as base_model:
        outputs = base_model.generate(prompts, sampling_params)

    # load model with two shards and serialize with encryption
    model_path = str(tmp_path / model_ref / "model-%02d.tensors")
    key_path = tmp_path / (model_ref + ".key")

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=str(key_path),
    )

    tensorize_irwkv_model(
        engine_args=EngineArgs(
            model=model_ref,
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,
            enforce_eager=True,
        ),
        tensorizer_config=tensorizer_config,
    )
    assert os.path.isfile(model_path % 0), "Serialization subprocess failed"
    assert os.path.isfile(model_path % 1), "Serialization subprocess failed"

    with irwkv_runner(
        model_ref,
        tensor_parallel_size=2,
        load_format="tensorizer",
        disable_custom_all_reduce=True,
        enforce_eager=True,
        model_loader_extra_config=tensorizer_config,
    ) as loaded_irwkv_model:
        deserialized_outputs = loaded_irwkv_model.generate(prompts, sampling_params)

    assert outputs == deserialized_outputs


@pytest.mark.flaky(reruns=3)
def test_irwkv_tensorized_model_has_same_outputs(
    model_ref, irwkv_runner, tmp_path, model_path
):
    gc.collect()
    torch.accelerator.empty_cache()
    config = TensorizerConfig(tensorizer_uri=str(model_path))
    args = EngineArgs(model=model_ref)

    with irwkv_runner(model_ref) as irwkv_model:
        outputs = irwkv_model.generate(prompts, sampling_params)

    tensorize_irwkv_model(args, config)
    assert is_irwkv_tensorized(config)

    with irwkv_runner(
        model_ref, load_format="tensorizer", model_loader_extra_config=config
    ) as loaded_irwkv_model:
        deserialized_outputs = loaded_irwkv_model.generate(prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs


def test_load_with_just_model_tensors(just_serialize_model_tensors, model_ref):
    # For backwards compatibility, ensure Tensorizer can be still be loaded
    # for inference by passing the model reference name, not a local/S3 dir,
    # and the location of the model tensors

    model_dir = just_serialize_model_tensors

    extra_config = {"tensorizer_uri": f"{model_dir}/model.tensors"}

    ## Start OpenAI API server
    args = [
        "--load-format",
        "tensorizer",
        "--model-loader-extra-config",
        json.dumps(extra_config),
    ]

    with RemoteOpenAIServer(model_ref, args):
        # This test only concerns itself with being able to load the model
        # and successfully initialize the server
        pass


def test_assert_serialization_kwargs_passed_to_tensor_serializer(
    tmp_path, irwkv_runner, model_ref
):
    serialization_params = {
        "limit_cpu_concurrency": 2,
    }
    model_path = tmp_path / "model.tensors"
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    def serialization_test(self, *args, **kwargs):
        # This is performed in the ephemeral worker process, so monkey-patching
        # will actually work, and cleanup is guaranteed so don't
        # need to reset things

        original_dict = serialization_params
        to_compare = {}

        original = tensorizer.serialization.TensorSerializer.__init__

        def tensorizer_serializer_wrapper(self, *args, **kwargs):
            nonlocal to_compare
            to_compare = kwargs.copy()
            return original(self, *args, **kwargs)

        tensorizer.serialization.TensorSerializer.__init__ = (
            tensorizer_serializer_wrapper
        )

        tensorizer_config = TensorizerConfig(**kwargs["tensorizer_config"])
        self.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )
        return to_compare | original_dict == to_compare

    kwargs = {"tensorizer_config": config.to_serializable()}

    with irwkv_runner(model_ref) as runner:
        assert assert_from_collective_rpc(runner.get_llm(), serialization_test, kwargs)


@pytest.mark.parametrize(
    (
        "serialization_error",
        "renderer_shutdown_error",
        "engine_shutdown_error",
        "expected_error",
    ),
    [
        (None, None, None, None),
        (RuntimeError("serialization failed"), None, None, "serialization failed"),
        (
            RuntimeError("serialization failed"),
            ValueError("renderer shutdown failed"),
            ValueError("engine shutdown failed"),
            "serialization failed",
        ),
        (
            None,
            ValueError("renderer shutdown failed"),
            None,
            "renderer shutdown failed",
        ),
        (None, None, ValueError("engine shutdown failed"), "engine shutdown failed"),
    ],
)
def test_tensorize_irwkv_model_shuts_down_engine(
    monkeypatch,
    serialization_error,
    renderer_shutdown_error,
    engine_shutdown_error,
    expected_error,
):
    from irwkv.v1.engine.llm_engine import LLMEngine

    engine_args = Mock()
    config = Mock(encryption_keyfile=None)
    engine = Mock()
    engine.collective_rpc.side_effect = serialization_error
    engine.renderer.shutdown.side_effect = renderer_shutdown_error
    engine.engine_core.shutdown.side_effect = engine_shutdown_error
    monkeypatch.setattr(LLMEngine, "from_irwkv_config", Mock(return_value=engine))
    monkeypatch.setenv("IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "7")

    if expected_error is None:
        tensorize_irwkv_model(engine_args, config)
    else:
        with pytest.raises(Exception, match=expected_error) as exc_info:
            tensorize_irwkv_model(engine_args, config)
        notes = getattr(exc_info.value, "__notes__", ())
        if serialization_error is not None and renderer_shutdown_error is not None:
            assert any("renderer shutdown failed" in note for note in notes)
        if serialization_error is not None and engine_shutdown_error is not None:
            assert any("engine shutdown failed" in note for note in notes)

    engine.engine_core.shutdown.assert_called_once_with(timeout=17.0)
    engine.renderer.shutdown.assert_called_once_with()


def test_assert_deserialization_kwargs_passed_to_tensor_deserializer(
    tmp_path, capfd, model_ref
):
    deserialization_kwargs = {
        "num_readers": "bar",  # illegal value
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_path = tmp_path / "model.tensors"
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_irwkv_model(args, config)

    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
    )

    engine_args = EngineArgs(
        model=model_ref,
        load_format="tensorizer",
        model_loader_extra_config=loader_tc.to_serializable(),
    )

    irwkv_config = engine_args.create_engine_config()
    executor = DummyExecutor(irwkv_config)

    assert_specific_tensorizer_error_is_raised(
        executor,
        tensorizer.serialization.TensorDeserializer,
        "__init__",
        TypeError,
    )


def test_assert_stream_kwargs_passed_to_tensor_deserializer(tmp_path, capfd, model_ref):
    deserialization_kwargs = {
        "num_readers": 1,
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_path = tmp_path / "model.tensors"
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_irwkv_model(args, config)

    stream_kwargs = {"mode": "foo"}

    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
        stream_kwargs=stream_kwargs,
    )

    engine_args = EngineArgs(
        model=model_ref,
        load_format="tensorizer",
        model_loader_extra_config=loader_tc.to_serializable(),
    )

    irwkv_config = engine_args.create_engine_config()
    executor = DummyExecutor(irwkv_config)

    assert_specific_tensorizer_error_is_raised(
        executor,
        irwkv.model_executor.model_loader.tensorizer,
        "open_stream",
        ValueError,
    )


@pytest.mark.parametrize("illegal_value", BLACKLISTED_TENSORIZER_ARGS)
def test_blacklisted_parameter_for_loading(
    tmp_path, irwkv_runner, capfd, illegal_value, model_ref
):
    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_path = tmp_path / "model.tensors"
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_irwkv_model(args, config)

    loader_tc = {"tensorizer_uri": str(model_path), illegal_value: "foo"}

    try:
        irwkv_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=loader_tc,
        )
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            f"ValueError: {illegal_value} is not an allowed Tensorizer argument."
        ) in combined_output
