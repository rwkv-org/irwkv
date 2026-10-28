# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import contextlib
import contextvars
import dataclasses
import fnmatch
import json
import os
import tempfile
import threading
import time
from collections.abc import Generator, MutableMapping
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import regex as re
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import PreTrainedConfig

import irwkv.envs as envs
from irwkv.config import (
    IrwkvConfig,
    ModelConfig,
    ParallelConfig,
    set_current_irwkv_config,
)
from irwkv.logger import init_logger
from irwkv.platforms import current_platform
from irwkv.transformers_utils.repo_utils import hf_api
from irwkv.utils.argparse_utils import FlexibleArgumentParser
from irwkv.utils.import_utils import PlaceholderModule

if TYPE_CHECKING:
    from irwkv.engine.arg_utils import EngineArgs

try:
    from tensorizer import (
        DecryptionParams,
        EncryptionParams,
        TensorDeserializer,
        TensorSerializer,
    )
    from tensorizer.stream_io import open_stream
    from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor

except ImportError:
    tensorizer = PlaceholderModule("tensorizer")
    DecryptionParams = tensorizer.placeholder_attr("DecryptionParams")
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")
    TensorDeserializer = tensorizer.placeholder_attr("TensorDeserializer")
    TensorSerializer = tensorizer.placeholder_attr("TensorSerializer")
    open_stream = tensorizer.placeholder_attr("stream_io.open_stream")
    convert_bytes = tensorizer.placeholder_attr("utils.convert_bytes")
    get_mem_usage = tensorizer.placeholder_attr("utils.get_mem_usage")
    no_init_or_tensor = tensorizer.placeholder_attr("utils.no_init_or_tensor")

__all__ = [
    "EncryptionParams",
    "DecryptionParams",
    "TensorDeserializer",
    "TensorSerializer",
    "open_stream",
    "convert_bytes",
    "get_mem_usage",
    "no_init_or_tensor",
    "TensorizerConfig",
]

logger = init_logger(__name__)
_TENSORIZER_ENGINE_CLEANUP_GRACE_S = 10.0


def is_valid_deserialization_uri(uri: str | None) -> bool:
    if uri:
        scheme = uri.lower().split("://")[0]
        return scheme in {"s3", "http", "https"} or os.path.exists(uri)
    return False


def tensorizer_kwargs_arg(value):
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError(
            f"Not deserializable to dict: {value}. serialization_kwargs and "
            f"deserialization_kwargs must be "
            f"deserializable from a JSON string to a dictionary. "
        )
    return loaded


class MetaTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func._schema.name == "aten::empty" and "device" not in kwargs:
            kwargs["device"] = "meta"

        return func(*args, **kwargs)


def meta_tensor_mode(
    loading_code=None,
):
    if loading_code is None:
        return _NoInitOrTensorImpl.context_manager()
    elif callable(loading_code):
        with _NoInitOrTensorImpl.context_manager():
            return loading_code()
    else:
        raise TypeError(
            "expected a callable to evaluate,"
            " or None if being used as a context manager;"
            f' got an object of type "{type(loading_code).__name__}" instead.'
        )


class _NoInitOrTensorImpl:
    _MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)
    _MODULE_ORIGINALS = tuple((m, m.reset_parameters) for m in _MODULES)

    is_active = contextvars.ContextVar("_NoInitOrTensorImpl.is_active", default=False)
    _count_active: int = 0
    _count_active_lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def context_manager(cls):
        if cls.is_active.get():
            yield
            return

        with cls._count_active_lock:
            cls._count_active += 1
            if cls._count_active == 1:
                for mod in cls._MODULES:
                    mod.reset_parameters = cls._disable(mod.reset_parameters)

        reset_token = cls.is_active.set(True)

        try:
            with MetaTensorMode():
                yield
        finally:
            cls.is_active.reset(reset_token)
            with cls._count_active_lock:
                cls._count_active -= 1
                if cls._count_active == 0:
                    for mod, original in cls._MODULE_ORIGINALS:
                        mod.reset_parameters = original

    @staticmethod
    def _disable(func):
        def wrapper(*args, **kwargs):
            if not _NoInitOrTensorImpl.is_active.get():
                return func(*args, **kwargs)

        return wrapper


@dataclass
class TensorizerConfig(MutableMapping):
    tensorizer_uri: str | None = None
    tensorizer_dir: str | None = None
    irwkv_tensorized: bool | None = None
    verify_hash: bool | None = None
    num_readers: int | None = None
    encryption_keyfile: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint: str | None = None
    stream_kwargs: dict[str, Any] | None = None
    serialization_kwargs: dict[str, Any] | None = None
    deserialization_kwargs: dict[str, Any] | None = None
    _extra_serialization_attrs: dict[str, Any] | None = field(init=False, default=None)
    model_class: type[torch.nn.Module] | None = field(init=False, default=None)
    hf_config: PreTrainedConfig | None = field(init=False, default=None)
    dtype: str | torch.dtype | None = field(init=False, default=None)
    _is_sharded: bool = field(init=False, default=False)
    _fields: ClassVar[tuple[str, ...]]
    _keys: ClassVar[frozenset[str]]
    """Configuration class for Tensorizer settings.

    These settings configure the behavior of model serialization and
    deserialization using Tensorizer.

    Attributes:
        tensorizer_uri: Path to serialized model tensors. Can be a local file
            path or a S3 URI. Unless a `tensorizer_dir` is passed to this
            object's initializer, this is a required argument.
        tensorizer_dir: Path to a directory containing serialized model tensors,
            and all other potential model artifacts to load the model, such as
            configs and tokenizer files. Can be passed instead of
            `tensorizer_uri` where the `model.tensors` file will be assumed
            to be in this directory.
        irwkv_tensorized: If True, indicates that the serialized model is a
            irwkv model. This is used to determine the behavior of the
            TensorDeserializer when loading tensors from a serialized model.
            It is far faster to deserialize a irwkv model as it utilizes
            tensorizer's optimized GPU loading. Note that this is now
            deprecated, as serialized irwkv models are now automatically
            inferred as irwkv models.
        verify_hash: If True, the hashes of each tensor will be verified
            against the hashes stored in the metadata. A `HashMismatchError`
            will be raised if any of the hashes do not match.
        num_readers: Controls how many threads are allowed to read concurrently
            from the source file. Default is `None`, which will dynamically set
            the number of readers based on the number of available
            resources and model size. This greatly increases performance.
        encryption_keyfile: File path to a binary file containing a
            binary key to use for decryption. `None` (the default) means
            no decryption. See the example script in
            examples/features/tensorize_irwkv_model.py.
        s3_access_key_id: The access key for the S3 bucket. Can also be set via
            the S3_ACCESS_KEY_ID environment variable.
        s3_secret_access_key: The secret access key for the S3 bucket. Can also
            be set via the S3_SECRET_ACCESS_KEY environment variable.
        s3_endpoint: The endpoint for the S3 bucket. Can also be set via the
            S3_ENDPOINT_URL environment variable.
    """

    def __post_init__(self):
        # check if the configuration is for a sharded irwkv model
        self._is_sharded = (
            isinstance(self.tensorizer_uri, str)
            and re.search(r"%0\dd", self.tensorizer_uri) is not None
        )

        if self.tensorizer_dir and self.tensorizer_uri:
            logger.warning_once(
                "Provided both tensorizer_dir and tensorizer_uri. "
                "Inferring tensorizer_dir from tensorizer_uri as the "
                "latter takes precedence."
            )
            self.tensorizer_dir = os.path.dirname(self.tensorizer_uri)
        if not self.tensorizer_uri:
            if self.tensorizer_dir:
                self.tensorizer_uri = f"{self.tensorizer_dir}/model.tensors"
            else:
                raise ValueError(
                    "Unable to resolve tensorizer_uri. "
                    "A valid tensorizer_uri or tensorizer_dir must be provided."
                )
        else:
            self.tensorizer_dir = os.path.dirname(self.tensorizer_uri)

        if not self.serialization_kwargs:
            self.serialization_kwargs = {}
        if not self.deserialization_kwargs:
            self.deserialization_kwargs = {}

    def to_serializable(self) -> dict[str, Any]:
        # Due to TensorizerConfig needing to be msgpack-serializable, it needs
        # support for morphing back and forth between itself and its dict
        # representation

        # TensorizerConfig's representation as a dictionary is meant to be
        # linked to TensorizerConfig in such a way that the following is
        # technically initializable:
        # TensorizerConfig(**my_tensorizer_cfg.to_serializable())

        # This means the dict must not retain non-initializable parameters
        # and post-init attribute states

        # Also don't want to retain private and unset parameters, so only retain
        # not None values and public attributes

        raw_tc_dict = asdict(self)
        blacklisted = []

        if "tensorizer_uri" in raw_tc_dict and "tensorizer_dir" in raw_tc_dict:
            blacklisted.append("tensorizer_dir")

        tc_dict = {}
        for k, v in raw_tc_dict.items():
            if (
                k not in blacklisted
                and k not in tc_dict
                and not k.startswith("_")
                and v is not None
            ):
                tc_dict[k] = v

        return tc_dict

    def _construct_tensorizer_args(self) -> "TensorizerArgs":
        return TensorizerArgs(self)  # type: ignore

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1 and not self._is_sharded:
            raise ValueError(
                "For a sharded model, tensorizer_uri should include a"
                " string format template like '%04d' to be formatted"
                " with the rank of the shard"
            )

    def open_stream(self, tensorizer_args: "TensorizerArgs | None" = None):
        if tensorizer_args is None:
            tensorizer_args = self._construct_tensorizer_args()

        return open_stream(self.tensorizer_uri, **tensorizer_args.stream_kwargs)

    def keys(self):
        return self._keys

    def __len__(self):
        return len(fields(self))

    def __iter__(self):
        return iter(self._fields)

    def __getitem__(self, item: str) -> Any:
        if item not in self.keys():
            raise KeyError(item)
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.keys():
            # Disallow modifying invalid keys
            raise KeyError(key)
        setattr(self, key, value)

    def __delitem__(self, key, /):
        if key not in self.keys():
            raise KeyError(key)
        delattr(self, key)


TensorizerConfig._fields = tuple(f.name for f in fields(TensorizerConfig))
TensorizerConfig._keys = frozenset(TensorizerConfig._fields)


@dataclass
class TensorizerArgs:
    tensorizer_uri: str | None = None
    tensorizer_dir: str | None = None
    encryption_keyfile: str | None = None

    def __init__(self, tensorizer_config: TensorizerConfig):
        for k, v in tensorizer_config.items():
            setattr(self, k, v)
        self.file_obj = tensorizer_config.tensorizer_uri
        self.s3_access_key_id = (
            tensorizer_config.s3_access_key_id or envs.S3_ACCESS_KEY_ID
        )
        self.s3_secret_access_key = (
            tensorizer_config.s3_secret_access_key or envs.S3_SECRET_ACCESS_KEY
        )
        self.s3_endpoint = tensorizer_config.s3_endpoint or envs.S3_ENDPOINT_URL

        self.stream_kwargs = {
            "s3_access_key_id": tensorizer_config.s3_access_key_id,
            "s3_secret_access_key": tensorizer_config.s3_secret_access_key,
            "s3_endpoint": tensorizer_config.s3_endpoint,
            **(tensorizer_config.stream_kwargs or {}),
        }

        self.deserialization_kwargs = {
            "verify_hash": tensorizer_config.verify_hash,
            "encryption": tensorizer_config.encryption_keyfile,
            "num_readers": tensorizer_config.num_readers,
            **(tensorizer_config.deserialization_kwargs or {}),
        }

        if self.encryption_keyfile:
            with open_stream(
                tensorizer_config.encryption_keyfile,
                **self.stream_kwargs,
            ) as stream:
                key = stream.read()
                decryption_params = DecryptionParams.from_key(key)
                self.deserialization_kwargs["encryption"] = decryption_params

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Tensorizer CLI arguments."""
        # Tensorizer options arg group
        group = parser.add_argument_group(
            "tensorizer options",
            description=(
                "Options for configuring the behavior of the"
                " tensorizer deserializer when "
                "load_format=tensorizer is specified when "
                "initializing an LLMEngine, either via the CLI "
                "when running the irwkv OpenAI inference server "
                "with a JSON string passed to "
                "--model-loader-extra-config or as arguments given "
                "to TensorizerConfig when passed to "
                "model_loader_extra_config in the constructor "
                "for LLMEngine."
            ),
        )

        group.add_argument(
            "--tensorizer-uri",
            type=str,
            help="Path to serialized model tensors. Can be a local file path,"
            " or an HTTP(S) or S3 URI.",
        )
        group.add_argument(
            "--verify-hash",
            action="store_true",
            help="If enabled, the hashes of each tensor will be verified"
            " against the hashes stored in the file metadata. An exception"
            " will be raised if any of the hashes do not match.",
        )
        group.add_argument(
            "--encryption-keyfile",
            type=str,
            default=None,
            help="The file path to a binary file containing a binary key to "
            "use for decryption. Can be a file path or S3 network URI.",
        )
        group.add_argument(
            "--num-readers",
            default=None,
            type=int,
            help="Controls how many threads are allowed to read concurrently "
            "from the source file. Default is `None`, which will dynamically "
            "set the number of readers based on the available resources "
            "and model size. This greatly increases performance.",
        )
        group.add_argument(
            "--s3-access-key-id",
            type=str,
            default=None,
            help="The access key for the S3 bucket. Can also be set via the "
            "S3_ACCESS_KEY_ID environment variable.",
        )
        group.add_argument(
            "--s3-secret-access-key",
            type=str,
            default=None,
            help="The secret access key for the S3 bucket. Can also be set via "
            "the S3_SECRET_ACCESS_KEY environment variable.",
        )
        group.add_argument(
            "--s3-endpoint",
            type=str,
            default=None,
            help="The endpoint for the S3 bucket. Can also be set via the "
            "S3_ENDPOINT_URL environment variable.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TensorizerArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        tensorizer_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return tensorizer_args


def _check_tensors_on_meta_device(model: nn.Module) -> None:
    for tensor in model.state_dict().values():
        if tensor.device.type == "meta":
            raise ValueError(
                "The serialized model contains tensors on the meta device,"
                " indicating that some tensors were not loaded properly."
                " Please check that the parameters of the model being"
                " specified match that of the serialized model, such as"
                " its quantization."
            )


def init_tensorizer_model(
    tensorizer_config: TensorizerConfig, irwkv_config: IrwkvConfig
) -> nn.Module:
    assert tensorizer_config.hf_config is not None
    model_args = tensorizer_config.hf_config
    model_args.dtype = tensorizer_config.dtype
    assert tensorizer_config.model_class is not None
    # TODO: Do we need to consider old-style model class?
    with meta_tensor_mode(), set_current_irwkv_config(irwkv_config, check_compile=True):
        return tensorizer_config.model_class(irwkv_config=irwkv_config)


def deserialize_tensorizer_model(
    model: nn.Module, tensorizer_config: TensorizerConfig
) -> None:
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    if not is_valid_deserialization_uri(tensorizer_config.tensorizer_uri):
        raise ValueError(
            f"{tensorizer_config.tensorizer_uri} is not a valid "
            f"tensorizer URI. Please check that the URI is correct. "
            f"It must either point to a local existing file, or have a "
            f"S3, HTTP or HTTPS scheme."
        )
    before_mem = get_mem_usage()
    start = time.perf_counter()
    device_index = torch.accelerator.current_device_index()
    device_type = current_platform.device_type
    with (
        open_stream(
            tensorizer_config.tensorizer_uri, mode="rb", **tensorizer_args.stream_kwargs
        ) as stream,
        TensorDeserializer(
            stream,
            device=f"{device_type}:{device_index}",
            **tensorizer_args.deserialization_kwargs,
        ) as deserializer,
    ):
        deserializer.load_into_module(model)
        end = time.perf_counter()

    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    deserializer.close()
    logger.info(
        "Deserialized %s in %0.2fs, %s/s", total_bytes_str, end - start, per_second
    )
    logger.info("Memory usage before: %s", before_mem)
    logger.info("Memory usage after: %s", after_mem)

    _check_tensors_on_meta_device(model)
    del model.irwkv_tensorized_marker


def tensorizer_weights_iterator(
    tensorizer_args: "TensorizerArgs",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    logger.warning(
        "Deserializing HuggingFace models is not optimized for "
        "loading on irwkv, as tensorizer is forced to load to CPU. "
        "Consider deserializing a irwkv model instead for faster "
        "load times. See the "
        "examples/features/tensorize_irwkv_model.py example script "
        "for serializing irwkv models."
    )

    deserializer_args = tensorizer_args.deserialization_kwargs
    stream_kwargs = tensorizer_args.stream_kwargs
    stream = open_stream(tensorizer_args.tensorizer_uri, **stream_kwargs)
    with TensorDeserializer(stream, **deserializer_args, device="cpu") as state:
        yield from state.items()
    del state


def is_irwkv_tensorized(tensorizer_config: "TensorizerConfig") -> bool:
    """Infer if the model is a irwkv model by checking the weights for
    a irwkv tensorized marker.

    Args:
        tensorizer_config: The TensorizerConfig object containing the
            tensorizer_uri to the serialized model.

    Returns:
        bool: True if the model is a irwkv model, False otherwise.

    """
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    deserializer = TensorDeserializer(
        open_stream(tensorizer_args.tensorizer_uri, **tensorizer_args.stream_kwargs),
        **tensorizer_args.deserialization_kwargs,
        lazy_load=True,
    )
    if tensorizer_config.irwkv_tensorized:
        logger.warning(
            "Please note that newly serialized irwkv models are automatically "
            "inferred as irwkv models, so setting irwkv_tensorized=True is "
            "only necessary for models serialized prior to this change."
        )
        return True
    return ".irwkv_tensorized_marker" in deserializer


def serialize_extra_artifacts(
    tensorizer_args: TensorizerArgs,
    model: str,
    revision: str | None = None,
) -> None:
    ignore_patterns = [
        "*.pt",
        "*.safetensors",
        "*.bin",
        "*.cache",
        "*.gitattributes",
        "*.md",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        if os.path.isdir(model):
            artifact_dir = model
        else:
            hf_api().snapshot_download(
                model,
                revision=revision,
                local_dir=tmpdir,
                ignore_patterns=ignore_patterns,
            )
            artifact_dir = tmpdir
        for artifact in os.scandir(artifact_dir):
            if not artifact.is_file() or any(
                fnmatch.fnmatch(artifact.name, pattern) for pattern in ignore_patterns
            ):
                continue
            with (
                open(artifact.path, "rb") as f,
                open_stream(
                    f"{tensorizer_args.tensorizer_dir}/{artifact.name}",
                    mode="wb+",
                    **tensorizer_args.stream_kwargs,
                ) as stream,
            ):
                logger.info("Writing artifact %s", artifact.name)
                stream.write(f.read())


def serialize_irwkv_model(
    model: nn.Module,
    tensorizer_config: TensorizerConfig,
    model_config: "ModelConfig",
) -> nn.Module:
    model.register_parameter(
        "irwkv_tensorized_marker",
        nn.Parameter(torch.tensor((1,), device="meta"), requires_grad=False),
    )

    tensorizer_args = tensorizer_config._construct_tensorizer_args()

    encryption_params = None
    if (keyfile := tensorizer_config.encryption_keyfile) is not None:
        with open(keyfile, "rb") as f:
            key = f.read()
        encryption_params = EncryptionParams(key=key)

    if (output_file := tensorizer_args.tensorizer_uri) is None:
        raise ValueError("tensorizer_uri must be specified for serialization.")
    if tensorizer_config._is_sharded:
        from irwkv.distributed import get_tensor_model_parallel_rank

        output_file = output_file % get_tensor_model_parallel_rank()

    with open_stream(
        output_file, mode="wb+", **tensorizer_args.stream_kwargs
    ) as stream:
        serializer = TensorSerializer(
            stream,
            encryption=encryption_params,
            **(tensorizer_config.serialization_kwargs or {}),
        )
        serializer.write_module(model)
        serializer.close()

    serialize_extra_artifacts(
        tensorizer_args,
        model_config.model,
        revision=model_config.revision,
    )

    logger.info("Successfully serialized model to %s", str(output_file))
    return model


def tensorize_irwkv_model(
    engine_args: "EngineArgs",
    tensorizer_config: TensorizerConfig,
    generate_keyfile: bool = True,
):
    """Utility to load a model and then serialize it with Tensorizer.

    Intended to be used separately from running a irwkv server since it
    creates its own Engine instance.
    """
    engine_config = engine_args.create_engine_config()
    tensorizer_config.verify_with_parallel_config(engine_config.parallel_config)

    # generate the encryption key before creating the engine to support sharding
    if (
        generate_keyfile
        and (keyfile := tensorizer_config.encryption_keyfile) is not None
    ):
        encryption_params = EncryptionParams.random()
        with open_stream(
            keyfile,
            mode="wb+",
            s3_access_key_id=tensorizer_config.s3_access_key_id,
            s3_secret_access_key=tensorizer_config.s3_secret_access_key,
            s3_endpoint=tensorizer_config.s3_endpoint,
        ) as stream:
            stream.write(encryption_params.key)

    from irwkv.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_irwkv_config(engine_config)
    error: BaseException | None = None
    try:
        engine.collective_rpc(
            "save_tensorized_model",
            kwargs={"tensorizer_config": tensorizer_config.to_serializable()},
        )
    except BaseException as operation_error:
        error = operation_error

    def shutdown_engine_core() -> None:
        engine.engine_core.shutdown(
            timeout=(
                envs.IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                + _TENSORIZER_ENGINE_CLEANUP_GRACE_S
            )
        )

    for name, callback in (
        ("renderer", engine.renderer.shutdown),
        ("engine core", shutdown_engine_core),
    ):
        try:
            callback()
        except BaseException as shutdown_error:
            logger.exception("Failed to shut down tensorization %s", name)
            if error is None:
                error = shutdown_error
            elif hasattr(error, "add_note"):
                error.add_note(f"{name} shutdown also failed: {shutdown_error!r}")

    if error is not None:
        raise error
