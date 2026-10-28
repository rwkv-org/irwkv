# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import huggingface_hub
from transformers import GenerationConfig, PreTrainedConfig
from transformers.configuration_utils import ALLOWED_LAYER_TYPES
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from irwkv import envs
from irwkv.logger import init_logger
from irwkv.transformers_utils.utils import (
    parse_safetensors_file_metadata,
    without_trust_remote_code,
)

from .config_parser_base import ConfigParserBase
from .repo_utils import (
    file_or_path_exists,
    get_hf_file_to_dict,
    hf_api,
    list_repo_files,
    try_get_local_file,
    with_retry,
)

if envs.IRWKV_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

logger = init_logger(__name__)

_ST_POOLING_MODULE_TYPES = {
    "sentence_transformers.models.Pooling",
    "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
}
_ST_NORMALIZE_MODULE_TYPES = {
    "sentence_transformers.models.Normalize",
    "sentence_transformers.base.modules.normalize.Normalize",
    "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
}
_DENSE_MODULE_TYPES = {
    "sentence_transformers.models.Dense",
    "sentence_transformers.base.modules.dense.Dense",
    "pylate.models.Dense.Dense",
}


_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}


def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    return set(rope_parameters.keys()).issubset(ALLOWED_LAYER_TYPES)


class HFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PreTrainedConfig]:
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        trust_remote_code |= kwargs.get("trust_remote_code", False)
        kwargs = without_trust_remote_code(kwargs)
        config_dict, _ = PreTrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )
        kwargs.pop("hf_overrides", None)
        try:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs,
            )
        except ValueError as e:
            if (
                not trust_remote_code
                and "requires you to execute the configuration file" in str(e)
            ):
                err_msg = (
                    "Failed to load the model config. If the model "
                    "is a custom model not yet available in the "
                    "HuggingFace transformers library, consider setting "
                    "`trust_remote_code=True` in LLM or using the "
                    "`--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            raise
        config = _maybe_remap_hf_config_attrs(config)
        return config_dict, config


_CONFIG_FORMAT_TO_CONFIG_PARSER: dict[str, type[ConfigParserBase]] = {
    "hf": HFConfigParser,
}

ConfigFormat = Literal[
    "auto",
    "hf",
]


def get_config_parser(config_format: str) -> ConfigParserBase:
    """Get the config parser for a given config format."""
    if config_format not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format]()


def register_config_parser(config_format: str):
    """Register a customized irwkv config parser.
     When a config format is not supported by irwkv, you can register a customized
    config parser to support it.

    Args:
         config_format (str): The config parser format name.

    Examples:
         >>> from irwkv.transformers_utils.config import (get_config_parser,
                                                         register_config_parser)
         >>> from irwkv.transformers_utils.config_parser_base import ConfigParserBase
         >>>
         >>> @register_config_parser("custom_config_parser")
         ... class CustomConfigParser(ConfigParserBase):
         ...     def parse(
         ...         self,
         ...         model: Union[str, Path],
         ...         trust_remote_code: bool,
         ...         revision: str | None = None,
         ...         code_revision: str | None = None,
         ...         **kwargs,
         ...     ) -> tuple[dict, PreTrainedConfig]:
         ...         raise NotImplementedError
         >>>
         >>> type(get_config_parser("custom_config_parser"))
         <class 'CustomConfigParser'>

    """  # noqa: E501

    def _wrapper(config_parser_cls):
        if config_format in _CONFIG_FORMAT_TO_CONFIG_PARSER:
            logger.warning(
                "Config format `%s` is already registered, and will be "
                "overwritten by the new parser class `%s`.",
                config_format,
                config_parser_cls,
            )
        if not issubclass(config_parser_cls, ConfigParserBase):
            raise ValueError(
                "The config parser must be a subclass of `ConfigParserBase`."
            )
        _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format] = config_parser_cls
        logger.info(
            "Registered config parser `%s` with config format `%s`",
            config_parser_cls,
            config_format,
        )
        return config_parser_cls

    return _wrapper


def set_default_rope_theta(config: PreTrainedConfig, default_theta: float) -> None:
    """Some models may have no rope_theta in their config but still use RoPE.
    This function sets a default rope_theta if it's missing."""
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    if "rope_theta" not in config.rope_parameters:
        config.rope_parameters["rope_theta"] = default_theta


def patch_legacy_rope_type(rope_parameters: dict[str, Any] | None) -> None:
    """Patch legacy RoPE type fields for backwards compatibility with
    older custom models which would otherwise fail to load."""
    # No RoPE parameters to patch
    if rope_parameters is None:
        return

    def _patch_legacy_rope_type(rope_parameters: dict[str, Any]) -> None:
        # Case 1: Both legacy and modern fields present - check for conflicts
        if "rope_type" in rope_parameters and "type" in rope_parameters:
            rope_type = rope_parameters["rope_type"]
            rope_type_legacy = rope_parameters["type"]
            if (rope_type_legacy == "su" and rope_type == "longrope") or (
                rope_type_legacy == "mrope" and rope_type == "default"
            ):
                pass  # No action needed
            elif rope_type != rope_type_legacy:
                raise ValueError(
                    f"Found conflicts between 'rope_type={rope_type}' (modern "
                    f"field) and 'type={rope_type_legacy}' (legacy field). "
                    "You should only specify one of them."
                )
        # Case 2: Only legacy field present - patch to modern format with warning
        if "rope_type" not in rope_parameters and "type" in rope_parameters:
            rope_parameters["rope_type"] = rope_parameters["type"]
            logger.info("Replacing legacy 'type' key with 'rope_type'")
        # Case 3: No rope_type field present - nothing to patch
        if "rope_type" not in rope_parameters:
            return
        # Patch legacy rope_type values with warning
        if rope_parameters["rope_type"] == "su":
            rope_parameters["rope_type"] = "longrope"
            logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
        elif rope_parameters["rope_type"] == "mrope":
            if "mrope_section" not in rope_parameters:
                raise ValueError(
                    "Legacy rope_type 'mrope' requires "
                    "'mrope_section' in rope_parameters"
                )
            rope_parameters["rope_type"] = "default"
            logger.warning("Replacing legacy rope_type 'mrope' with 'default'")

    # Handle nested rope_parameters in interleaved sliding attention models
    if is_rope_parameters_nested(rope_parameters):
        for rope_parameters_layer_type in rope_parameters.values():
            if rope_parameters_layer_type is not None:
                _patch_legacy_rope_type(rope_parameters_layer_type)
    else:
        _patch_legacy_rope_type(rope_parameters)


def patch_rope_parameters(config: PreTrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    from irwkv.config.utils import getattr_iter

    # Older custom models may use non-standard field names which need patching.
    names = ["rope_theta", "rotary_emb_base"]
    rope_theta = getattr_iter(config, names, None, warn=True)
    names = ["partial_rotary_factor", "rotary_pct", "rotary_emb_fraction"]
    partial_rotary_factor = getattr_iter(config, names, None, warn=True)

    if rope_theta is not None or getattr(config, "rope_parameters", None):
        # Patch these fields in case they used non-standard names
        if rope_theta is not None:
            config.rope_theta = rope_theta
        if partial_rotary_factor is not None:
            config.partial_rotary_factor = partial_rotary_factor
        # Standardize and validate RoPE parameters
        patch_legacy_rope_type(getattr(config, "rope_parameters", None))
        config.standardize_rope_params()
        config.validate_rope()


def iter_rope_parameters(config: PreTrainedConfig) -> Iterator[dict[str, Any]]:
    """Yield a config's rope parameters, one dict per layer type if nested."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        return

    if is_rope_parameters_nested(rope_parameters):
        yield from (p for p in rope_parameters.values() if isinstance(p, dict) and p)
    elif rope_parameters:
        yield rope_parameters


def _mrope_section(config: PreTrainedConfig) -> Sequence[int] | None:
    """Return the M-RoPE section this config declares, if any.

    `xdrope_section` is the legacy name HunYuan-VL checkpoints use for the same
    field; upstream Transformers normalises it to `mrope_section`.
    """
    from irwkv.config.utils import getattr_iter

    names = ("mrope_section", "xdrope_section")

    for params in iter_rope_parameters(config):
        for i, name in enumerate(names):
            section = params.get(name)
            if isinstance(section, (list, tuple)):
                if i > 0:
                    logger.warning_once(
                        "rope_parameters contains a deprecated key '%s'. "
                        "Please use the preferred key '%s' instead.",
                        name,
                        names[0],
                    )
                return section

    section = getattr_iter(config, names, None, warn=True)
    return section if isinstance(section, (list, tuple)) else None


def _uses_mrope(config: PreTrainedConfig) -> bool:
    return _mrope_section(config) is not None


def uses_mrope(config: PreTrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    return (
        _uses_mrope(config)
        or _uses_mrope(config.get_text_config())
        or thinker_uses_mrope(config)
    )


def thinker_uses_mrope(config: PreTrainedConfig) -> bool:
    """Detect if the model contains a thinker config and it uses M-ROPE."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        return False

    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is None:
        return False

    return uses_mrope(thinker_text_config)


def mrope_num_dims(config: PreTrainedConfig) -> int:
    """Number of M-RoPE position channels the model consumes.

    Each section entry sizes one channel, so the section length is the channel
    count. Interleaved M-RoPE also accepts a 2 section variant whose positions
    are still 3D, so never return fewer than 3.
    """
    if not uses_mrope(config):
        return 0

    for candidate in (config.get_text_config(), config):
        mrope_section = _mrope_section(candidate)
        if mrope_section is not None:
            return max(len(mrope_section), 3)

    # Custom configs may declare M-RoPE without exposing the sections.
    return 3


def is_encoder_decoder(config: PreTrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""

    def _is_encoder_decoder(config: PreTrainedConfig) -> bool:
        return getattr(config, "is_encoder_decoder", False)

    return _is_encoder_decoder(config) or _is_encoder_decoder(config.get_text_config())


def _maybe_remap_hf_config_attrs(config: PreTrainedConfig) -> PreTrainedConfig:
    """Remap config attributes to match the expected names."""
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr, new_attr)
    return config


def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    hf_overrides_kw: dict[str, Any] | None = None,
    hf_overrides_fn: Callable[[PreTrainedConfig], PreTrainedConfig] | None = None,
    **kwargs,
) -> PreTrainedConfig:
    if config_format == "auto":
        try:
            if file_or_path_exists(model, HF_CONFIG_NAME, revision=revision):
                config_format = "hf"
            else:
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "With config_format 'auto', ensure your model has "
                    "config.json (HF format). "
                    "Otherwise please specify your_custom_config_format "
                    "in engine args for customized config parser."
                )

        except Exception as e:
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
            ).format(model=model)

            raise ValueError(error_message) from e

    config_parser = get_config_parser(config_format)
    # Retry to tolerate a concurrent HF cache refresh briefly hiding config.json.
    config_dict, config = with_retry(
        lambda: config_parser.parse(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            hf_overrides=hf_overrides_kw or hf_overrides_fn,
            **kwargs,
        ),
        f"Error parsing config for {model}",
    )

    # Architecture mapping for models without explicit architectures field
    if not config.architectures:
        if config.model_type not in MODEL_MAPPING_NAMES:
            logger.warning(
                "Model config does not have a top-level 'architectures' field: "
                "expecting `hf_overrides={'architectures': ['...']}` to be passed "
                "in engine args."
            )
        else:
            model_type = MODEL_MAPPING_NAMES[config.model_type]
            config.update({"architectures": [model_type]})

    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    # Exhaustively patch RoPE parameters everywhere they might be
    patch_rope_parameters(config)
    patch_rope_parameters(config.get_text_config())
    SubConfigs: TypeAlias = dict[str, PreTrainedConfig]
    sub_configs: SubConfigs | None = getattr(config, "sub_configs", None)
    if sub_configs:
        for sub_config in sub_configs:
            patch_rope_parameters(getattr(config, sub_config))

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


@cache
def get_sentence_transformer_tokenizer_config(
    model: str | Path, revision: str | None = "main"
) -> dict[str, Any] | None:
    """Returns the tokenization configuration dictionary for a
    given Sentence Transformer BERT model.

    Args:
        model (str|Path): The name of the Sentence Transformer BERT model.
        revision (str, optional): The revision of the model to use. Defaults to 'main'.

    Returns:
    - dict: A dictionary containing the configuration parameters
    for the Sentence Transformer BERT model.

    """
    sentence_transformer_config_files = [
        "sentence_bert_config.json",
        "sentence_roberta_config.json",
        "sentence_distilbert_config.json",
        "sentence_camembert_config.json",
        "sentence_albert_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
    ]
    encoder_dict = None

    for config_file in sentence_transformer_config_files:
        if isinstance(
            try_get_local_file(model=model, file_name=config_file, revision=revision),
            Path,
        ):
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    if not encoder_dict and not Path(model).is_absolute():
        try:
            # If model is on HuggingfaceHub, get the repo files
            repo_files = list_repo_files(model, revision=revision)
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                encoder_dict = get_hf_file_to_dict(config_name, model, revision)
                if encoder_dict:
                    break

    if not encoder_dict:
        return None

    logger.info("Found sentence-transformers tokenize configuration.")

    if "max_seq_length" in encoder_dict:
        return encoder_dict
    return None


def maybe_register_config_serialize_by_value() -> None:
    """Try to register HF model configuration class to serialize by value

    If trust_remote_code is set, and the model's config file specifies an
    `AutoConfig` class, then the config class is typically an instance of
    a custom class imported from the HF modules cache.

    Examples:

    >>> from transformers import AutoConfig
    >>> klass = AutoConfig.from_pretrained(
    ...     "meta-llama/Meta-Llama-3-8B", trust_remote_code=True
    ... )
    >>> klass.__class__  # transformers.models.llama.configuration_llama.LlamaConfig
    >>> import transformers_modules  # error, not initialized
    >>> klass = AutoConfig.from_pretrained(
    ...     "deepseek-ai/DeepSeek-V2.5", trust_remote_code=True
    ... )
    >>> import transformers_modules  # success, initialized
    >>> klass.__class__  # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

    In the DeepSeek example, the config class is an instance of a custom
    class that is not serializable by default. This class will not be
    importable in spawned workers, and won't exist at all on
    other nodes, which breaks serialization of the config.

    In this function we tell the cloudpickle serialization library to pass
    instances of these generated classes by value instead of by reference,
    i.e. the class definition is serialized along with its data so that the
    class module does not need to be importable on the receiving end.

    See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """  # noqa
    try:
        import transformers_modules

        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from irwkv.config import IrwkvConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of IrwkvConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: IrwkvConfig):
            return (pickle.loads, (cloudpickle.dumps(config),))

        multiprocessing.reducer.register(IrwkvConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # ray vendors its own version of cloudpickle
            from irwkv.v1.executor.ray_utils import ray

            if ray:
                ray.cloudpickle.register_pickle_by_value(transformers_modules)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e,
        )


def get_hf_image_processor_config(
    model: str | Path,
    hf_token: bool | str | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if envs.IRWKV_USE_MODELSCOPE:
        return dict()
    return get_image_processor_config(
        model, token=hf_token, revision=revision, **kwargs
    )


def get_hf_text_config(config: PreTrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    text_config = config.get_text_config()

    if text_config is not config and not hasattr(text_config, "num_attention_heads"):
        raise ValueError(
            "The text_config extracted from the model config does not have "
            "`num_attention_heads` attribute. This indicates a mismatch "
            "between the model config and irwkv's expectations. Please "
            "ensure that the model config is compatible with irwkv."
        )

    return text_config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    hf_token: bool | str | None = None,
    config_file_name: str = "generation_config.json",
) -> GenerationConfig | None:
    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
            token=hf_token,
            config_file_name=config_file_name,
        )
    except OSError:  # Not found
        if config_file_name != "generation_config.json":
            return None
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                config_format=config_format,
                token=hf_token,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
    model: str,
    *,
    revision: str | None = None,
):
    get_safetensors_metadata_partial = partial(
        hf_api().get_safetensors_metadata, model, revision=revision
    )

    try:
        return with_retry(
            get_safetensors_metadata_partial,
            "Error retrieving safetensors",
            fatal_errors=(huggingface_hub.errors.NotASafetensorsRepoError,),
        )
    except Exception:
        return None


def try_get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any] | None:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


def _read_safetensors_metadata_in_dir(local_dir: Path) -> dict[str, Any]:
    return {
        param_name: info
        for file_path in local_dir.glob("*.safetensors")
        if file_path.is_file()
        for param_name, info in parse_safetensors_file_metadata(file_path).items()
    }


def _read_safetensors_metadata_in_repo(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    api = hf_api()
    try:
        filenames = [
            filename
            for filename in api.list_repo_files(model, revision=revision)
            if filename.endswith(".safetensors")
        ]
        return {
            param_name: asdict(info)
            for filename in filenames
            for param_name, info in api.parse_safetensors_file_metadata(
                model, filename, revision=revision
            ).tensors.items()
        }
    except Exception as e:
        logger.debug("Could not read safetensors headers for %s: %s", model, e)
        return {}


def get_safetensors_params_metadata(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Get the safetensors parameters metadata for remote/local model repository."""
    if (model_path := Path(model)).exists():
        return _read_safetensors_metadata_in_dir(model_path)

    repo_mt = try_get_safetensors_metadata(model, revision=revision)
    if repo_mt and (files_mt := repo_mt.files_metadata):
        return {
            param_name: asdict(info)
            for file_mt in files_mt.values()
            for param_name, info in file_mt.tensors.items()
        }

    # `get_safetensors_metadata` only looks for `model.safetensors` and its
    # index, so older checkpoints naming their weights differently land here
    # even though their headers are perfectly readable.
    if metadata := _read_safetensors_metadata_in_repo(model, revision=revision):
        return metadata

    # Hub fetch failed (e.g. 429, network unreachable). Fall back to the
    # local HF cache: weights may already be cached from a prior run, and
    # weight loading itself uses the same cache.
    try:
        local_dir = hf_api().snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=["*.safetensors"],
            local_files_only=True,
        )
    except huggingface_hub.errors.LocalEntryNotFoundError as e:
        logger.warning_once(
            "Could not retrieve safetensors metadata for %s "
            "(Hub fetch failed and no local cache snapshot is available): %s.",
            model,
            str(e),
        )
        return {}

    metadata = _read_safetensors_metadata_in_dir(Path(local_dir))
    if not metadata:
        logger.warning_once(
            "Could not retrieve safetensors metadata for %s "
            "(Hub fetch failed and the cached snapshot holds no weights yet).",
            model,
        )
    return metadata


@cache
def checkpoint_has_lm_head(model: str, *, revision: str | None = None) -> bool | None:
    """Whether the checkpoint contains an `lm_head` tensor of its own.

    Args:
        model: Name or path of the model repository.
        revision: The specific model version to use.

    Returns:
        `None` if the checkpoint contents could not be determined, for example
        because it is not stored as safetensors.

    """
    metadata = get_safetensors_params_metadata(model, revision=revision)
    if not metadata:
        return None
    return any(name.endswith("lm_head.weight") for name in metadata)
