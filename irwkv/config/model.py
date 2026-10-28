# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import InitVar, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator

import irwkv.envs as envs
from irwkv.config.model_arch import (
    ModelArchitectureConfig,
)
from irwkv.config.multimodal import (
    MMCacheType,
    MMEncoderTPMode,
    MMHasherAlgorithm,
    MMProcessorDevice,
    MMTensorIPC,
    MultiModalConfig,
)
from irwkv.config.scheduler import RunnerType
from irwkv.config.utils import config, getattr_iter
from irwkv.logger import init_logger
from irwkv.platforms import current_platform
from irwkv.transformers_utils.config import (
    ConfigFormat,
    checkpoint_has_lm_head,
    get_config,
    get_hf_image_processor_config,
    get_hf_text_config,
    get_sentence_transformer_tokenizer_config,
    is_encoder_decoder,
    mrope_num_dims,
    try_get_generation_config,
    uses_mrope,
)
from irwkv.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from irwkv.transformers_utils.repo_utils import resolve_revision
from irwkv.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from irwkv.transformers_utils.utils import maybe_model_redirect
from irwkv.utils.import_utils import LazyLoader
from irwkv.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    import irwkv.model_executor.models as me_models
    from irwkv.config.load import LoadConfig
    from irwkv.config.parallel import ParallelConfig
    from irwkv.v1.sample.logits_processor import LogitsProcessor
else:
    PreTrainedConfig = Any

    me_models = LazyLoader("model_executor", globals(), "irwkv.model_executor.models")
    LoadConfig = Any
    ParallelConfig = Any
    LogitsProcessor = Any

logger = init_logger(__name__)

# Process-local record of which (arch, target) model-class overrides have been
# registered in *this* process. Must not live on ModelConfig: that instance is
# pickled to each worker, so an instance flag would arrive already "registered"
# while the worker's own global ModelRegistry is still untouched.
_REGISTERED_MODEL_CLASS_OVERRIDES: set[tuple[str, str]] = set()

RunnerOption = Literal["auto", "generate"]
ConvertType = Literal["none"]
ConvertOption = Literal["auto", ConvertType]
TokenizerMode = Literal["auto", "hf"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
PROCESSED_LOGPROBS_MODES: tuple[LogprobsMode, ...] = (
    "processed_logits",
    "processed_logprobs",
)
HfOverrides = dict[str, Any] | Callable[[PreTrainedConfig], PreTrainedConfig]
ModelImpl = Literal["auto", "irwkv"]
LayerBlockType = Literal["attention", "linear_attention", "mamba"]

AttnTypeStr = Literal[
    "decoder", "encoder", "encoder_only", "encoder_decoder", "attention_free", "hybrid"
]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""

    model: str = ""
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    model_weights: str = ""
    """Original model weights path. Used when the model is pulled from object
    storage (e.g., RunAI) to preserve the original URI while `model` points to
    the local directory."""
    runner: RunnerOption = "auto"
    """The type of model runner to use. Each irwkv instance only supports one
    model runner, even if the same model can be used for multiple types."""
    convert: ConvertOption = "auto"
    """Convert the model using adapters defined in
    [irwkv.model_executor.models.adapters][]. The most common use case is to
    adapt a text generation model to be used for pooling tasks."""
    tokenizer: str = None  # type: ignore[assignment]
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode | str = "auto"
    """Tokenizer mode: use `auto`, `hf`, or a plugin-registered mode."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: ModelDType | torch.dtype = "auto"
    """Data type for model weights and activations:

    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
      precision for BF16 models.
    - "half" for FP16. Recommended for AWQ quantization.
    - "float16" is the same as "half".
    - "bfloat16" for a balance between precision and range.
    - "float" is shorthand for FP32 precision.
    - "float32" for FP32 precision."""
    seed: int = 0
    """Random seed for reproducibility.

    We must set the global seed because otherwise,
    different tensor parallel workers would sample different tokens,
    leading to inconsistent results."""
    hf_config: PreTrainedConfig = field(init=False)
    """The Hugging Face config of the model."""
    hf_text_config: PreTrainedConfig = field(init=False)
    """The Hugging Face config of the text model (same as hf_config for text models)."""
    is_submodel_config: bool = field(default=False, init=False)
    """Whether this is a submodule view derived by `IrwkvConfig.with_hf_config`
    (e.g. a multimodal model's text stack). Its architecture list is empty, so
    deployment-level validation must not run against it."""
    word_embeddings_untied_by_checkpoint: bool = field(default=False, init=False)
    """Whether `tie_word_embeddings` was overridden to `False` because the checkpoint
    contains an `lm_head` of its own. The two may still turn out to be identical, in
    which case they are re-tied once the weights have been loaded."""
    hf_config_path: str | None = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    allowed_media_domains: list[str] | None = None
    """If set, only media URLs that belong to this domain can be used for
    multi-modal inputs. """
    revision: str | None = None
    """The specific model version to use. It can be a branch name, a tag name,
    or a commit id. If unspecified, will use the default version."""
    code_revision: str | None = None
    """The specific revision to use for the model code on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    tokenizer_revision: str | None = None
    """The specific revision to use for the tokenizer on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    max_model_len: int = Field(default=None, ge=-1)  # type: ignore[assignment]
    """Model context length (prompt and output). If unspecified, will be
    automatically derived from the model config.

    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format. Examples:

    - 1k -> 1000
    - 1K -> 1024
    - 25.6k -> 25,600
    - -1 or 'auto' -> Automatically choose the maximum model length that fits in
      GPU memory. This will use the model's maximum context length if it fits,
      otherwise it will find the largest length that can be accommodated."""
    enforce_eager: bool = False
    """Whether to always use eager-mode PyTorch. If True, we will disable CUDA
    graph and always execute the model in eager mode. If False, we will use
    CUDA graph and eager execution in hybrid for maximal performance and
    flexibility.

    NOTE: This disables both `torch.compile` and CUDA graphs, and is
    equivalent to setting `-cc.mode=none -cc.cudagraph_mode=none`."""
    return_sampling_mask: bool = False
    """Whether to return the post-processing token support for each sample."""
    max_logprobs: int = Field(default=20, ge=-1)
    """Maximum number of log probabilities to return when `logprobs` is
    specified in `SamplingParams`. The default value comes the default for the
    OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    vocab_size) logprobs are allowed to be returned and it may cause OOM."""
    logprobs_mode: LogprobsMode = "raw_logprobs"
    """Indicates the content returned in the logprobs and prompt_logprobs.
    Supported mode:
    1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    Raw means the values before applying any logit processors, like bad words.
    Processed means the values after applying all processors, including
    temperature and top_k/top_p.
    Note: for prompt_logprobs, processed_* and raw_* yield identical results
    because prompt tokens do not go through sampling processors.
    """
    use_fp64_gumbel: bool = False
    """Whether to use FP64 (instead of FP32) random noise for Gumbel-max and
    equivalent exponential-race sampling. FP64 preserves lower-tail sampling
    events that fp32 uniform/exponential draws can truncate, at the cost of
    significantly lower throughput on most GPUs."""
    enable_trace_replay: bool = False
    """Whether to allow requests to set
    `SamplingParams.trace_decode_token_ids`, which forces decoding to follow a
    predetermined token sequence while still computing real logprobs. Reserved
    for debugging and RL workflows: enabling it reserves a per-request trace
    buffer, so it is off by default."""
    disable_sliding_window: bool = False
    """Whether to disable sliding window. If True, we will disable the sliding
    window functionality of the model, capping to sliding window size. If the
    model does not support sliding window, this argument is ignored."""
    disable_cascade_attn: bool = True
    """Disable cascade attention for V1. While cascade attention does not
    change the mathematical correctness, disabling it could be useful for
    preventing potential numerical issues. This defaults to True, so users
    must opt in to cascade attention by setting this to False. Even when this
    is set to False, cascade attention will only be used when the heuristic
    tells that it's beneficial."""
    skip_tokenizer_init: bool = False
    """Skip initialization of tokenizer and detokenizer. Expects valid
    `prompt_token_ids` and `None` for prompt from the input. The generated
    output will contain token ids."""
    enable_prompt_embeds: bool = False
    """If `True`, enables passing text embeddings as inputs via the
    `prompt_embeds` key.

    WARNING: The irwkv engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!"""
    served_model_name: str | list[str] | None = None
    """The model name(s) used in the API. If multiple names are provided, the
    server will respond to any of the provided names. The model name in the
    model field of a response will be the first name in this list. If not
    specified, the model name will be the same as the `--model` argument. Noted
    that this name(s) will also be used in `model_name` tag content of
    prometheus metrics, if multiple names provided, metrics tag will take the
    first one."""
    config_format: str | ConfigFormat = "auto"
    """The format of the model config to load:

    - "auto" detects a Hugging Face config.
    - "hf" loads a Hugging Face config."""
    hf_token: bool | str | None = None
    """The token to use as HTTP bearer authorization for remote files . If
    `True`, will use the token generated when running `hf auth login`
    (stored in `~/.cache/huggingface/token`)."""
    hf_overrides: HfOverrides = field(default_factory=dict)
    """If a dictionary, contains arguments to be forwarded to the Hugging Face
    config. If a callable, it is called to update the HuggingFace config."""
    model_class_overrides: dict[str, str] = field(default_factory=dict)
    """Override the model class used for an architecture."""
    generation_config: str = "auto"
    """The folder path to the generation config. Defaults to `"auto"`, the
    generation config will be loaded from model path. If set to `"irwkv"`, no
    generation config is loaded, irwkv defaults will be used. If set to a folder
    path, the generation config will be loaded from the specified folder path.
    If `max_new_tokens` is specified in generation config, then it sets a
    server-wide limit on the number of output tokens for all requests."""
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    """Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If
    used with `--generation-config auto`, the override parameters will be
    merged with the default config from the model. If used with
    `--generation-config irwkv`, only the override parameters are used."""
    enable_sleep_mode: bool = False
    """Enable sleep mode for the engine (only cuda and
    hip platforms are supported)."""
    sleep_preserve_parameter_names: list[str] = field(default_factory=list)
    """Parameter-name globs to preserve across level-2 sleep.
    The sender must omit these parameters; loaders must preserve their storage."""
    sleep_mode_backend: str = "cumem"
    """Mechanism used to free and restore GPU state for sleep mode. ``"cumem"``
    (default) uses the built-in ``CuMemAllocator`` and is behavior-compatible
    with prior releases. Additional backends (CUDA checkpoint, CRIU, durable
    snapshot) may be registered in-tree or by plugins (RFC #34303)."""
    sleep_mode_offload_cudagraph: bool = False
    """Back up CUDA graph memory to CPU during sleep, restored in place on wake.
    Takes effect with enable_sleep_mode, the cumem backend and CUDA graphs."""
    enable_nccl_comm_suspend: bool = False
    """Enable releasing NCCL communicator memory during sleep mode
    (``ncclCommSuspend``/``ncclCommResume``). Experimental; when disabled
    (the default) sleep still releases weights/KV-cache memory as before."""
    enable_cumem_allocator: bool = False
    """Enable the custom cumem allocator to leverage advanced GPU memory
    allocation features such as multi-node NVLink support.

    Sleep mode automatically enables this allocator. Only cuda and hip
    platforms are supported.
    """
    model_impl: ModelImpl = "auto"
    """Select the RWKV model implementation."""
    logits_processors: list[str | type[LogitsProcessor]] | None = None
    """One or more logits processors' fully-qualified class names or class
    definitions"""
    io_processor_plugin: str | None = None
    """IOProcessor plugin name to load at model startup"""
    renderer_num_workers: int = 1
    """Number of worker threads in the renderer thread pool. The pool is
    consumed by the async renderer path (e.g. the OpenAI-compatible API
    server started by `irwkv serve`) to parallelize tokenization, chat
    template rendering, and multimodal preprocessing across concurrent
    requests.

    The offline `LLM` entrypoint uses the synchronous renderer path and
    processes prompts (including multimodal preprocessing) serially, so
    this setting has no effect there."""

    # Pooler config

    # Multimodal config and init vars
    multimodal_config: MultiModalConfig | None = None
    """Configuration for multimodal model. If `None`, this will be inferred
    from the architecture of `self.model`."""
    language_model_only: InitVar[bool] = False
    limit_mm_per_prompt: InitVar[dict[str, int | dict[str, int]] | None] = None
    enable_mm_embeds: InitVar[bool | None] = None
    media_io_kwargs: InitVar[dict[str, dict[str, Any]] | None] = None
    mm_processor_kwargs: InitVar[dict[str, Any] | None] = None
    mm_processor_cache_gb: InitVar[float | None] = None
    mm_processor_cache_type: InitVar[MMCacheType | None] = None
    mm_hasher_algorithm: InitVar[MMHasherAlgorithm | None] = None
    mm_shm_cache_max_object_size_mb: InitVar[int | None] = None
    mm_encoder_only: InitVar[bool | None] = None
    mm_encoder_tp_mode: InitVar[MMEncoderTPMode | None] = None
    mm_encoder_attn_backend: InitVar[AttentionBackendEnum | str | None] = None
    mm_encoder_attn_dtype: InitVar[str | None] = None
    mm_encoder_fp8_scale_path: InitVar[str | None] = None
    mm_encoder_fp8_scale_save_path: InitVar[str | None] = None
    mm_encoder_fp8_scale_save_margin: InitVar[float | None] = None
    interleave_mm_strings: InitVar[bool | None] = None
    skip_mm_profiling: InitVar[bool | None] = None
    video_pruning_rate: InitVar[float | None] = None
    video_pruning_method: InitVar[str | None] = None
    mm_tensor_ipc: InitVar[MMTensorIPC] = None
    mm_ipc_gpu_memory_gb: InitVar[float | None] = None
    mm_device_do_normalize: InitVar[bool | None] = None
    mm_processor_device: InitVar[MMProcessorDevice | None] = None

    def compute_hash(self) -> str:
        """WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        ignored_factors = {
            "convert",
            "tokenizer",
            "tokenizer_mode",
            "seed",
            "hf_config_path",
            "allowed_local_media_path",
            "allowed_media_domains",
            "tokenizer_revision",
            "enforce_eager",
            "return_sampling_mask",
            "logprobs_mode",
            "use_fp64_gumbel",
            "enable_trace_replay",
            "disable_cascade_attn",
            "skip_tokenizer_init",
            "served_model_name",
            "config_format",
            "hf_token",
            "hf_overrides",
            "logits_processors",
            "io_processor_plugin",
            "multimodal_config",
            "limit_mm_per_prompt",
            "media_io_kwargs",
            "mm_processor_kwargs",
            "mm_processor_cache_gb",
            "mm_processor_cache_type",
            "mm_shm_cache_max_object_size_mb",
            "mm_encoder_tp_mode",
            "interleave_mm_strings",
            "skip_mm_profiling",
            "mm_ipc_gpu_memory_gb",
        }

        from irwkv.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)

        # NOTE: For some models (e.g, Qwen3-VL), whether the MM code path is enabled
        # affects the computation graph of the language model, therefore we add it
        # here early.
        if self.multimodal_config:
            factors["language_model_only"] = self.multimodal_config.language_model_only
        return hash_factors(factors)

    def _update_nested(
        self,
        target: PreTrainedConfig | dict[str, Any],
        updates: dict[str, Any],
    ) -> None:
        """Recursively updates a config or dict with nested updates."""
        for key, value in updates.items():
            if isinstance(value, dict):
                # Get the nested target
                if isinstance(target, dict):
                    nested_target = target.get(key)
                else:
                    nested_target = getattr(target, key, None)

                # If nested target exists and can be updated recursively
                if nested_target is not None and (
                    isinstance(nested_target, dict)
                    or hasattr(nested_target, "__dict__")
                ):
                    self._update_nested(nested_target, value)
                    continue

            # Set the value (base case)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)

    def _apply_dict_overrides(
        self,
        config: PreTrainedConfig,
        overrides: dict[str, Any],
    ) -> None:
        """Apply dict overrides, handling both nested configs and dict values."""
        from transformers import PreTrainedConfig

        for key, value in overrides.items():
            attr = getattr(config, key, None)
            if attr is not None and isinstance(attr, PreTrainedConfig):
                # It's a nested config - recursively update it
                self._update_nested(attr, value)
            else:
                # It's a dict-valued parameter - set it directly
                setattr(config, key, value)

    def __post_init__(
        self,
        # Multimodal config init vars
        language_model_only: bool,
        limit_mm_per_prompt: dict[str, int | dict[str, int]] | None,
        enable_mm_embeds: bool | None,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
        mm_processor_kwargs: dict[str, Any] | None,
        mm_processor_cache_gb: float | None,
        mm_processor_cache_type: MMCacheType | None,
        mm_hasher_algorithm: MMHasherAlgorithm | None,
        mm_shm_cache_max_object_size_mb: int | None,
        mm_encoder_only: bool | None,
        mm_encoder_tp_mode: MMEncoderTPMode | None,
        mm_encoder_attn_backend: AttentionBackendEnum | str | None,
        mm_encoder_attn_dtype: str | None,
        mm_encoder_fp8_scale_path: str | None,
        mm_encoder_fp8_scale_save_path: str | None,
        mm_encoder_fp8_scale_save_margin: float | None,
        interleave_mm_strings: bool | None,
        skip_mm_profiling: bool | None,
        video_pruning_rate: float | None,
        video_pruning_method: str | None,
        mm_tensor_ipc: MMTensorIPC,
        mm_ipc_gpu_memory_gb: float | None,
        mm_device_do_normalize: bool | None,
        mm_processor_device: MMProcessorDevice | None,
    ) -> None:
        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(
            self.model, self.served_model_name
        )
        requested_revision = self.revision
        self.model = maybe_model_redirect(self.model)
        # The tokenizer is consistent with the model by default.
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        if callable(self.hf_overrides):
            hf_overrides_kw: dict[str, Any] = {}
            hf_overrides_fn = self.hf_overrides
            dict_overrides: dict[str, Any] = {}
        else:
            # Separate dict overrides from flat ones
            # We'll determine how to apply dict overrides after loading the config
            hf_overrides_kw = {}
            dict_overrides = {}
            for key, value in self.hf_overrides.items():
                if isinstance(value, dict):
                    dict_overrides[key] = value
                else:
                    hf_overrides_kw[key] = value
            hf_overrides_fn = None

        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        # If loading model/tokenizer from HF Hub, resolve the revision once
        # to prevent resolving it multiple times downstream. A resolved revision
        # only pins the repo it was resolved for, so each repo needs its own call.
        self.revision = resolve_revision(
            self.model,
            self.revision,
            self.hf_token,
        )

        # The config can live in another repo, which `self.revision` does not pin.
        # It stays `None` if the config comes from `self.model`, so that call sites
        # fall back to `self.revision` the same way they fall back to `self.model`.
        self._hf_config_revision = None
        if self.hf_config_path and self.hf_config_path != self.model:
            self._hf_config_revision = resolve_revision(
                self.hf_config_path,
                self.revision,
                self.hf_token,
            )

        if (
            self.tokenizer == self.model
            and self.tokenizer_revision == requested_revision
        ):
            self.tokenizer_revision = self.revision
        else:
            self.tokenizer_revision = resolve_revision(
                self.tokenizer,
                self.tokenizer_revision,
                self.hf_token,
            )

        if self.enable_sleep_mode:
            if not current_platform.is_sleep_mode_available():
                raise ValueError("Sleep mode is not supported on current platform.")
            if current_platform.is_cuda_alike() and not self.enable_cumem_allocator:
                logger.info_once(
                    "Enabling cumem allocator because sleep mode requires it."
                )
                self.enable_cumem_allocator = True
        if (
            self.enable_cumem_allocator
            and not current_platform.is_cumem_allocator_available()
        ):
            raise ValueError("cumem allocator is not supported on current platform.")

        hf_config = get_config(
            self.hf_config_path or self.model,
            self.trust_remote_code,
            self._hf_config_revision or self.revision,
            self.code_revision,
            self.config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
            token=self.hf_token,
        )
        self.hf_config = hf_config
        if dict_overrides:
            self._apply_dict_overrides(hf_config, dict_overrides)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.model_arch_config = self.get_model_arch_config()
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )
        self.encoder_config = self._get_encoder_config()
        # Image-processor metadata is only consumed by multimodal models.
        # Probing it for text-only models causes avoidable Hub requests.
        self.hf_image_processor_config: dict[str, Any] = {}

        architectures = self.architectures
        registry = self.registry
        self.runner_type = self._get_runner_type(
            architectures, self.runner, self.convert
        )
        self.convert_type = self._get_convert_type(
            architectures, self.runner_type, self.convert
        )

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            revision=self.revision,
            config_format=self.config_format,
        )

        # Some checkpoints set sliding_window to 0 to indicate that sliding window is
        # disabled, but irwkv uses None for that. Convert 0 to None to avoid errors.
        # Set before get_and_verify_max_len to ensure that max_model_len does not get
        # capped to 0.
        if self.get_sliding_window() == 0:
            self.disable_sliding_window = True
            self.hf_text_config.sliding_window = None

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)

        if self.is_encoder_decoder:
            mm_processor_cache_gb = 0
            logger.info("Encoder-decoder model detected, disabling mm processor cache.")

        # Init multimodal config if needed
        if self._model_info.supports_multimodal:
            self.hf_image_processor_config = get_hf_image_processor_config(
                self.model, hf_token=self.hf_token, revision=self.revision
            )
            if (
                mm_encoder_tp_mode == "data"
                and not self._model_info.supports_multimodal_encoder_tp_data
            ):
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`."
                )
                mm_encoder_tp_mode = "weights"

            mm_processor_kwargs = MultiModalConfig.fold_mm_processor_device(
                mm_processor_kwargs, mm_processor_device
            )

            mm_config_kwargs = dict(
                language_model_only=language_model_only,
                limit_per_prompt=limit_mm_per_prompt,
                enable_mm_embeds=enable_mm_embeds,
                media_io_kwargs=media_io_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                mm_processor_cache_gb=mm_processor_cache_gb,
                mm_processor_cache_type=mm_processor_cache_type,
                mm_hasher_algorithm=mm_hasher_algorithm,
                mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
                mm_encoder_only=mm_encoder_only,
                mm_encoder_tp_mode=mm_encoder_tp_mode,
                mm_encoder_attn_backend=mm_encoder_attn_backend,
                mm_encoder_attn_dtype=mm_encoder_attn_dtype,
                mm_encoder_fp8_scale_path=mm_encoder_fp8_scale_path,
                mm_encoder_fp8_scale_save_path=mm_encoder_fp8_scale_save_path,
                mm_encoder_fp8_scale_save_margin=mm_encoder_fp8_scale_save_margin,
                interleave_mm_strings=interleave_mm_strings,
                skip_mm_profiling=skip_mm_profiling,
                video_pruning_rate=video_pruning_rate,
                video_pruning_method=video_pruning_method,
                mm_tensor_ipc=mm_tensor_ipc,
                mm_ipc_gpu_memory_gb=mm_ipc_gpu_memory_gb,
                mm_device_do_normalize=self._resolve_mm_device_do_normalize(
                    mm_device_do_normalize
                ),
            )

            mm_config_kwargs = {
                k: v for k, v in mm_config_kwargs.items() if v is not None
            }

            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)  # type: ignore[arg-type]

            pruning_spec = self.multimodal_config.get_video_pruning_spec()
            supported_pruning = self._model_info.supported_video_pruning_methods
            if (
                pruning_spec is not None
                and supported_pruning
                and pruning_spec[0] not in supported_pruning
            ):
                raise ValueError(
                    f"Video pruning method '{pruning_spec[0]}' is not "
                    f"supported by {self._model_info.architecture} "
                    f"(supported methods: {supported_pruning})."
                )

            # Rebuild after multimodal_config exists so text-only mm_prefix
            # clearing is applied (and cached for later with_hf_config calls).
            self.model_arch_config = self.get_model_arch_config()

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            self.hf_text_config.sliding_window = None

        # Avoid running try_verify_and_update_config multiple times
        self.config_updated = False
        self._try_verify_and_update_model_config()
        self._verify_quantization()
        self._verify_cuda_graph()

    def _supports_multimodal_inputs(self) -> bool:
        return False

    def _cached_supports_multimodal_inputs(self) -> bool:
        return False

    def get_model_arch_config(self) -> ModelArchitectureConfig:
        convertor = ModelArchConfigConvertorBase(
            self.hf_config,
            self.hf_text_config,
        )
        return convertor.convert()

    @field_validator("tokenizer", "max_model_len", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    @field_validator("tokenizer_mode", mode="after")
    def _lowercase_tokenizer_mode(cls, tokenizer_mode: str) -> str:
        return tokenizer_mode.lower()

    @model_validator(mode="after")
    def validate_model_config_after(self: "ModelConfig") -> "ModelConfig":
        """Called after __post_init__."""
        if not isinstance(self.tokenizer, str):
            raise ValueError(
                f"tokenizer must be a string, got "
                f"{type(self.tokenizer).__name__}: {self.tokenizer!r}. "
                "Please provide a valid tokenizer path or HuggingFace model ID."
            )
        if not isinstance(self.max_model_len, int) or self.max_model_len < 1:
            raise ValueError(
                f"max_model_len must be a positive integer, "
                f"got {type(self.max_model_len).__name__}: {self.max_model_len!r}. "
                "Example: max_model_len=2048"
            )
        if self.enable_prompt_embeds and self.is_encoder_decoder:
            # No encoder-decoder model accepts `inputs_embeds`; their decoders
            # embed `input_ids` internally.
            raise ValueError(
                "--enable-prompt-embeds is not supported with encoder-decoder models."
            )
        return self

    def _resolve_mm_device_do_normalize(
        self, mm_device_do_normalize: bool | None
    ) -> bool:
        if mm_device_do_normalize is None:
            if envs.IRWKV_USE_RUST_FRONTEND:
                logger.debug(
                    "IRWKV_USE_RUST_FRONTEND is set. "
                    "Rust frontend does not currently support mm_device_do_normalize, "
                    "forcing mm_device_do_normalize = False."
                )
                mm_device_do_normalize = False
            else:
                mm_device_do_normalize = (
                    self._model_info.supports_mm_device_do_normalize
                )
                logger.debug(
                    "mm_device_do_normalize is %s by default.",
                    "enabled" if mm_device_do_normalize else "disabled",
                )
        else:
            if mm_device_do_normalize and envs.IRWKV_USE_RUST_FRONTEND:
                logger.warning(
                    "IRWKV_USE_RUST_FRONTEND is set. "
                    "Rust frontend does not currently support mm_device_do_normalize, "
                    "forcing mm_device_do_normalize = False."
                )
                mm_device_do_normalize = False

            if (
                mm_device_do_normalize
                and not self._model_info.supports_mm_device_do_normalize
            ):
                logger.warning(
                    "Model does not support mm_device_do_normalize, "
                    "forcing mm_device_do_normalize = False."
                )
                mm_device_do_normalize = False

            logger.debug(
                "mm_device_do_normalize is %s.",
                "enabled" if mm_device_do_normalize else "disabled",
            )

        return mm_device_do_normalize

    @property
    def registry(self):
        self._maybe_register_model_class_overrides()
        return me_models.ModelRegistry

    def _maybe_register_model_class_overrides(self) -> None:
        # Apply ``model_class_overrides`` here because this property is the
        # single chokepoint through which every model-class inspect/resolve
        # passes, in both the engine front-end and every worker process. The
        # guard is process-local (see ``_REGISTERED_MODEL_CLASS_OVERRIDES``), so
        # each worker re-registers into its own ModelRegistry exactly once
        # rather than trusting a pickled-in instance flag.
        if not self.model_class_overrides:
            return
        pending = [
            (arch, target)
            for arch, target in self.model_class_overrides.items()
            if (arch, target) not in _REGISTERED_MODEL_CLASS_OVERRIDES
        ]
        if not pending:
            return
        logger.warning_once(
            "Applying model_class_overrides %s. This is intended for "
            "development/debugging.",
            str(self.model_class_overrides),
        )
        for arch, target in pending:
            me_models.ModelRegistry.register_model(arch, target)
            _REGISTERED_MODEL_CLASS_OVERRIDES.add((arch, target))

    @property
    def architectures(self) -> list[str]:
        return self.model_arch_config.architectures

    @property
    def architecture(self) -> str:
        """The architecture irwkv actually used."""
        return self._architecture

    def maybe_pull_model_tokenizer_for_runai(self, model: str, tokenizer: str) -> None:
        """Pull model/tokenizer from Object Storage to temporary
        directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path

        """
        # Skip if model_weights is already set (model already pulled)
        if self.model_weights:
            return

        if not (is_runai_obj_uri(model) or is_runai_obj_uri(tokenizer)):
            return

        if is_runai_obj_uri(model):
            object_storage_model = ObjectStorageModel(url=model)
            object_storage_model.pull_files(
                model, allow_pattern=["*.model", "*.py", "*.json"]
            )
            self.model_weights = model
            self.model = object_storage_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                object_storage_model.pull_files(
                    model,
                    ignore_pattern=[
                        "*.pt",
                        "*.safetensors",
                        "*.bin",
                        "*.tensors",
                        "*.pth",
                    ],
                )
                self.tokenizer = object_storage_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_runai_obj_uri(tokenizer):
            object_storage_tokenizer = ObjectStorageModel(url=tokenizer)
            object_storage_tokenizer.pull_files(
                tokenizer,
                ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors", "*.pth"],
            )
            self.tokenizer = object_storage_tokenizer.dir

    def _get_encoder_config(self) -> dict[str, Any] | None:
        return get_sentence_transformer_tokenizer_config(self.model, self.revision)

    def maybe_untie_word_embeddings(self) -> None:
        """Stop trusting `tie_word_embeddings` when the checkpoint disagrees.

        A config may claim the word embeddings are tied while the checkpoint
        ships an `lm_head` of its own. Tying regardless would silently discard
        that tensor, so build the `lm_head` as if untied and let it load. The
        two are compared once loaded, and re-tied if they turn out to match, by
        [maybe_retie_word_embeddings][irwkv.model_executor.model_loader.weight_tying.maybe_retie_word_embeddings].

        Transformers makes the same decision in `PreTrainedModel.tie_weights`,
        where it can compare the two tensors directly.
        """
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            return
        if not checkpoint_has_lm_head(self.model, revision=self.revision):
            return

        logger.debug(
            "The config for %s says the word embeddings are tied, but the checkpoint "
            "contains an lm_head. Loading it to find out whether they really are tied.",
            self.model,
        )
        # Both levels must be updated: `IrwkvConfig.with_hf_config` reads the top
        # level, and the text config is what the language model itself sees.
        self.hf_config.tie_word_embeddings = False
        self.hf_config.get_text_config().tie_word_embeddings = False
        self.word_embeddings_untied_by_checkpoint = True

    def _get_default_runner_type(self, architectures: list[str]) -> RunnerType:
        return "generate"

    def _get_runner_type(
        self, architectures: list[str], runner: RunnerOption, convert: ConvertOption
    ) -> RunnerType:
        return "generate"

    def _get_default_convert_type(
        self, architectures: list[str], runner_type: RunnerType
    ) -> ConvertType:
        return "none"

    def _get_convert_type(
        self, architectures: list[str], runner_type: RunnerType, convert: ConvertOption
    ) -> ConvertType:
        return "none"

    def _verify_quantization(self) -> None:
        if self.model_arch_config.quantization_config is not None:
            raise ValueError("RWKV requires unquantized BF16 weights.")

    def _verify_cuda_graph(self) -> None:
        # CUDAGraph capture not supported for encoder-decoder models on ROCm
        unsupported_rocm = self.is_encoder_decoder
        if unsupported_rocm and not self.enforce_eager and current_platform.is_rocm():
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.",
                self.model_arch_config.model_type,
            )
            self.enforce_eager = True

    def _try_verify_and_update_model_config(self):
        # Avoid running try_verify_and_update_config multiple times
        if getattr(self, "config_updated", False):
            return

        architecture = self.architecture
        if architecture is None:
            return

        from irwkv.model_executor.models.config import (
            MODELS_CONFIG_MAP,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_model_config(self)

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
            self.architectures, self
        ):
            raise NotImplementedError(
                "Pipeline parallelism is not supported for this model. "
                "Supported models implement the `SupportsPP` interface."
            )

        decode_context_parallel_size = parallel_config.decode_context_parallel_size
        if decode_context_parallel_size > 1 and not self.use_mla:
            total_num_kv_heads = self.get_total_num_kv_heads()
            if tensor_parallel_size <= total_num_kv_heads:
                raise ValueError(
                    "Decode context parallelism for GQA/MQA requires "
                    f"`--tensor-parallel-size` ({tensor_parallel_size}) to be "
                    "greater than the model's total number of KV heads "
                    f"({total_num_kv_heads}). Increase `--tensor-parallel-size` "
                    "or set `--decode-context-parallel-size 1`."
                )

            max_dcp_size = tensor_parallel_size // total_num_kv_heads
            if decode_context_parallel_size > max_dcp_size:
                raise ValueError(
                    "`--decode-context-parallel-size` "
                    f"({decode_context_parallel_size}) exceeds the maximum "
                    f"supported value ({max_dcp_size}) for "
                    f"`--tensor-parallel-size` ({tensor_parallel_size}) and "
                    f"{total_num_kv_heads} model KV heads."
                )

            num_q_per_kv = total_num_attention_heads // total_num_kv_heads
            if num_q_per_kv % decode_context_parallel_size != 0:
                raise ValueError(
                    "The model's number of query heads per KV head "
                    f"({num_q_per_kv}) must be divisible by "
                    "`--decode-context-parallel-size` "
                    f"({decode_context_parallel_size}) for GQA/MQA."
                )

        # torch_shm uses a single IPC queue to rank 0; DP>1 is
        # incompatible because API servers can't know which
        # CoreEngine the scheduler will assign work to. TP>1 is
        # also not supported because this requires broadcasting
        # MM tensors between all TP ranks.
        if (
            self.multimodal_config is not None
            and self.multimodal_config.mm_tensor_ipc == "torch_shm"
            and parallel_config.world_size_across_dp > 1
        ):
            raise ValueError(
                "mm_tensor_ipc='torch_shm' is not supported with "
                "data_parallel_size > 1 or tensor_parallel_size > 1 "
                "or pipeline_parallel_size > 1."
            )

    def get_sliding_window(self) -> int | None:
        """Get the sliding window size from the HF text config if present."""
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.model_arch_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.model_arch_config.hidden_size

    def get_inputs_embeds_size(self) -> int:
        # The size of inputs_embeds is usually identical to the size
        # of the hidden states, however there are exceptions, such as
        # embedding models like CLIP and SigLIP
        names = ("projection_dim", "projection_size")
        return getattr_iter(
            self.hf_text_config, names, default_factory=self.get_hidden_size
        )

    @property
    def is_deepseek_mla(self) -> bool:
        return self.model_arch_config.is_deepseek_mla

    @cached_property
    def is_mm_prefix_lm(self) -> bool:
        return self.model_arch_config.is_mm_prefix_lm

    @property
    def rswa_window(self) -> int | None:
        return self.model_arch_config.rswa_window

    def get_head_size(self) -> int:
        return self.model_arch_config.head_size

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        return self.model_arch_config.total_num_kv_heads

    def get_num_kv_heads(
        self,
        parallel_config: ParallelConfig,
        arch_config: ModelArchitectureConfig | None = None,
    ) -> int:
        """Returns the number of KV heads per GPU.

        Pass ``arch_config`` (from ``model_arch_config[layer_idx]``) to size a
        single layer of a heterogeneous model rather than the model as a whole.
        """
        arch_config = arch_config or self.model_arch_config
        total_num_kv_heads = arch_config.total_num_kv_heads
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(
        self,
        parallel_config: ParallelConfig,
        arch_config: ModelArchitectureConfig | None = None,
    ) -> int:
        """Returns the number of attention heads per GPU.

        Pass ``arch_config`` (from ``model_arch_config[layer_idx]``) to size a
        single layer of a heterogeneous model rather than the model as a whole.
        """
        arch_config = arch_config or self.model_arch_config
        num_heads = arch_config.total_num_attention_heads
        return num_heads // parallel_config.tensor_parallel_size

    def get_total_num_hidden_layers(self) -> int:
        return self.model_arch_config.total_num_hidden_layers

    def get_layers_start_end_indices(
        self, parallel_config: ParallelConfig
    ) -> tuple[int, int]:
        from irwkv.distributed.utils import get_pp_indices

        total_num_hidden_layers = self.get_total_num_hidden_layers()

        # the layout order is: DP x PP x TP
        pp_rank = (
            parallel_config.rank // parallel_config.tensor_parallel_size
        ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: ParallelConfig,
        block_type: LayerBlockType = "attention",
    ) -> int:
        return 0 if block_type == "attention" else self.get_num_layers(parallel_config)

    def get_mamba_chunk_size(self) -> int:
        return 2048

    def get_multimodal_config(self) -> MultiModalConfig:
        """Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.

        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(
        self, config_file_name: str = "generation_config.json"
    ) -> dict[str, Any]:
        """This method attempts to retrieve the non-default values of the
        generation config for this model.

        The generation config can contain information about special tokens, as
        well as sampling parameters. Which is why this method exists separately
        to `get_diff_sampling_param`.

        Returns:
            A dictionary containing the non-default generation config.

        """
        if self.generation_config in {"auto", "irwkv"}:
            config = try_get_generation_config(
                self.hf_config_path or self.model,
                trust_remote_code=self.trust_remote_code,
                revision=self._hf_config_revision or self.revision,
                code_revision=self.code_revision,
                config_format=self.config_format,
                hf_token=self.hf_token,
                config_file_name=config_file_name,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
                code_revision=self.code_revision,
                config_format=self.config_format,
                hf_token=self.hf_token,
                config_file_name=config_file_name,
            )

        if config is None:
            return {}

        return config.to_diff_dict()

    def get_diff_sampling_param(
        self, config_file_name: str = "generation_config.json"
    ) -> dict[str, Any]:
        """This method returns a dictionary containing the non-default sampling
        parameters with `override_generation_config` applied.

        The default sampling parameters are:

        - irwkv's neutral defaults if `self.generation_config="irwkv"`
        - the model's defaults if `self.generation_config="auto"`
        - as defined in `generation_config.json` if
            `self.generation_config="path/to/generation_config/dir"`

        Returns:
            A dictionary containing the non-default sampling parameters.

        """
        src = self.generation_config

        config = (
            {}
            if src == "irwkv"
            else self.try_get_generation_config(config_file_name=config_file_name)
        )

        # Overriding with given generation config
        config.update(self.override_generation_config)

        available_params = [
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        if self.hf_config.model_type == "rwkv":
            available_params.extend(["penalty_decay", "stop_strings"])
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p) for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to irwkv's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
            if "stop_strings" in diff_sampling_param:
                diff_sampling_param["stop"] = diff_sampling_param.pop("stop_strings")
        else:
            diff_sampling_param = {}

        if diff_sampling_param and src != "irwkv":
            logger.warning_once(
                "Default irwkv sampling parameters have been overridden by %s: `%s`. "
                "If this is not intended, please relaunch irwkv instance "
                "with `--generation-config irwkv`.",
                "the model's `generation_config.json`" if src == "auto" else src,
                str(diff_sampling_param),
                scope="local",
            )

        return diff_sampling_param

    @cached_property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_alibi(self) -> bool:
        return False

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def mrope_num_dims(self) -> int:
        return mrope_num_dims(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_multimodal_raw_input_only_model(self) -> bool:
        return self._model_info.supports_multimodal_raw_input_only

    @property
    def supports_multimodal_inputs(self) -> bool:
        return self._cached_supports_multimodal_inputs()

    @property
    def requires_raw_input_tokens(self) -> bool:
        return self._model_info.requires_raw_input_tokens

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        return self._model_info.is_hybrid

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def supports_mamba_prefix_caching(self) -> bool:
        return self._model_info.supports_mamba_prefix_caching

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla

    @property
    def is_matryoshka(self) -> bool:
        return bool(getattr(self.hf_config, "matryoshka_dimensions", None)) or getattr(
            self.hf_config, "is_matryoshka", False
        )

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    @property
    def use_sep_token(self) -> bool:
        # cross_encoder models defaults to using separating token.
        # `llm as reranker` defaults to not using separating token.

        use_pad_token = getattr(self.hf_config, "use_pad_token", None)
        if use_pad_token is not None:
            logger.warning_once(
                "use_pad_token has been deprecated; please use use_sep_token instead."
            )
            return use_pad_token

        return getattr(self.hf_config, "use_sep_token", True)

    @property
    def head_dtype(self) -> torch.dtype:
        """The "head" refers to the last Linear layer(s) of an LLM,
        such as the lm_head in a generation model,
        or the score or classifier in a classification model.

        - Pooling models default to an fp32 head; use
          --hf-overrides '{"head_dtype": "model"}' to disable it.
        - Generation models default to the model dtype; set
          --hf-overrides '{"head_dtype": "float32"}' to run the lm_head in
          fp32, which is required for RL training-inference consistency
          (the trainer computes logits in fp32).
        """
        head_dtype = _get_head_dtype(
            config=self.hf_config, dtype=self.dtype, runner_type=self.runner_type
        )

        if head_dtype not in current_platform.supported_dtypes:
            logger.warning_once(
                "The current platform does not support [%s] head dtype, "
                "fallback to model dtype [%s].",
                head_dtype,
                self.dtype,
            )
            return self.dtype

        logger.debug_once("head dtype: %s", head_dtype)
        return head_dtype

    @property
    def embedding_size(self):
        return self.get_hidden_size()

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            model_arch_config=self.model_arch_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window=self.get_sliding_window(),
            encoder_config=self.encoder_config,
        )
        logger.info("Using max model len %s", max_model_len)
        return max_model_len

    @property
    def attn_type(self) -> AttnTypeStr:
        return (
            "attention_free" if self.is_attention_free else self._model_info.attn_type
        )

    @property
    def is_chunked_prefill_supported(self) -> bool:
        return True

    @property
    def is_prefix_caching_supported(self) -> bool:
        return self.attn_type == "decoder" or self.is_hybrid

    @property
    def is_quantized(self) -> bool:
        return getattr(self.hf_config, "quantization_config", None) is not None

    def is_nvfp4_quantized(self) -> bool:
        return False


def get_served_model_name(model: str, served_model_name: str | list[str] | None):
    """If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
    architecture: str,
    *,
    runner_type: RunnerType | None = None,
    convert_type: ConvertType | None = None,
) -> tuple[str, tuple[RunnerType, ConvertType]] | None:
    for suffix, (
        default_runner_type,
        default_convert_type,
    ) in iter_architecture_defaults():
        if (
            (runner_type is None or runner_type == default_runner_type)
            and (convert_type is None or convert_type == default_convert_type)
            and architecture.endswith(suffix)
        ):
            return suffix, (default_runner_type, default_convert_type)

    return None


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def str_dtype_to_torch_dtype(type: str):
    return _STR_DTYPE_TO_TORCH_DTYPE.get(type)


def _get_and_verify_dtype(
    model_id: str,
    config: PreTrainedConfig,
    dtype: str | torch.dtype,
    *,
    revision: str | None = None,
    config_format: str | ConfigFormat = "hf",
) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            return torch.bfloat16
        dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(dtype)
    if dtype != torch.bfloat16:
        raise ValueError("RWKV inference requires bfloat16 weights.")
    return dtype


def _get_head_dtype(
    config: PreTrainedConfig, dtype: torch.dtype, runner_type: str
) -> torch.dtype:
    head_dtype: str | torch.dtype | None = getattr(config, "head_dtype", None)

    if head_dtype == "model":
        return dtype
    elif isinstance(head_dtype, str):
        head_dtype = head_dtype.lower()
        if head_dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {head_dtype!r}")
        return _STR_DTYPE_TO_TORCH_DTYPE[head_dtype]
    elif isinstance(head_dtype, torch.dtype):
        return head_dtype
    elif head_dtype is None:
        if torch.float32 not in current_platform.supported_dtypes:
            return dtype
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {head_dtype}")


def _get_and_verify_max_len(
    hf_config: PreTrainedConfig,
    model_arch_config: ModelArchitectureConfig,
    tokenizer_config: dict | None,
    max_model_len: int | None,
    disable_sliding_window: bool,
    sliding_window: int | None,
    encoder_config: dict[str, Any] | None = None,
) -> int:
    """Get and verify the model's maximum length."""
    (derived_max_model_len, max_len_key) = (
        model_arch_config.derived_max_model_len_and_key
    )

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if (
        disable_sliding_window
        and sliding_window is not None
        and sliding_window < derived_max_model_len
    ):
        max_len_key = "sliding_window"
        derived_max_model_len = sliding_window

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len
        )
        derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the keys "
            "to determine the original maximum length of the model. "
            "Assuming the model's maximum length is %d.",
            default_max_len,
        )
        derived_max_model_len = default_max_len

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user didn't specify `max_model_len` or specified -1 (auto-fit),
    # then use that derived from the model config as a default value.
    # When -1 is specified, the engine will later auto-fit to available memory.
    if max_model_len is None or max_model_len == -1:
        max_model_len = int(derived_max_model_len)
        max_model_len = current_platform.check_max_model_len(max_model_len)

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is None or max_model_len > model_max_length:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json)."
            )
            warning = (
                "IRWKV_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
                "caution. If the model uses relative position encoding (RoPE), "
                "positions exceeding derived_max_model_len lead to nan. If the "
                "model uses absolute position encoding, positions exceeding "
                "derived_max_model_len will cause a CUDA array out-of-bounds "
                "error."
            )
            if envs.IRWKV_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning_once("%s %s", msg, warning)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    f"the env var IRWKV_ALLOW_LONG_MAX_MODEL_LEN=1. {warning}"
                )
    return int(max_model_len)
