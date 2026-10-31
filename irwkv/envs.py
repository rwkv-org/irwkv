# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    IRWKV_HOST_IP: str = ""
    IRWKV_PORT: int | None = None
    IRWKV_RPC_BASE_PATH: str = tempfile.gettempdir()
    IRWKV_USE_MODELSCOPE: bool = False
    IRWKV_USE_FASTOKENS: bool = False
    IRWKV_RINGBUFFER_WARNING_INTERVAL: int = 60
    IRWKV_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    IRWKV_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    IRWKV_ENGINE_READY_TIMEOUT_S: int = 600
    IRWKV_CHAT_TEMPLATE_RENDER_TIMEOUT: float = 30.0
    IRWKV_API_KEY: str | None = None
    IRWKV_DEBUG_LOG_API_SERVER_RESPONSE: bool = False
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_ENDPOINT_URL: str | None = None
    IRWKV_MODEL_REDIRECT_PATH: str | None = None
    IRWKV_CACHE_ROOT: str = os.path.expanduser("~/.cache/irwkv")
    IRWKV_CONFIG_ROOT: str = os.path.expanduser("~/.config/irwkv")
    IRWKV_CONFIGURE_LOGGING: bool = True
    IRWKV_LOGGING_LEVEL: str = "INFO"
    IRWKV_LOGGING_PREFIX: str = ""
    IRWKV_LOGGING_STREAM: str = "ext://sys.stdout"
    IRWKV_LOGGING_CONFIG_PATH: str | None = None
    IRWKV_LOGGING_COLOR: str = "auto"
    NO_COLOR: bool = False
    FORCE_COLOR: bool = False
    IRWKV_LOG_STATS_INTERVAL: float = 10.0
    IRWKV_TRACE_FUNCTION: int = 0
    IRWKV_USE_FLASHINFER_SAMPLER: bool = True
    IRWKV_PP_LAYER_PARTITION: str | None = None
    IRWKV_CPU_KVCACHE_SPACE: int | None = 0
    IRWKV_CPU_OMP_THREADS_BIND: str = "auto"
    IRWKV_CPU_NUM_OF_RESERVED_CPU: int | None = None
    IRWKV_CPU_CI_ENV: bool = False
    IRWKV_ZENTORCH_WEIGHT_PREPACK: bool = True
    IRWKV_CPU_INT4_W4A8: bool = True
    IRWKV_XLA_CACHE_PATH: str = os.path.join(IRWKV_CACHE_ROOT, "xla_cache")
    IRWKV_XLA_CHECK_RECOMPILATION: bool = False
    IRWKV_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN: int = 8192
    IRWKV_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"] = "auto"
    IRWKV_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    IRWKV_USE_RAY_WRAPPED_PP_COMM: bool = True
    IRWKV_XPU_PP_MICROBATCH: bool = True
    IRWKV_USE_RAY_V2_EXECUTOR_BACKEND: bool = False
    IRWKV_DISTRIBUTED_USE_SPLIT_GROUP: bool = False
    IRWKV_XLA_USE_SPMD: bool = False
    IRWKV_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    IRWKV_ASSETS_CACHE: str = os.path.join(IRWKV_CACHE_ROOT, "assets")
    IRWKV_ASSETS_CACHE_MODEL_CLEAN: bool = False
    IRWKV_IMAGE_FETCH_TIMEOUT: int = 5
    IRWKV_VIDEO_FETCH_TIMEOUT: int = 30
    IRWKV_AUDIO_FETCH_TIMEOUT: int = 10
    IRWKV_MEDIA_CACHE: str = ""
    IRWKV_MEDIA_CACHE_MAX_SIZE_MB: int = 5120
    IRWKV_MEDIA_CACHE_TTL_HOURS: float = 24
    IRWKV_MEDIA_FETCH_MAX_RETRIES: int = 3
    IRWKV_MAX_MEDIA_DOWNLOAD_SIZE_MB: int = 256
    IRWKV_MEDIA_URL_ALLOW_REDIRECTS: bool = True
    IRWKV_MEDIA_LOADING_THREAD_COUNT: int = 8
    IRWKV_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    IRWKV_MAX_AUDIO_DECODE_DURATION_S: int = 600
    IRWKV_MAX_AUDIO_DECODE_BYTES: int = 268_435_456
    IRWKV_MAX_AUDIO_PREPROCESS_WORKERS: int = max(1, min(os.cpu_count() or 1, 2))
    IRWKV_MAX_EMBED_DECODE_BYTES: int = 2_147_483_648
    IRWKV_MAX_IMAGE_PIXELS: int = 178_956_970
    IRWKV_VIDEO_LOADER_BACKEND: str = "opencv"
    IRWKV_MEDIA_CONNECTOR: str = "http"
    IRWKV_TARGET_DEVICE: str = "cuda"
    IRWKV_MAIN_CUDA_VERSION: str = "13.0"
    IRWKV_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
    IRWKV_BATCH_INVARIANT: bool = False
    IRWKV_TRITON_USE_TD: bool | None = None
    IRWKV_GPU_SYNC_CHECK: Literal["warn", "error"] | None = None
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    IRWKV_BUILD_COMMIT: str = "unknown"
    IRWKV_BUILD_PIPELINE: str = "local"
    IRWKV_BUILD_URL: str = ""
    IRWKV_IMAGE_TAG: str = ""
    IRWKV_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None
    VERBOSE: bool = False
    IRWKV_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    IRWKV_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    IRWKV_MAX_N_SEQUENCES: int = 16384
    IRWKV_MAX_COMPLETION_PROMPTS: int = 1024
    IRWKV_MAX_STOP_STRINGS: int = 4
    IRWKV_MAX_NUM_BAD_WORDS: int = 128
    IRWKV_MAX_BAD_WORDS_TOTAL_TOKENS: int = 1024
    IRWKV_PLUGINS: list[str] | None = None
    IRWKV_USE_AOT_COMPILE: bool = False
    IRWKV_USE_BYTECODE_HOOK: bool = True
    IRWKV_FORCE_AOT_LOAD: bool = False
    IRWKV_USE_MEGA_AOT_ARTIFACT: bool = False
    IRWKV_FASTSAFETENSORS_QUEUE_SIZE: int = 0
    IRWKV_TRITON_FORCE_FIRST_CONFIG: bool = False
    IRWKV_TRITON_JIT_WARMUP_NUM_THREADS: int = 4
    IRWKV_SKIP_P2P_CHECK: bool = False
    IRWKV_DISABLED_KERNELS: list[str] = []
    IRWKV_USE_HW_AGNOSTIC: bool = False
    IRWKV_DISABLE_PYNCCL: bool = False
    IRWKV_USE_OINK_OPS: bool = False
    IRWKV_MXFP4_EMULATION_DEQUANT_AT_LOAD: bool = False
    IRWKV_MXFP8_EMULATION_DEQUANT_AT_LOAD: bool = True
    IRWKV_ROCM_USE_AITER: bool = False
    IRWKV_ROCM_USE_AITER_CUSTOM_AR: bool = True
    IRWKV_ROCM_USE_AITER_LINEAR: bool = True
    IRWKV_ROCM_USE_AITER_LINEAR_HIPBMM: bool = False
    IRWKV_ROCM_USE_AITER_RMSNORM: bool = True
    IRWKV_ROCM_USE_AITER_MHA: bool = True
    IRWKV_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False
    IRWKV_ROCM_USE_AITER_TRITON_ROPE: bool = False
    IRWKV_ROCM_USE_AITER_FP8BMM: bool = True
    IRWKV_ROCM_USE_AITER_FP4BMM: bool = True
    IRWKV_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
    IRWKV_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False
    IRWKV_ROCM_USE_AITER_TRITON_GEMM: bool = True
    IRWKV_ROCM_USE_SKINNY_GEMM: bool = True
    IRWKV_ROCM_FP8_PADDING: bool = True
    IRWKV_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False
    IRWKV_ENABLE_V1_MULTIPROCESSING: bool = True
    IRWKV_LOG_BATCHSIZE_INTERVAL: float = -1
    IRWKV_DISABLE_COMPILE_CACHE: bool = False
    IRWKV_USE_LAYERNAME: bool = True
    IRWKV_USE_RUST_FRONTEND: bool = False
    IRWKV_USE_RUST_BENCH: bool = False
    IRWKV_RUST_FRONTEND_PATH: str | None = "auto"
    IRWKV_SERVER_DEV_MODE: bool = False
    IRWKV_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    IRWKV_RAY_PER_WORKER_GPUS: float = 1.0
    IRWKV_RAY_BUNDLE_INDICES: str = ""
    IRWKV_CUDART_SO_PATH: str | None = None
    IRWKV_DP_RANK: int = 0
    IRWKV_DP_RANK_LOCAL: int = -1
    IRWKV_DP_SIZE: int = 1
    IRWKV_USE_STANDALONE_COMPILE: bool = True
    IRWKV_ENABLE_PREGRAD_PASSES: bool = True
    IRWKV_USE_BREAKABLE_CUDAGRAPH: bool = False
    IRWKV_DP_MASTER_IP: str = ""
    IRWKV_DP_MASTER_PORT: int = 0
    IRWKV_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    IRWKV_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"
    IRWKV_RAY_DP_PLACEMENT_NODE_IPS: str = ""
    IRWKV_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str = ""
    IRWKV_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    IRWKV_MARLIN_USE_ATOMIC_ADD: bool = False
    IRWKV_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    IRWKV_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None
    IRWKV_HUMMING_INPUT_QUANT_CONFIG: dict[str, Any] | None = None
    IRWKV_HUMMING_USE_F16_ACCUM: bool = False
    IRWKV_DEEPEPLL_NVFP4_DISPATCH: bool = False
    IRWKV_V1_USE_OUTLINES_CACHE: bool = False
    IRWKV_TPU_USING_PATHWAYS: bool = False
    IRWKV_USE_DEEP_GEMM: bool = True
    IRWKV_USE_DEEP_GEMM_E8M0: bool = True
    IRWKV_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool = True
    IRWKV_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True
    IRWKV_FLASHINFER_AUTOTUNE_CACHE_DIR: str | None = None
    IRWKV_RWKV_TUNING_MODE: Literal["inspect", "require", "tune"] = "tune"
    IRWKV_RWKV_TUNING_LEGACY_DIR: str | None = None
    IRWKV_RWKV_SESSION_CACHE_SIZE: int = 128
    IRWKV_ALLOW_RUNTIME_RWKV_UPDATING: bool = False
    IRWKV_RWKV_UPLOAD_MAX_BYTES: int = 1 << 30
    IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS: list[str] | None = None
    IRWKV_FLASHINFER_ALLREDUCE_BACKEND: Literal["auto", "trtllm", "mnnvl"] = "auto"
    IRWKV_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024
    IRWKV_XGRAMMAR_CACHE_MB: int = 0
    IRWKV_REGEX_COMPILATION_TIMEOUT_S: int = 5
    IRWKV_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    IRWKV_ALLOW_INSECURE_SERIALIZATION: bool = False
    IRWKV_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False
    IRWKV_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    IRWKV_NIXL_SIDE_CHANNEL_PORT: int = 5600
    IRWKV_P2P_SIDE_CHANNEL_HOST: str = "localhost"
    IRWKV_P2P_SIDE_CHANNEL_PORT: int = 5710
    IRWKV_EC_SIDE_CHANNEL_HOST: str = "localhost"
    IRWKV_EC_SIDE_CHANNEL_PORT: int = 5601
    IRWKV_MOONCAKE_BOOTSTRAP_PORT: int = 8998
    IRWKV_MOONCAKE_STORE_TIER_LOG: bool = False
    IRWKV_MOONCAKE_LOAD_RECV_THREADS: int = 1
    IRWKV_MOONCAKE_DISK_STAGING_USABLE_RATIO: float = 0.9
    MOONCAKE_PREFERRED_SEGMENT: str | None = None
    MOONCAKE_REQUESTER_LOCAL_HOSTNAME: str | None = None
    IRWKV_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    IRWKV_ENFORCE_STRICT_TOOL_CALLING: bool = True
    IRWKV_MQ_MAX_CHUNK_BYTES_MB: int = 16
    IRWKV_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS: int = 5
    IRWKV_KV_CACHE_LAYOUT: (
        Literal["LBNHC", "LBHNC", "LHBNC", "NHD", "HND", "BLHNC", "BLNHC", "BHLNC"]
        | None
    ) = None
    IRWKV_SSM_CONV_STATE_LAYOUT: Literal["SD", "DS"] | None = None
    IRWKV_COMPUTE_NANS_IN_LOGITS: bool = False
    IRWKV_RAISE_ON_LOGIT_NANS: bool = False
    IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION: Literal[
        "FP", "INT8", "INT6", "INT4", "INT3", "NONE"
    ] = "NONE"
    IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
    IRWKV_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB: int | None = None
    IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB: int | None = None
    IRWKV_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480
    IRWKV_MOONCAKE_CONNECTOR_TIMEOUT: float = 30.0
    IRWKV_ENABLE_CUDAGRAPH_GC: bool = False
    IRWKV_LOOPBACK_IP: str = ""
    IRWKV_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True
    IRWKV_ENABLE_RESPONSES_API_STORE: bool = False
    IRWKV_HAS_FLASHINFER_CUBIN: bool = False
    IRWKV_ALLREDUCE_USE_SYMM_MEM: bool = True
    IRWKV_ALLREDUCE_USE_FLASHINFER: bool = True
    IRWKV_ALLREDUCE_USE_FLASHINFER_PCIE_IPC: bool = False
    IRWKV_TUNED_CONFIG_FOLDER: str | None = None
    IRWKV_ENABLE_STARTUP_PLAN: bool = False
    IRWKV_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()
    IRWKV_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False
    IRWKV_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False
    IRWKV_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    IRWKV_NVTX_SCOPES_FOR_PROFILING: bool = False
    IRWKV_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    IRWKV_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "IRWKV_OBJECT_STORAGE_SHM_BUFFER"
    IRWKV_DEEPEP_BUFFER_SIZE_MB: int = 1024
    IRWKV_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False
    IRWKV_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False
    IRWKV_DEEPEP_V2_ALLOW_HYBRID_MODE: bool = True
    IRWKV_DEEPEP_V2_PREFER_OVERLAP: bool = False
    IRWKV_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION: bool = False
    IRWKV_DBO_COMM_SMS: int = 20
    IRWKV_PATTERN_MATCH_DEBUG: str | None = None
    IRWKV_DEBUG_DUMP_PATH: str | None = None
    IRWKV_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True
    IRWKV_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True
    IRWKV_USE_NCCL_SYMM_MEM: bool = False
    IRWKV_NCCL_INCLUDE_PATH: str | None = None
    IRWKV_GC_DEBUG: str = ""
    IRWKV_DEBUG_WORKSPACE: bool = False
    IRWKV_DISABLE_SHARED_EXPERTS_STREAM: bool = False
    IRWKV_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256
    IRWKV_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"
    IRWKV_USE_V2_MODEL_RUNNER: bool | None = None
    IRWKV_LOG_MODEL_INSPECTION: bool = False
    IRWKV_DEBUG_MFU_METRICS: bool = False
    IRWKV_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool = False
    IRWKV_WEIGHT_OFFLOADING_DISABLE_UVA: bool = False
    IRWKV_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS: int = 0
    IRWKV_WSL2_ENABLE_PIN_MEMORY: bool = False
    IRWKV_ENABLE_CUDA_COMPATIBILITY: bool = False
    IRWKV_CUDA_COMPATIBILITY_PATH: str | None = None
    IRWKV_SKIP_MODEL_NAME_VALIDATION: bool = False
    """If set, irwkv will skip model name validation in API requests.
    This allows any model name to be accepted in the 'model' field of requests,
    making the server model-name agnostic. Useful for proxy/gateway scenarios."""
    IRWKV_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = True
    IRWKV_XPU_FORCE_N_CONTIG_WEIGHT: bool = False
    IRWKV_XPU_USE_SAMPLER_KERNEL: bool = True
    IRWKV_GPU_NIC_PCIE_MAPPING: str = ""
    IRWKV_NIC_SELECTION_VARS: str = ""
    IRWKV_ENABLE_HPC_OPS: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_json_str_or_file(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if os.path.exists(value):
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def disable_compile_cache() -> bool:
    return bool(int(os.getenv("IRWKV_DISABLE_COMPILE_CACHE", "0")))


def use_aot_compile() -> bool:
    from irwkv.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1"
        if is_torch_equal_or_newer("2.10.0") and not disable_compile_cache()
        else "0"
    )

    return os.environ.get("IRWKV_USE_AOT_COMPILE", default_value) == "1"


def use_mega_aot_artifact():
    from irwkv.utils.torch_utils import is_torch_equal_or_newer

    default_value = (
        "1" if is_torch_equal_or_newer("2.12.0.dev") and use_aot_compile() else "0"
    )

    return os.environ.get("IRWKV_USE_MEGA_AOT_ARTIFACT", default_value) == "1"


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """Create a lambda that validates environment variable against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict

    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """Create a lambda that validates environment variable
    containing comma-separated values against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default list of values if not set
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables
        dict that returns list of strings

    """

    def _get_validated_env_list() -> list[str]:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Split comma-separated values and strip whitespace
        values = [v.strip() for v in value.split(",") if v.strip()]

        if not values:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        # Validate each value
        for val in values:
            if not case_sensitive:
                check_value = val.lower()
                check_choices = [choice.lower() for choice in actual_choices]
            else:
                check_value = val
                check_choices = actual_choices

            if check_value not in check_choices:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Valid options: {actual_choices}."
                )

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """Creates a lambda which that validates environment variable
    containing comma-separated values against allowed choices which
    returns choices as a set.
    """

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def get_irwkv_port() -> int | None:
    """Get the port from IRWKV_PORT environment variable.

    Returns:
        The port number as an integer if IRWKV_PORT is set, None otherwise.

    Raises:
        ValueError: If IRWKV_PORT is a URI, suggest k8s service discovery issue.

    """
    if "IRWKV_PORT" not in os.environ:
        return None

    port = os.getenv("IRWKV_PORT", "0")

    try:
        return int(port)
    except ValueError as err:
        from urllib3.util import parse_url

        parsed = parse_url(port)
        if parsed.scheme:
            raise ValueError(
                f"IRWKV_PORT '{port}' appears to be a URI. "
                "This may be caused by Kubernetes service discovery."
            ) from None
        raise ValueError(f"IRWKV_PORT '{port}' must be a valid integer") from err


def _generate_shm_name() -> str:
    import pybase64

    # Fit macOS's 30-character limit without truncating the UUID.
    encoded_uuid = pybase64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii")
    return "irwkv_m_" + encoded_uuid.rstrip("=")


def get_env_or_set_default(
    env_name: str,
    default_factory: Callable[[], str],
) -> Callable[[], str]:
    """Create a lambda that returns an environment variable value if set,
    or generates and sets a default value using the provided factory function.
    """

    def _get_or_set_default() -> str:
        value = os.getenv(env_name)
        if value is not None:
            return value

        default_value = default_factory()
        os.environ[env_name] = default_value
        return default_value

    return _get_or_set_default


# The start-* and end* here are used by the documentation generator
# to extract the used env vars.

# --8<-- [start:env-vars-definition]

logger = logging.getLogger(__name__)


def _resolve_rust_cli_path() -> str | None:
    """Resolve the irwkv-rs binary path.

    Returns None unless IRWKV_USE_RUST_FRONTEND or IRWKV_USE_RUST_BENCH is enabled.
    When enabled, resolves IRWKV_RUST_FRONTEND_PATH ("auto" by default)
    to the actual binary path.
    """
    use_rust = bool(int(os.environ.get("IRWKV_USE_RUST_FRONTEND", "0"))) or bool(
        int(os.environ.get("IRWKV_USE_RUST_BENCH", "0"))
    )
    raw = os.environ.get("IRWKV_RUST_FRONTEND_PATH", "auto")

    if not use_rust:
        if os.environ.get("IRWKV_RUST_FRONTEND_PATH") is not None:
            logger.warning(
                "IRWKV_RUST_FRONTEND_PATH is set without enabling "
                "IRWKV_USE_RUST_FRONTEND or IRWKV_USE_RUST_BENCH. "
                "Set one of them to 1 to use the irwkv-rs binary."
            )
        return None

    if raw.lower() in ("auto", "1", "true"):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(pkg_dir, "irwkv-rs")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

        raise FileNotFoundError(
            "IRWKV_RUST_FRONTEND_PATH=auto but the irwkv-rs binary was "
            f"not found at {candidate}. "
            "Build with setuptools-rust or set the path explicitly."
        )
    return raw


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device for the CUDA build.
    "IRWKV_TARGET_DEVICE": lambda: os.getenv("IRWKV_TARGET_DEVICE", "cuda").lower(),
    # Main CUDA version of irwkv. This follows PyTorch but can be overridden.
    "IRWKV_MAIN_CUDA_VERSION": lambda: (
        os.getenv("IRWKV_MAIN_CUDA_VERSION", "").lower() or "13.0"
    ),
    # Controls PyTorch float32 matmul precision mode within irwkv workers.
    # Valid options mirror torch.set_float32_matmul_precision
    "IRWKV_FLOAT32_MATMUL_PRECISION": env_with_choices(
        "IRWKV_FLOAT32_MATMUL_PRECISION",
        "highest",
        ["highest", "high", "medium"],
        case_sensitive=False,
    ),
    # Enable batch-invariant mode: deterministic results regardless of
    # batch composition. Requires NVIDIA GPU with compute capability >= 9.0.
    "IRWKV_BATCH_INVARIANT": lambda: bool(int(os.getenv("IRWKV_BATCH_INVARIANT", "0"))),
    # Use tensor descriptors for Q/K/V loads and output stores in the
    # Triton unified-attention kernel.  Enables HW 2D block reads on
    # Intel XPU; the non-TD branch is dead-code-eliminated at Triton
    # compile time so other platforms see no overhead.  Tri-state override:
    # unset (default) lets the `triton_attn` backend auto-select per
    # platform (currently auto-enabled on XPU only); ``1`` forces TD on;
    # ``0`` forces TD off.  Useful for A/B benchmarking the TD path.
    "IRWKV_TRITON_USE_TD": lambda: {"1": True, "0": False}.get(
        os.getenv("IRWKV_TRITON_USE_TD", "").strip()
    ),
    # If set, enable GPU<->CPU synchronization checking around the worker's
    # `execute_model` and `sample_tokens` calls, via PyTorch's sync debug mode
    # plus wrappers flagging `non_blocking` CPU<->CUDA copies that silently
    # block the host (CPU tensors that are not pinned or not densely laid out).
    # Valid values are "warn" (warn on each sync) or "error" (raise on sync).
    # Unset disables the check. See `torch.cuda.set_sync_debug_mode`.
    "IRWKV_GPU_SYNC_CHECK": env_with_choices(
        "IRWKV_GPU_SYNC_CHECK", None, ["warn", "error"]
    ),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # Build provenance metadata embedded in official irwkv-openai images.
    # Set via Docker ENV at image build time; informational only.
    "IRWKV_BUILD_COMMIT": lambda: os.environ.get("IRWKV_BUILD_COMMIT", "unknown"),
    "IRWKV_BUILD_PIPELINE": lambda: os.environ.get("IRWKV_BUILD_PIPELINE", "local"),
    "IRWKV_BUILD_URL": lambda: os.environ.get("IRWKV_BUILD_URL", ""),
    "IRWKV_IMAGE_TAG": lambda: os.environ.get("IRWKV_IMAGE_TAG", ""),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": env_with_choices(
        "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]
    ),
    # If set, irwkv will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # Root directory for irwkv configuration files
    # Defaults to `~/.config/irwkv` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how irwkv finds its configuration files
    # during runtime, but also affects how irwkv installs its configuration
    # files during **installation**.
    "IRWKV_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "IRWKV_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "irwkv"),
        )
    ),
    # ================== Runtime Env Vars ==================
    # Root directory for irwkv cache files
    # Defaults to `~/.cache/irwkv` unless `XDG_CACHE_HOME` is set
    "IRWKV_CACHE_ROOT": lambda: os.path.expanduser(
        os.getenv(
            "IRWKV_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "irwkv"),
        )
    ),
    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "IRWKV_HOST_IP": lambda: os.getenv("IRWKV_HOST_IP", ""),
    # used in distributed environment to manually set the communication port
    # Note: if IRWKV_PORT is set, and some code asks for multiple ports, the
    # IRWKV_PORT will be used as the first port, and the rest will be generated
    # by incrementing the IRWKV_PORT value.
    "IRWKV_PORT": get_irwkv_port,
    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    "IRWKV_RPC_BASE_PATH": lambda: os.getenv(
        "IRWKV_RPC_BASE_PATH", tempfile.gettempdir()
    ),
    # If true, will load models from ModelScope instead of Hugging Face Hub.
    "IRWKV_USE_MODELSCOPE": lambda: (
        os.environ.get("IRWKV_USE_MODELSCOPE", "False").strip().lower() in ("1", "true")
    ),
    # If true, replace the Rust BPE backend that powers HF fast tokenizers
    # with the `fastokens` (https://github.com/crusoecloud/fastokens) shim.
    # Applies to any tokenizer mode that loads an HF fast tokenizer
    # (`hf`, `deepseek_v32`, `deepseek_v4`, …). The `fastokens`
    # Python package must be installed.
    "IRWKV_USE_FASTOKENS": lambda: bool(int(os.getenv("IRWKV_USE_FASTOKENS", "0"))),
    # Interval in seconds to log a warning message when the ring buffer is full
    "IRWKV_RINGBUFFER_WARNING_INTERVAL": lambda: int(
        os.environ.get("IRWKV_RINGBUFFER_WARNING_INTERVAL", "60")
    ),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "IRWKV_NCCL_SO_PATH": lambda: os.environ.get("IRWKV_NCCL_SO_PATH", None),
    # when `IRWKV_NCCL_SO_PATH` is not set, irwkv will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # flag to control the chunk size (in MB) for sleeping memory allocations under ROCm
    "IRWKV_ROCM_SLEEP_MEM_CHUNK_SIZE": lambda: int(
        os.environ.get("IRWKV_ROCM_SLEEP_MEM_CHUNK_SIZE", "256")
    ),
    # Feature flag to enable/disable Inductor standalone compile.
    # In torch <= 2.7 we ignore this flag; in torch >= 2.9 this is
    # enabled by default.
    "IRWKV_USE_STANDALONE_COMPILE": lambda: (
        os.environ.get("IRWKV_USE_STANDALONE_COMPILE", "1") == "1"
    ),
    # Inductor's pre-grad passes don't do anything for irwkv.
    # The pre-grad passes get run even on cache-hit and negatively impact
    # irwkv cold compile times by O(1s)
    # Can remove this after the following issue gets fixed
    # TODO(luka): maybe_inplace requires this
    # https://github.com/pytorch/pytorch/issues/174502
    "IRWKV_ENABLE_PREGRAD_PASSES": lambda: (
        os.environ.get("IRWKV_ENABLE_PREGRAD_PASSES", "1") == "1"
    ),
    # Experimental: breakable cudagraph does not rely on torch.compile
    "IRWKV_USE_BREAKABLE_CUDAGRAPH": lambda: (
        os.environ.get("IRWKV_USE_BREAKABLE_CUDAGRAPH", "0") == "1"
    ),
    # Debug pattern matching inside custom passes.
    # Should be set to the fx.Node name (e.g. 'getitem_34' or 'scaled_mm_3').
    "IRWKV_PATTERN_MATCH_DEBUG": lambda: os.environ.get(
        "IRWKV_PATTERN_MATCH_DEBUG", None
    ),
    # Dump fx graphs to the given directory.
    # It will override CompilationConfig.debug_dump_path if set.
    "IRWKV_DEBUG_DUMP_PATH": lambda: os.environ.get("IRWKV_DEBUG_DUMP_PATH", None),
    # Feature flag to enable/disable AOT compilation. This will ensure
    # compilation is done in warmup phase and the compilation will be
    # reused in subsequent calls.
    "IRWKV_USE_AOT_COMPILE": use_aot_compile,
    # Feature flag to enable/disable bytecode in
    # TorchCompileWithNoGuardsWrapper.
    "IRWKV_USE_BYTECODE_HOOK": lambda: bool(
        int(os.environ.get("IRWKV_USE_BYTECODE_HOOK", "1"))
    ),
    # Force irwkv to always load AOT compiled models from disk. Failure
    # to load will result in a hard error when this is enabled.
    # Will be ignored when IRWKV_USE_AOT_COMPILE is disabled.
    "IRWKV_FORCE_AOT_LOAD": lambda: os.environ.get("IRWKV_FORCE_AOT_LOAD", "0") == "1",
    # Enable loading compiled models directly from cached standalone compile artifacts
    # without re-splitting graph modules. This reduces overhead during model
    # loading by using reconstruct_serializable_fn_from_mega_artifact.
    "IRWKV_USE_MEGA_AOT_ARTIFACT": use_mega_aot_artifact,
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # Timeout in seconds for waiting for engine cores to become ready
    # during startup. Default is 600 seconds (10 minutes).
    "IRWKV_ENGINE_READY_TIMEOUT_S": lambda: int(
        os.environ.get("IRWKV_ENGINE_READY_TIMEOUT_S", "600")
    ),
    # Maximum wall-clock seconds allowed for a single chat template render.
    # Set to 0 to disable the timeout.
    "IRWKV_CHAT_TEMPLATE_RENDER_TIMEOUT": lambda: float(
        os.environ.get("IRWKV_CHAT_TEMPLATE_RENDER_TIMEOUT", "30")
    ),
    # API key for irwkv API server
    "IRWKV_API_KEY": lambda: os.environ.get("IRWKV_API_KEY", None),
    # Whether to log responses from API Server for debugging
    "IRWKV_DEBUG_LOG_API_SERVER_RESPONSE": lambda: (
        os.environ.get("IRWKV_DEBUG_LOG_API_SERVER_RESPONSE", "False").lower() == "true"
    ),
    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID": lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY": lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL": lambda: os.environ.get("S3_ENDPOINT_URL", None),
    # Logging configuration
    # If set to 0, irwkv will not configure logging
    # If set to 1, irwkv will configure logging using the default configuration
    #    or the configuration file specified by IRWKV_LOGGING_CONFIG_PATH
    "IRWKV_CONFIGURE_LOGGING": lambda: bool(
        int(os.getenv("IRWKV_CONFIGURE_LOGGING", "1"))
    ),
    "IRWKV_LOGGING_CONFIG_PATH": lambda: os.getenv("IRWKV_LOGGING_CONFIG_PATH"),
    # this is used for configuring the default logging level
    "IRWKV_LOGGING_LEVEL": lambda: os.getenv("IRWKV_LOGGING_LEVEL", "INFO").upper(),
    # this is used for configuring the default logging stream
    "IRWKV_LOGGING_STREAM": lambda: os.getenv(
        "IRWKV_LOGGING_STREAM", "ext://sys.stdout"
    ),
    # if set, IRWKV_LOGGING_PREFIX will be prepended to all log messages
    "IRWKV_LOGGING_PREFIX": lambda: os.getenv("IRWKV_LOGGING_PREFIX", ""),
    # Controls colored logging output. Options: "auto" (default, colors when terminal),
    # "1" (always use colors), "0" (never use colors)
    "IRWKV_LOGGING_COLOR": lambda: os.getenv("IRWKV_LOGGING_COLOR", "auto"),
    # Standard unix flag for disabling ANSI color codes
    "NO_COLOR": lambda: os.getenv("NO_COLOR", "0") != "0",
    # De-facto standard flag for forcing ANSI color codes (e.g. non-tty case)
    "FORCE_COLOR": lambda: os.getenv("FORCE_COLOR", "0") != "0",
    # If set, irwkv will log stats at this interval in seconds
    # If not set, irwkv will log stats every 10 seconds.
    "IRWKV_LOG_STATS_INTERVAL": lambda: (
        val
        if (val := float(os.getenv("IRWKV_LOG_STATS_INTERVAL", "10."))) > 0.0
        else 10.0
    ),
    # Trace function calls
    # If set to 1, irwkv will trace function calls
    # Useful for debugging
    "IRWKV_TRACE_FUNCTION": lambda: int(os.getenv("IRWKV_TRACE_FUNCTION", "0")),
    # Whether to use the FlashInfer top-k / top-p sampler on CUDA. Enabled
    # by default when the hardware supports it — set to 0 to opt out
    # explicitly, which forces the PyTorch-native (Triton for bs>=8) path.
    "IRWKV_USE_FLASHINFER_SAMPLER": lambda: (
        bool(int(os.environ["IRWKV_USE_FLASHINFER_SAMPLER"]))
        if "IRWKV_USE_FLASHINFER_SAMPLER" in os.environ
        else True
    ),
    # Pipeline stage partition strategy
    "IRWKV_PP_LAYER_PARTITION": lambda: os.getenv("IRWKV_PP_LAYER_PARTITION", None),
    # (CPU backend only) CPU key-value cache space.
    # default is None and will be set as 4 GB
    "IRWKV_CPU_KVCACHE_SPACE": lambda: (
        int(os.getenv("IRWKV_CPU_KVCACHE_SPACE", "0"))
        if "IRWKV_CPU_KVCACHE_SPACE" in os.environ
        else None
    ),
    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "IRWKV_CPU_OMP_THREADS_BIND": lambda: os.getenv(
        "IRWKV_CPU_OMP_THREADS_BIND", "auto"
    ),
    # (CPU backend only) CPU cores not used by OMP threads .
    # Those CPU cores will not be used by OMP threads of a rank.
    "IRWKV_CPU_NUM_OF_RESERVED_CPU": lambda: (
        int(os.getenv("IRWKV_CPU_NUM_OF_RESERVED_CPU", "0"))
        if "IRWKV_CPU_NUM_OF_RESERVED_CPU" in os.environ
        else None
    ),
    # (CPU backend only) whether irwkv is running in a CI environment.
    "IRWKV_CPU_CI_ENV": lambda: bool(int(os.getenv("IRWKV_CPU_CI_ENV", "0"))),
    # (Zen CPU backend) eagerly prepack weights into ZenDNN blocked layout
    # at model load time. Eliminates per-inference layout conversion overhead.
    "IRWKV_ZENTORCH_WEIGHT_PREPACK": lambda: bool(
        int(os.getenv("IRWKV_ZENTORCH_WEIGHT_PREPACK", "1"))
    ),
    # (CPU backend only) whether to use SGLang INT4 W4A8 kernels for AWQ, and
    # on Zen CPUs whether to serve int4 checkpoints as DA8W4 rather than W4A16.
    "IRWKV_CPU_INT4_W4A8": lambda: bool(int(os.getenv("IRWKV_CPU_INT4_W4A8", "1"))),
    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    "IRWKV_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(
        "IRWKV_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto", ["auto", "nccl", "shm"]
    ),
    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph.
    "IRWKV_USE_RAY_COMPILED_DAG_OVERLAP_COMM": lambda: bool(
        int(os.getenv("IRWKV_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
    ),
    # If the env var is set, it uses a Ray Communicator wrapping
    # irwkv's pipeline parallelism communicator to interact with Ray's
    # Compiled Graph. Otherwise, it uses Ray's NCCL communicator.
    "IRWKV_USE_RAY_WRAPPED_PP_COMM": lambda: bool(
        int(os.getenv("IRWKV_USE_RAY_WRAPPED_PP_COMM", "1"))
    ),
    # Using a flag to control microbatch on XPU device, users on XPU device can
    # decide to disable it when they needed.
    "IRWKV_XPU_PP_MICROBATCH": lambda: bool(
        int(os.getenv("IRWKV_XPU_PP_MICROBATCH", "1"))
    ),
    # When True and distributed_executor_backend="ray", use RayExecutorV2
    # (MQ-based) instead of RayDistributedExecutor (compiled-graph backend).
    "IRWKV_USE_RAY_V2_EXECUTOR_BACKEND": lambda: bool(
        int(os.getenv("IRWKV_USE_RAY_V2_EXECUTOR_BACKEND", "1"))
    ),
    # When True, GroupCoordinator constructs its CPU/device subgroups via
    # ``torch.distributed.split_group(backend=...)``
    # and ``init_distributed_environment`` initializes the default PG with
    # mixed ``cpu:gloo,cuda:nccl`` backend + eager ``device_id`` binding.
    "IRWKV_DISTRIBUTED_USE_SPLIT_GROUP": lambda: bool(
        int(os.getenv("IRWKV_DISTRIBUTED_USE_SPLIT_GROUP", "0"))
    ),
    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "IRWKV_WORKER_MULTIPROC_METHOD": env_with_choices(
        "IRWKV_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]
    ),
    # Path to the cache for storing downloaded assets
    "IRWKV_ASSETS_CACHE": lambda: os.path.expanduser(
        os.getenv(
            "IRWKV_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "irwkv", "assets"),
        )
    ),
    # If the env var is set, we will clean model file in
    # this path $IRWKV_ASSETS_CACHE/model_streamer/$model_name
    "IRWKV_ASSETS_CACHE_MODEL_CLEAN": lambda: bool(
        int(os.getenv("IRWKV_ASSETS_CACHE_MODEL_CLEAN", "0"))
    ),
    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "IRWKV_IMAGE_FETCH_TIMEOUT": lambda: int(
        os.getenv("IRWKV_IMAGE_FETCH_TIMEOUT", "5")
    ),
    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "IRWKV_VIDEO_FETCH_TIMEOUT": lambda: int(
        os.getenv("IRWKV_VIDEO_FETCH_TIMEOUT", "30")
    ),
    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "IRWKV_AUDIO_FETCH_TIMEOUT": lambda: int(
        os.getenv("IRWKV_AUDIO_FETCH_TIMEOUT", "10")
    ),
    # Directory for caching media downloads (images, video, audio fetched
    # from URLs during inference). Empty string disables caching.
    "IRWKV_MEDIA_CACHE": lambda: os.getenv("IRWKV_MEDIA_CACHE", ""),
    # Maximum cache size in MB. When exceeded, least-recently-used entries
    # are evicted. Default is 5120 (5 GB).
    "IRWKV_MEDIA_CACHE_MAX_SIZE_MB": lambda: int(
        os.getenv("IRWKV_MEDIA_CACHE_MAX_SIZE_MB", "5120")
    ),
    # Time-to-live in hours for cached media files. Entries older than this
    # are evicted regardless of cache size. Default is 24 hours.
    "IRWKV_MEDIA_CACHE_TTL_HOURS": lambda: float(
        os.getenv("IRWKV_MEDIA_CACHE_TTL_HOURS", "24")
    ),
    # Maximum number of retries for fetching media (images, audio, video)
    # from URLs. Each retry quadruples the timeout. Default is 3.
    "IRWKV_MEDIA_FETCH_MAX_RETRIES": lambda: int(
        os.getenv("IRWKV_MEDIA_FETCH_MAX_RETRIES", "3")
    ),
    # Maximum size in MB for a single remote media download. The limit is
    # enforced while streaming the response body so oversized or infinite
    # responses cannot grow the API server heap without bound. Default is 256.
    "IRWKV_MAX_MEDIA_DOWNLOAD_SIZE_MB": lambda: int(
        os.getenv("IRWKV_MAX_MEDIA_DOWNLOAD_SIZE_MB", "256")
    ),
    # Whether to allow HTTP redirects when fetching from media URLs.
    # Default to True
    "IRWKV_MEDIA_URL_ALLOW_REDIRECTS": lambda: bool(
        int(os.getenv("IRWKV_MEDIA_URL_ALLOW_REDIRECTS", "1"))
    ),
    # Max number of workers for the thread pool handling
    # media bytes loading. Set to 1 to disable parallel processing.
    # Default is 8
    "IRWKV_MEDIA_LOADING_THREAD_COUNT": lambda: int(
        os.getenv("IRWKV_MEDIA_LOADING_THREAD_COUNT", "8")
    ),
    # Maximum filesize in MB for a single audio file. Enforced on all
    # audio inputs (multimodal chat, speech-to-text uploads, data: URLs,
    # and local file:// paths). Files larger than this will be rejected
    # before decoding. Default is 25 MB.
    "IRWKV_MAX_AUDIO_CLIP_FILESIZE_MB": lambda: int(
        os.getenv("IRWKV_MAX_AUDIO_CLIP_FILESIZE_MB", "25")
    ),
    # Maximum decoded audio duration in seconds.  Compressed audio files
    # (e.g. OPUS at very low bitrate) can expand into gigabytes of float32
    # PCM.  This limit is enforced *during* decoding so the memory is never
    # allocated.  Default is 600s (10 minutes).
    "IRWKV_MAX_AUDIO_DECODE_DURATION_S": lambda: int(
        os.getenv("IRWKV_MAX_AUDIO_DECODE_DURATION_S", "600")
    ),
    # Maximum float32 PCM bytes that audio decoding may allocate.
    # Guards against sample-rate forgery where an attacker inflates
    # the header sample rate to bypass the duration guard while the
    # actual frame count still causes a multi-GiB allocation.
    # Default is 256 MiB (sufficient for 600s mono 48 kHz float32).
    "IRWKV_MAX_AUDIO_DECODE_BYTES": lambda: int(
        os.getenv("IRWKV_MAX_AUDIO_DECODE_BYTES", "268435456")
    ),
    # Maximum bytes a client-supplied embedding payload (`prompt_embeds`,
    # `image_embeds`, `audio_embeds`, `video_embeds`) may allocate once it is
    # densified. A sparse tensor carries its own declared shape, so a payload
    # of a few hundred bytes can expand into hundreds of GiB. The limit is
    # checked before `to_dense()` so the memory is never allocated. Set to 0
    # to disable. Default is 2 GiB, which covers a 128K-token float32
    # embedding at hidden_size 4096.
    "IRWKV_MAX_EMBED_DECODE_BYTES": lambda: int(
        os.getenv("IRWKV_MAX_EMBED_DECODE_BYTES", "2147483648")
    ),
    # Maximum number of worker threads used for STT preprocessing. The default
    # intentionally caps at 2 because that performed best in profiling.
    # https://github.com/vllm-project/vllm/pull/44612#issuecomment-4662757781
    "IRWKV_MAX_AUDIO_PREPROCESS_WORKERS": lambda: int(
        os.getenv(
            "IRWKV_MAX_AUDIO_PREPROCESS_WORKERS",
            str(max(1, min(os.cpu_count() or 1, 2))),
        )
    ),
    # Maximum decoded image size in pixels.  Small compressed images can
    # expand into gigabytes of raster memory.  This limit is enforced before
    # decoding so the memory is never allocated.  Default matches PIL's
    # built-in 2x decompression-bomb threshold (~179M pixels, ~680 MB RGB).
    "IRWKV_MAX_IMAGE_PIXELS": lambda: int(
        os.getenv("IRWKV_MAX_IMAGE_PIXELS", "178956970")
    ),
    # Backend for Video IO — selects the frame-sampling algorithm.
    # - "opencv": uniform sampling.
    # - "opencv_dynamic": duration-aware dynamic sampling.
    #
    # Custom backend implementations can be registered
    # via `@VIDEO_LOADER_REGISTRY.register("my_custom_video_loader")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "IRWKV_VIDEO_LOADER_BACKEND": lambda: os.getenv(
        "IRWKV_VIDEO_LOADER_BACKEND", "opencv"
    ),
    # Media connector implementation.
    # - "http": Default connector that supports fetching media via HTTP.
    #
    # Custom implementations can be registered
    # via `@MEDIA_CONNECTOR_REGISTRY.register("my_custom_media_connector")` and
    # imported at runtime.
    # If a non-existing backend is used, an AssertionError will be thrown.
    "IRWKV_MEDIA_CONNECTOR": lambda: os.getenv("IRWKV_MEDIA_CONNECTOR", "http"),
    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "IRWKV_XLA_CACHE_PATH": lambda: os.path.expanduser(
        os.getenv(
            "IRWKV_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "irwkv", "xla_cache"),
        )
    ),
    # If set, assert on XLA recompilation after each execution step.
    "IRWKV_XLA_CHECK_RECOMPILATION": lambda: bool(
        int(os.getenv("IRWKV_XLA_CHECK_RECOMPILATION", "0"))
    ),
    # Enable SPMD mode for TPU backend.
    "IRWKV_XLA_USE_SPMD": lambda: bool(int(os.getenv("IRWKV_XLA_USE_SPMD", "0"))),
    # KV context length each adaptive-verification profiling request pretends to
    # carry, so the profiled step reads a realistic amount of cache.
    # Raise it for long-context deployments, where step cost is dominated by
    # attention over a much larger KV cache than the default assumes.
    # Clamped to max_model_len - query_len. Default: 8192 tokens
    "IRWKV_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN": lambda: int(
        os.getenv("IRWKV_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "8192")
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "IRWKV_KEEP_ALIVE_ON_ENGINE_DEATH": lambda: bool(
        int(os.getenv("IRWKV_KEEP_ALIVE_ON_ENGINE_DEATH", "0"))
    ),
    # If the env var IRWKV_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set IRWKV_ALLOW_LONG_MAX_MODEL_LEN=1.
    "IRWKV_ALLOW_LONG_MAX_MODEL_LEN": lambda: (
        os.environ.get("IRWKV_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower()
        in ("1", "true")
    ),
    "IRWKV_TEST_FORCE_LOAD_FORMAT": lambda: os.getenv(
        "IRWKV_TEST_FORCE_LOAD_FORMAT", "dummy"
    ),
    # Queue size for fastsafetensors ParallelLoader pipelined weight
    # loading. Peak load-time VRAM is roughly
    # model_weights + (1 + queue_size) * shard_size.
    # Default 0 preserves the non-pipelined memory footprint so this
    # change does not shrink the loadable-model envelope. Set to 1
    # (or higher) to overlap producing the next shard's device buffer
    # with the consumer copying the current shard into model params,
    # at the cost of `queue_size` extra shard-sized buffers resident
    # at peak during loading.
    "IRWKV_FASTSAFETENSORS_QUEUE_SIZE": lambda: int(
        os.getenv("IRWKV_FASTSAFETENSORS_QUEUE_SIZE", "0")
    ),
    # Timeout in seconds for keeping HTTP connections alive in API server
    "IRWKV_HTTP_TIMEOUT_KEEP_ALIVE": lambda: int(
        os.environ.get("IRWKV_HTTP_TIMEOUT_KEEP_ALIVE", "5")
    ),
    # Maximum allowed value for the `n` sampling parameter (number of output
    # sequences per request). Limits resource consumption to prevent
    # denial-of-service via excessively large fan-out. Default: 16384.
    "IRWKV_MAX_N_SEQUENCES": lambda: int(
        os.environ.get("IRWKV_MAX_N_SEQUENCES", "16384")
    ),
    # Maximum number of prompts allowed in a single /v1/completions request
    # when the prompt field is a list. Prevents unbounded fan-out of engine
    # requests from a single API call. Default: 1024.
    "IRWKV_MAX_COMPLETION_PROMPTS": lambda: int(
        os.environ.get("IRWKV_MAX_COMPLETION_PROMPTS", "1024")
    ),
    # Maximum number of stop strings allowed in a single request.
    "IRWKV_MAX_STOP_STRINGS": lambda: int(
        os.environ.get("IRWKV_MAX_STOP_STRINGS", "4")
    ),
    # Maximum number of bad-word token sequences generated per request.
    "IRWKV_MAX_NUM_BAD_WORDS": lambda: int(
        os.environ.get("IRWKV_MAX_NUM_BAD_WORDS", "128")
    ),
    # Maximum total number of bad-word tokens (summed across all bad words)
    # allowed per request. Bounds the per-request GPU buffer width.
    "IRWKV_MAX_BAD_WORDS_TOTAL_TOKENS": lambda: int(
        os.environ.get("IRWKV_MAX_BAD_WORDS_TOTAL_TOKENS", "1024")
    ),
    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "IRWKV_PLUGINS": lambda: (
        None
        if "IRWKV_PLUGINS" not in os.environ
        else os.environ["IRWKV_PLUGINS"].split(",")
    ),
    # If set, monkey-patch triton.runtime.autotuner.Autotuner.run to skip
    # benchmarking and select the first valid config (walking past invalid
    # ones). Used to eliminate autotuning variability when measuring kernel
    # performance and applied before running any kernel.
    "IRWKV_TRITON_FORCE_FIRST_CONFIG": lambda: (
        os.environ.get("IRWKV_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower()
        in ("1", "true")
    ),
    # Maximum compiler threads per worker for registered CUDA Triton JIT warmup.
    # Set to 1 for serial warmup. Does not affect runtime JIT or autotuning.
    "IRWKV_TRITON_JIT_WARMUP_NUM_THREADS": lambda: int(
        os.getenv("IRWKV_TRITON_JIT_WARMUP_NUM_THREADS", "4")
    ),
    # We assume drivers can report p2p status correctly.
    # If the program hangs when using custom allreduce,
    # potantially caused by a bug in the driver (535 series),
    # if might be helpful to set IRWKV_SKIP_P2P_CHECK=0
    # so that irwkv can verify if p2p is actually working.
    # See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    "IRWKV_SKIP_P2P_CHECK": lambda: os.getenv("IRWKV_SKIP_P2P_CHECK", "1") == "1",
    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "IRWKV_DISABLED_KERNELS": lambda: (
        []
        if "IRWKV_DISABLED_KERNELS" not in os.environ
        else os.environ["IRWKV_DISABLED_KERNELS"].split(",")
    ),
    # Selects hw-agnostic layers for HF transformer backend
    "IRWKV_USE_HW_AGNOSTIC": lambda: (
        os.getenv("IRWKV_USE_HW_AGNOSTIC", "False").lower() in ("true", "1")
    ),
    # Disable pynccl (using torch.distributed instead)
    "IRWKV_DISABLE_PYNCCL": lambda: (
        os.getenv("IRWKV_DISABLE_PYNCCL", "False").lower() in ("true", "1")
    ),
    # Optional: enable external Oink custom ops (e.g., Blackwell RMSNorm).
    # Disabled by default.
    "IRWKV_USE_OINK_OPS": lambda: (
        os.getenv("IRWKV_USE_OINK_OPS", "False").lower() in ("true", "1")
    ),
    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    # Set to 1 to dequantize MXFP4 weights to BF16 once at load time and run as
    # a BF16 checkpoint (no per-step weight dequant). This improves emulation
    # latency at the cost of additional device memory. Default off.
    "IRWKV_MXFP4_EMULATION_DEQUANT_AT_LOAD": lambda: (
        os.getenv("IRWKV_MXFP4_EMULATION_DEQUANT_AT_LOAD", "False").lower()
        in ("true", "1")
    ),
    # On hardware without a native MXFP8 kernel (e.g. ROCm gfx942 / MI300), the
    # MXFP8 emulation path dequantizes weights MXFP8->BF16 once at load time and
    # runs as a BF16 checkpoint (no per-step dequant). Set to 0 to fall back to
    # per-step dequant: keeps the 1-byte MXFP8 weights (~half the weight memory)
    # at the cost of dequantizing every forward step (much slower). Default on.
    "IRWKV_MXFP8_EMULATION_DEQUANT_AT_LOAD": lambda: (
        os.getenv("IRWKV_MXFP8_EMULATION_DEQUANT_AT_LOAD", "True").lower()
        in ("true", "1")
    ),
    "IRWKV_ROCM_USE_AITER": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER", "False").lower() in ("true", "1")
    ),
    # Use AITER's CustomAllreduce as the custom-allreduce backend inside irwkv's
    # CudaCommunicator on ROCm. Also enables AITER AG/RS for DP communication.
    "IRWKV_ROCM_USE_AITER_CUSTOM_AR": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_CUSTOM_AR", "True").lower() in ("true", "1")
    ),
    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    # - use aiter tuned gemms for unquantized gemms
    "IRWKV_ROCM_USE_AITER_LINEAR": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_LINEAR", "True").lower() in ("true", "1")
    ),
    "IRWKV_ROCM_USE_AITER_LINEAR_HIPBMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_LINEAR_HIPBMM", "False").lower()
        in ("true", "1")
    ),
    # use aiter rms norm op if aiter ops are enabled.
    "IRWKV_ROCM_USE_AITER_RMSNORM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_RMSNORM", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter mha ops.
    # By default is enabled.
    "IRWKV_ROCM_USE_AITER_MHA": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_MHA", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter fp4 gemm asm.
    # By default is disabled.
    "IRWKV_ROCM_USE_AITER_FP4_ASM_GEMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_FP4_ASM_GEMM", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter rope.
    # By default is disabled.
    "IRWKV_ROCM_USE_AITER_TRITON_ROPE": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_TRITON_ROPE", "False").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp8 bmm kernel
    # By default is enabled.
    "IRWKV_ROCM_USE_AITER_FP8BMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_FP8BMM", "True").lower() in ("true", "1")
    ),
    # Whether to use aiter triton fp4 bmm kernel
    # By default is enabled.
    "IRWKV_ROCM_USE_AITER_FP4BMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_FP4BMM", "True").lower() in ("true", "1")
    ),
    # Use AITER triton unified attention for V1 attention
    "IRWKV_ROCM_USE_AITER_UNIFIED_ATTENTION": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_UNIFIED_ATTENTION", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter fusion shared experts ops.
    # By default is disabled.
    "IRWKV_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "False").lower()
        in ("true", "1")
    ),
    # Whether to use aiter triton kernels for gemm ops.
    # By default is enabled.
    "IRWKV_ROCM_USE_AITER_TRITON_GEMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_AITER_TRITON_GEMM", "True").lower() in ("true", "1")
    ),
    # use rocm skinny gemms
    "IRWKV_ROCM_USE_SKINNY_GEMM": lambda: (
        os.getenv("IRWKV_ROCM_USE_SKINNY_GEMM", "True").lower() in ("true", "1")
    ),
    # Pad the fp8 weights to 256 bytes for ROCm
    "IRWKV_ROCM_FP8_PADDING": lambda: bool(
        int(os.getenv("IRWKV_ROCM_FP8_PADDING", "1"))
    ),
    # Whether to use the shuffled kv cache layout
    "IRWKV_ROCM_SHUFFLE_KV_CACHE_LAYOUT": lambda: (
        os.getenv("IRWKV_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False").lower()
        in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Choice of quantization level: FP, INT8, INT6, INT4, INT3 or NONE
    # Recommended for large models to get allreduce
    "IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION": env_with_choices(
        "IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION",
        "NONE",
        ["FP", "INT8", "INT6", "INT4", "INT3", "NONE"],
    ),
    # Custom quick allreduce kernel for MI3* cards
    # Due to the lack of the bfloat16 asm instruction, bfloat16
    # kernels are slower than fp16,
    # If environment variable is set to 1, the input is converted to fp16
    "IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": lambda: (
        os.getenv("IRWKV_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "True").lower()
        in ("true", "1")
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the maximum allowed number of data bytes(MB) for custom quick
    # allreduce communication.
    # Default: 2048 MB.
    # Data exceeding this size will use either custom allreduce or RCCL
    # communication.
    "IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("IRWKV_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    ),
    # Custom quick allreduce kernel for MI3* cards.
    # Controls the minimum allowed number of data bytes(MB) required to use
    # custom quick allreduce communication.
    # If unset, use the built-in threshold table.
    "IRWKV_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB": lambda: maybe_convert_int(
        os.environ.get("IRWKV_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", None)
    ),
    # Controls the minimum tensor size (KB, where 1 KB = 1024 bytes) required
    # to use the configured QuickReduce codec. Smaller tensors use FP
    # QuickReduce. This does not affect QuickReduce eligibility.
    "IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB": lambda: maybe_convert_int(
        os.environ.get("IRWKV_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", None)
    ),
    # If set, enable multiprocessing in LLM for the V1 code path.
    "IRWKV_ENABLE_V1_MULTIPROCESSING": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_V1_MULTIPROCESSING", "1"))
    ),
    "IRWKV_LOG_BATCHSIZE_INTERVAL": lambda: float(
        os.getenv("IRWKV_LOG_BATCHSIZE_INTERVAL", "-1")
    ),
    "IRWKV_DISABLE_COMPILE_CACHE": disable_compile_cache,
    # If set to "0", disable LayerName opaque type for layer_name
    # parameters in custom ops.  Defaults to enabled on torch >= 2.11.
    "IRWKV_USE_LAYERNAME": lambda: bool(int(os.getenv("IRWKV_USE_LAYERNAME", "1"))),
    # If set, use the Rust frontend binary instead of the Python API server
    # process(es).
    "IRWKV_USE_RUST_FRONTEND": lambda: bool(
        int(os.getenv("IRWKV_USE_RUST_FRONTEND", "0"))
    ),
    # If set, use the packaged Rust client for `irwkv bench serve`.
    "IRWKV_USE_RUST_BENCH": lambda: bool(int(os.getenv("IRWKV_USE_RUST_BENCH", "0"))),
    # Path to the irwkv-rs binary. Defaults to "auto" which discovers the
    # binary installed with the irwkv package. Used when IRWKV_USE_RUST_FRONTEND=1
    # or IRWKV_USE_RUST_BENCH=1.
    "IRWKV_RUST_FRONTEND_PATH": lambda: _resolve_rust_cli_path(),
    # If set, irwkv will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "IRWKV_SERVER_DEV_MODE": lambda: bool(int(os.getenv("IRWKV_SERVER_DEV_MODE", "0"))),
    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "IRWKV_V1_OUTPUT_PROC_CHUNK_SIZE": lambda: int(
        os.getenv("IRWKV_V1_OUTPUT_PROC_CHUNK_SIZE", "128")
    ),
    # If set, irwkv will disable the MLA attention optimizations.
    # If set, irwkv will pick up the provided Flash Attention MLA
    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as irwkv.
    "IRWKV_RAY_PER_WORKER_GPUS": lambda: float(
        os.getenv("IRWKV_RAY_PER_WORKER_GPUS", "1.0")
    ),
    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "IRWKV_RAY_BUNDLE_INDICES": lambda: os.getenv("IRWKV_RAY_BUNDLE_INDICES", ""),
    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable IRWKV_CUDART_SO_PATH.
    "IRWKV_CUDART_SO_PATH": lambda: os.getenv("IRWKV_CUDART_SO_PATH", None),
    # Rank of the process in the data parallel setting
    "IRWKV_DP_RANK": lambda: int(os.getenv("IRWKV_DP_RANK", "0")),
    # Rank of the process in the data parallel setting.
    # Defaults to IRWKV_DP_RANK when not set.
    "IRWKV_DP_RANK_LOCAL": lambda: int(
        os.getenv("IRWKV_DP_RANK_LOCAL", sys.modules[__name__].IRWKV_DP_RANK)
    ),
    # World size of the data parallel setting
    "IRWKV_DP_SIZE": lambda: int(os.getenv("IRWKV_DP_SIZE", "1")),
    # IP address of the master node in the data parallel setting
    "IRWKV_DP_MASTER_IP": lambda: os.getenv("IRWKV_DP_MASTER_IP", "127.0.0.1"),
    # Port of the master node in the data parallel setting
    "IRWKV_DP_MASTER_PORT": lambda: int(os.getenv("IRWKV_DP_MASTER_PORT", "0")),
    # Randomize inputs during dummy runs when using Data Parallel
    "IRWKV_RANDOMIZE_DP_DUMMY_INPUTS": lambda: (
        os.environ.get("IRWKV_RANDOMIZE_DP_DUMMY_INPUTS", "0") == "1"
    ),
    # Strategy to pack the data parallel ranks for Ray.
    # Available options:
    # - "fill":
    #   for DP master node, allocate exactly data-parallel-size-local DP ranks,
    #   for non-master nodes, allocate as many DP ranks as can fit;
    # - "strict":
    #   allocate exactly data-parallel-size-local DP ranks to each picked node;
    # - "span":
    #   Should be used only when a single DP rank requires multiple nodes.
    #   allocate one DP rank over as many nodes as required for set world_size;
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "IRWKV_RAY_DP_PACK_STRATEGY": lambda: os.getenv(
        "IRWKV_RAY_DP_PACK_STRATEGY", "strict"
    ),
    # Optional comma-separated list of node IPs that Ray data-parallel
    # placement groups may use. When set, create_dp_placement_groups only
    # considers these nodes (the DP master node is always included).
    # This environment variable is ignored if data-parallel-backend is not Ray.
    "IRWKV_RAY_DP_PLACEMENT_NODE_IPS": lambda: os.getenv(
        "IRWKV_RAY_DP_PLACEMENT_NODE_IPS", ""
    ),
    # Comma-separated *additional* prefixes of env vars to copy from the
    # driver to Ray workers.  These are merged with the built-in defaults
    # defined in ``irwkv.ray.ray_env`` (IRWKV_, etc.).  Example: "MYLIB_,OTHER_"
    "IRWKV_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": lambda: os.getenv(
        "IRWKV_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", ""
    ),
    # Comma-separated *additional* individual env var names to copy from
    # the driver to Ray workers.  Merged with the built-in defaults
    # defined in ``irwkv.ray.ray_env`` (PYTHONHASHSEED).
    # Example: "MY_SECRET,MY_FLAG"
    "IRWKV_RAY_EXTRA_ENV_VARS_TO_COPY": lambda: os.getenv(
        "IRWKV_RAY_EXTRA_ENV_VARS_TO_COPY", ""
    ),
    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "IRWKV_MODEL_REDIRECT_PATH": lambda: os.environ.get(
        "IRWKV_MODEL_REDIRECT_PATH", None
    ),
    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "IRWKV_MARLIN_USE_ATOMIC_ADD": lambda: (
        os.environ.get("IRWKV_MARLIN_USE_ATOMIC_ADD", "0") == "1"
    ),
    # The activation dtype for marlin kernel
    "IRWKV_MARLIN_INPUT_DTYPE": env_with_choices(
        "IRWKV_MARLIN_INPUT_DTYPE", None, ["int8", "fp8"]
    ),
    # The online quantization dtype for humming kernel
    "IRWKV_HUMMING_ONLINE_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("IRWKV_HUMMING_ONLINE_QUANT_CONFIG", None)
    ),
    # The activation dtype config for humming kernel. Explicit schemas disable
    # fallback unless the config includes "allow_fallback": true.
    "IRWKV_HUMMING_INPUT_QUANT_CONFIG": lambda: maybe_convert_json_str_or_file(
        os.environ.get("IRWKV_HUMMING_INPUT_QUANT_CONFIG", None)
    ),
    # Whether to use fp16 accumulator mma
    "IRWKV_HUMMING_USE_F16_ACCUM": lambda: maybe_convert_bool(
        os.environ.get("IRWKV_HUMMING_USE_F16_ACCUM", "0")
    ),
    # Whether to use DeepEPLL kernels for NVFP4 quantization and dispatch method
    # only supported on Blackwell GPUs and with
    # https://github.com/deepseek-ai/DeepEP/pull/341
    "IRWKV_DEEPEPLL_NVFP4_DISPATCH": lambda: bool(
        int(os.getenv("IRWKV_DEEPEPLL_NVFP4_DISPATCH", "0"))
    ),
    # Whether to turn on the outlines cache for V1
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "IRWKV_V1_USE_OUTLINES_CACHE": lambda: (
        os.environ.get("IRWKV_V1_USE_OUTLINES_CACHE", "0") == "1"
    ),
    # Whether using Pathways
    "IRWKV_TPU_USING_PATHWAYS": lambda: bool(
        "proxy" in os.getenv("JAX_PLATFORMS", "").lower()
    ),
    # Allow use of DeepGemm kernels for fused moe ops.
    "IRWKV_USE_DEEP_GEMM": lambda: bool(int(os.getenv("IRWKV_USE_DEEP_GEMM", "1"))),
    # Whether to use E8M0 scaling when DeepGEMM is used on Blackwell GPUs.
    "IRWKV_USE_DEEP_GEMM_E8M0": lambda: bool(
        int(os.getenv("IRWKV_USE_DEEP_GEMM_E8M0", "1"))
    ),
    # Whether to create TMA-aligned scale tensor when DeepGEMM is used.
    "IRWKV_USE_DEEP_GEMM_TMA_ALIGNED_SCALES": lambda: bool(
        int(os.getenv("IRWKV_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", "1"))
    ),
    # Allow use of FlashInfer FP8 block-scale GEMM for linear layers.
    # This uses TensorRT-LLM kernels and requires SM90+ (Hopper).
    "IRWKV_BLOCKSCALE_FP8_GEMM_FLASHINFER": lambda: bool(
        int(os.getenv("IRWKV_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1"))
    ),
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "IRWKV_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("IRWKV_XGRAMMAR_CACHE_MB", "512")),
    # Maximum time in seconds allowed for regex compilation in structured
    # output backends (xgrammar, outlines). Prevents ReDoS attacks where
    # adversarial patterns cause exponential DFA state-space explosion.
    # Set to 0 to disable the timeout (not recommended in production).
    "IRWKV_REGEX_COMPILATION_TIMEOUT_S": lambda: int(
        os.getenv("IRWKV_REGEX_COMPILATION_TIMEOUT_S", "5")
    ),
    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded. The unit is bytes.
    "IRWKV_MSGPACK_ZERO_COPY_THRESHOLD": lambda: int(
        os.getenv("IRWKV_MSGPACK_ZERO_COPY_THRESHOLD", "256")
    ),
    # If set, allow insecure serialization using pickle.
    # This is useful for environments where it is deemed safe to use the
    # insecure method and it is needed for some reason.
    "IRWKV_ALLOW_INSECURE_SERIALIZATION": lambda: bool(
        int(os.getenv("IRWKV_ALLOW_INSECURE_SERIALIZATION", "0"))
    ),
    # Temporary: skip adding random suffix to internal request IDs. May be
    # needed for KV connectors that match request IDs across instances.
    "IRWKV_DISABLE_REQUEST_ID_RANDOMIZATION": lambda: bool(
        int(os.getenv("IRWKV_DISABLE_REQUEST_ID_RANDOMIZATION", "0"))
    ),
    # IP address used for NIXL handshake between remote agents.
    "IRWKV_NIXL_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "IRWKV_NIXL_SIDE_CHANNEL_HOST", "localhost"
    ),
    # Port used for NIXL handshake between remote agents.
    "IRWKV_NIXL_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("IRWKV_NIXL_SIDE_CHANNEL_PORT", "5600")
    ),
    # Address the P2P KV-offload control socket binds to. Defaults to
    # ``localhost`` (loopback only); must be set to the node IP for
    # cross-host P2P so remote peers can reach the socket.
    "IRWKV_P2P_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "IRWKV_P2P_SIDE_CHANNEL_HOST", "localhost"
    ),
    # Port the P2P KV-offload control socket binds to.
    "IRWKV_P2P_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("IRWKV_P2P_SIDE_CHANNEL_PORT", "5710")
    ),
    # IP address used for the EC connector's ZMQ side channel
    # (producer ROUTER bind, consumer DEALER dial).
    "IRWKV_EC_SIDE_CHANNEL_HOST": lambda: os.getenv(
        "IRWKV_EC_SIDE_CHANNEL_HOST", "localhost"
    ),
    # Port for the EC connector's ZMQ side channel; advertised to peers
    # via `ec_transfer_params.peer_port` on the producer's response.
    "IRWKV_EC_SIDE_CHANNEL_PORT": lambda: int(
        os.getenv("IRWKV_EC_SIDE_CHANNEL_PORT", "5601")
    ),
    # Per-request timeout (seconds) when a prefiller worker registers with
    # the Mooncake bootstrap server.
    "IRWKV_MOONCAKE_CONNECTOR_TIMEOUT": lambda: float(
        os.getenv("IRWKV_MOONCAKE_CONNECTOR_TIMEOUT", "30.0")
    ),
    # Port used for Mooncake handshake between remote agents.
    "IRWKV_MOONCAKE_BOOTSTRAP_PORT": lambda: int(
        os.getenv("IRWKV_MOONCAKE_BOOTSTRAP_PORT", "8998")
    ),
    # Log per-batch memory/disk tier breakdown on external GETs.
    "IRWKV_MOONCAKE_STORE_TIER_LOG": lambda: (
        os.getenv("IRWKV_MOONCAKE_STORE_TIER_LOG", "False").lower() in ("true", "1")
    ),
    # Number of parallel KV-load receive threads per worker rank. Lets the
    # per-request control overhead (Python prep + master key lookup) of one
    # request overlap with the RDMA transfer of another, keeping the transfer
    # engine's queue pairs busy. Helps when that overhead is significant or
    # per-request batches are too small to saturate the link on their own.
    "IRWKV_MOONCAKE_LOAD_RECV_THREADS": lambda: int(
        os.getenv("IRWKV_MOONCAKE_LOAD_RECV_THREADS", "1")
    ),
    # Fraction of the owner's DirectIO staging buffer to fill per GET batch.
    "IRWKV_MOONCAKE_DISK_STAGING_USABLE_RATIO": lambda: float(
        os.getenv("IRWKV_MOONCAKE_DISK_STAGING_USABLE_RATIO", "0.9")
    ),
    # Pin this rank to a specific owner segment ("host:port").
    "MOONCAKE_PREFERRED_SEGMENT": lambda: os.getenv("MOONCAKE_PREFERRED_SEGMENT"),
    # Override the hostname the rank registers as a Mooncake requester.
    "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": lambda: os.getenv(
        "MOONCAKE_REQUESTER_LOCAL_HOSTNAME"
    ),
    # Override the directory for the FlashInfer autotune config cache.
    "IRWKV_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.getenv(
        "IRWKV_FLASHINFER_AUTOTUNE_CACHE_DIR", None
    ),
    # Explicit RWKV preparation policy; production capture can require coverage.
    "IRWKV_RWKV_TUNING_MODE": env_with_choices(
        "IRWKV_RWKV_TUNING_MODE", "tune", ["inspect", "require", "tune"]
    ),
    "IRWKV_RWKV_TUNING_LEGACY_DIR": lambda: os.getenv("IRWKV_RWKV_TUNING_LEGACY_DIR"),
    "IRWKV_RWKV_SESSION_CACHE_SIZE": lambda: int(
        os.getenv("IRWKV_RWKV_SESSION_CACHE_SIZE", "128")
    ),
    "IRWKV_ALLOW_RUNTIME_RWKV_UPDATING": lambda: bool(
        int(os.getenv("IRWKV_ALLOW_RUNTIME_RWKV_UPDATING", "0"))
    ),
    "IRWKV_RWKV_UPLOAD_MAX_BYTES": lambda: int(
        os.getenv("IRWKV_RWKV_UPLOAD_MAX_BYTES", str(1 << 30))
    ),
    # Comma-separated FlashInfer op names to exclude from autotuning, using
    # the heuristic fallback tactic instead. Unset: skip "fp4_gemm" when the
    # CuTe-DSL NVFP4 linear kernel is selected. Empty: skip nothing.
    "IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS": lambda: (
        None
        if "IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS" not in os.environ
        else [
            v.strip()
            for v in os.environ["IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS"].split(",")
            if v.strip()
        ]
    ),
    # Flashinfer fused allreduce backend.
    "IRWKV_FLASHINFER_ALLREDUCE_BACKEND": env_with_choices(
        "IRWKV_FLASHINFER_ALLREDUCE_BACKEND",
        "auto",
        ["auto", "trtllm", "mnnvl"],
    ),
    # Control the workspace buffer size for the FlashInfer backend.
    "IRWKV_FLASHINFER_WORKSPACE_BUFFER_SIZE": lambda: int(
        os.getenv("IRWKV_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(394 * 1024 * 1024))
    ),
    # Regex timeout for use by the irwkv tool parsing plugins.
    "IRWKV_TOOL_PARSE_REGEX_TIMEOUT_SECONDS": lambda: int(
        os.getenv("IRWKV_TOOL_PARSE_REGEX_TIMEOUT_SECONDS", "1")
    ),
    # Enforce function parameter schemas in structural-tag based tool calling.
    "IRWKV_ENFORCE_STRICT_TOOL_CALLING": lambda: (
        os.getenv("IRWKV_ENFORCE_STRICT_TOOL_CALLING", "True").lower() in ("true", "1")
    ),
    # Control the max chunk bytes (in MB) for the rpc message queue.
    # Object larger than this threshold will be broadcast to worker
    # processes via zmq.
    "IRWKV_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
        os.getenv("IRWKV_MQ_MAX_CHUNK_BYTES_MB", "16")
    ),
    # Timeout in seconds for execute_model RPC calls in multiprocessing
    # executor (only applies when TP > 1).
    "IRWKV_EXECUTE_MODEL_TIMEOUT_SECONDS": lambda: int(
        os.getenv("IRWKV_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
    ),
    # Timeout in seconds for engine and worker process shutdown
    "IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS": lambda: int(
        os.getenv("IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    ),
    # KV Cache layout used throughout irwkv.
    # Some common values are:
    # - LBNHC
    # - LBHNC
    # Where N=num_states, H=num_heads and C=state_content. The default value will
    # leave the layout choice to the backend. Mind that backends may only
    # implement and support a subset of all possible layouts.
    # LHBNC hoists the head dim outside the block dim; backends must opt in via
    # AttentionBackend.supported_kv_cache_layouts().
    "IRWKV_KV_CACHE_LAYOUT": env_with_choices(
        "IRWKV_KV_CACHE_LAYOUT",
        None,
        ["LBNHC", "LBHNC", "LHBNC", "NHD", "HND", "BLHNC", "BLNHC", "BHLNC"],
    ),
    # SSM conv state layout used for Mamba models.
    # - SD: (state_len, dim) — dim contiguous (default)
    # - DS: (dim, state_len) — TP-sharded dim on dim1,
    #   consistent with SSM temporal state and LBHNC KV cache layout.
    "IRWKV_SSM_CONV_STATE_LAYOUT": env_with_choices(
        "IRWKV_SSM_CONV_STATE_LAYOUT", None, ["SD", "DS"]
    ),
    # Enable checking whether the generated logits contain NaNs,
    # indicating corrupted output. Useful for debugging low level bugs
    # or bad hardware but it may add compute overhead.
    "IRWKV_COMPUTE_NANS_IN_LOGITS": lambda: bool(
        int(os.getenv("IRWKV_COMPUTE_NANS_IN_LOGITS", "0"))
        or int(os.getenv("IRWKV_RAISE_ON_LOGIT_NANS", "0"))
    ),
    # Raise an exception when generated logits contain NaNs. Enabling this
    # also enables the NaN computation required to detect them.
    "IRWKV_RAISE_ON_LOGIT_NANS": lambda: bool(
        int(os.getenv("IRWKV_RAISE_ON_LOGIT_NANS", "0"))
    ),
    # Timeout (in seconds) for MooncakeConnector in PD disaggregated setup.
    "IRWKV_MOONCAKE_ABORT_REQUEST_TIMEOUT": lambda: int(
        os.getenv("IRWKV_MOONCAKE_ABORT_REQUEST_TIMEOUT", "480")
    ),
    # If set, it means we pre-downloaded cubin files and flashinfer will
    # read the cubin files directly.
    "IRWKV_HAS_FLASHINFER_CUBIN": lambda: bool(
        int(os.getenv("IRWKV_HAS_FLASHINFER_CUBIN", "0"))
    ),
    # Controls garbage collection during CUDA graph capture.
    # If set to 0 (default), enables GC freezing to speed up capture time.
    # If set to 1, allows GC to run during capture.
    "IRWKV_ENABLE_CUDAGRAPH_GC": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_CUDAGRAPH_GC", "0"))
    ),
    # Used to force set up loopback IP
    "IRWKV_LOOPBACK_IP": lambda: os.getenv("IRWKV_LOOPBACK_IP", ""),
    # Used to set the process name prefix for irwkv processes.
    # This is useful for debugging and monitoring purposes.
    # The default value is "IRWKV".
    "IRWKV_PROCESS_NAME_PREFIX": lambda: os.getenv(
        "IRWKV_PROCESS_NAME_PREFIX", "IRWKV"
    ),
    # Allow chunked local attention with hybrid kv cache manager.
    # Currently using the Hybrid KV cache manager with chunked local attention
    # in the Llama4 models (the only models currently using chunked local attn)
    # causes a latency regression. For this reason, we disable it by default.
    # This flag is used to allow users to enable it if they want to (to save on
    # kv-cache memory usage and enable longer contexts)
    # TODO(lucas): Remove this flag once latency regression is resolved.
    "IRWKV_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
        int(os.getenv("IRWKV_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))
    ),
    # Enables support for the "store" option in the OpenAI Responses API.
    # When set to 1, irwkv's OpenAI server will retain the input and output
    # messages for those requests in memory. By default, this is disabled (0),
    # and the "store" option is ignored.
    # NOTE/WARNING:
    # 1. Messages are kept in memory only (not persisted to disk) and will be
    #    lost when the irwkv server shuts down.
    # 2. Enabling this option will cause a memory leak, as stored messages are
    #    never removed from memory until the server terminates.
    "IRWKV_ENABLE_RESPONSES_API_STORE": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_RESPONSES_API_STORE", "0"))
    ),
    # Whether to use pytorch symmetric memory for allreduce
    "IRWKV_ALLREDUCE_USE_SYMM_MEM": lambda: bool(
        int(os.getenv("IRWKV_ALLREDUCE_USE_SYMM_MEM", "1"))
    ),
    # Whether to use FlashInfer allreduce
    "IRWKV_ALLREDUCE_USE_FLASHINFER": lambda: bool(
        int(os.getenv("IRWKV_ALLREDUCE_USE_FLASHINFER", "1"))
    ),
    # Whether to use FlashInfer's single-node PCIe CUDA-IPC all-reduce.
    # This backend has a strict single-stream contract and is opt-in while its
    # integration is being qualified.
    "IRWKV_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": lambda: bool(
        int(os.getenv("IRWKV_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"))
    ),
    # Experimental: use this to enable MCP tool calling for non harmony models
    "IRWKV_USE_EXPERIMENTAL_PARSER_CONTEXT": lambda: bool(
        int(os.getenv("IRWKV_USE_EXPERIMENTAL_PARSER_CONTEXT", "0"))
    ),
    # User override folder for tuned Triton-kernel configs. Shared by MoE,
    # Mamba SSU, and LoRA. Filenames are distinct so one folder can hold all.
    # Each component first checks this folder, then the configs shipped with
    # irwkv (if any). If no JSON matches, it uses a hard-coded heuristic.
    "IRWKV_TUNED_CONFIG_FOLDER": lambda: os.getenv("IRWKV_TUNED_CONFIG_FOLDER", None),
    # Opt-in persistence of the startup plan. When enabled, each worker
    # saves the memory-profiling result (the suggested --kv-cache-memory value
    # and the free-memory baseline) under IRWKV_CACHE_ROOT/startup_plan/,
    # keyed by a hardware+config fingerprint, and later boots auto-apply it
    # -- skipping memory profiling -- when the fingerprint matches and
    # current free memory >= the recorded baseline.
    # See irwkv/v1/worker/startup_plan.py.
    "IRWKV_ENABLE_STARTUP_PLAN": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_STARTUP_PLAN", "0"))
    ),
    # Valid values are container,code_interpreter,web_search_preview
    # ex IRWKV_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=container,code_interpreter
    # If the server_label of your mcp tool is not in this list it will
    # be completely ignored.
    "IRWKV_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": env_set_with_choices(
        "IRWKV_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        default=[],
        choices=["container", "code_interpreter", "web_search_preview"],
    ),
    # Enable automatic retry when tool call JSON parsing fails
    # If enabled, returns an error message to the model to retry
    # If disabled (default), raises an exception and fails the request
    "IRWKV_TOOL_JSON_ERROR_AUTOMATIC_RETRY": lambda: bool(
        int(os.getenv("IRWKV_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "0"))
    ),
    # Add optional custom scopes for profiling, disable to avoid overheads
    "IRWKV_CUSTOM_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("IRWKV_CUSTOM_SCOPES_FOR_PROFILING", "0"))
    ),
    # Add optional nvtx scopes for profiling, disable to avoid overheads
    "IRWKV_NVTX_SCOPES_FOR_PROFILING": lambda: bool(
        int(os.getenv("IRWKV_NVTX_SCOPES_FOR_PROFILING", "0"))
    ),
    # Represent block hashes in KV cache events as 64-bit integers instead of
    # raw bytes. Defaults to True for backward compatibility.
    "IRWKV_KV_EVENTS_USE_INT_BLOCK_HASHES": lambda: bool(
        int(os.getenv("IRWKV_KV_EVENTS_USE_INT_BLOCK_HASHES", "1"))
    ),
    # Name of the shared memory buffer used for object storage.
    # Only effective when mm_config.mm_processor_cache_type == "shm".
    # Automatically generates a unique UUID-based name per process tree
    # if not explicitly set.
    "IRWKV_OBJECT_STORAGE_SHM_BUFFER_NAME": get_env_or_set_default(
        "IRWKV_OBJECT_STORAGE_SHM_BUFFER_NAME",
        _generate_shm_name,
    ),
    # The size in MB of the buffers (NVL and RDMA) used by DeepEP
    "IRWKV_DEEPEP_BUFFER_SIZE_MB": lambda: int(
        os.getenv("IRWKV_DEEPEP_BUFFER_SIZE_MB", "1024")
    ),
    # Force DeepEP to use intranode kernel for inter-node communication in
    # high throughput mode. This is useful archive higher prefill throughput
    # on system supports multi-node nvlink (e.g GB200).
    "IRWKV_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE": lambda: bool(
        int(os.getenv("IRWKV_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE", "0"))
    ),
    # Allow DeepEP to use MNNVL (multi-node nvlink) for internode_ll kernel,
    # turn this for better latency on GB200 like system
    "IRWKV_DEEPEP_LOW_LATENCY_USE_MNNVL": lambda: bool(
        int(os.getenv("IRWKV_DEEPEP_LOW_LATENCY_USE_MNNVL", "0"))
    ),
    # DeepEP v2: use NVLink+RDMA hybrid communication across NVLink domains.
    #   * hybrid mode = multi-node RDMA
    #   * non hybrid mode = nvlink
    # Set to 1 (the default) to allow DeepEP to autoselect hybrid vs non-hybrid.
    # Set to 0 to disable hybrid mode.
    "IRWKV_DEEPEP_V2_ALLOW_HYBRID_MODE": lambda: bool(
        int(os.getenv("IRWKV_DEEPEP_V2_ALLOW_HYBRID_MODE", "1"))
    ),
    # DeepEP v2: use fewer SMs at slight throughput cost
    "IRWKV_DEEPEP_V2_PREFER_OVERLAP": lambda: bool(
        int(os.getenv("IRWKV_DEEPEP_V2_PREFER_OVERLAP", "0"))
    ),
    # DeepEP v2: trade precision for transfer size in combine
    "IRWKV_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION": lambda: bool(
        int(os.getenv("IRWKV_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION", "0"))
    ),
    # The number of SMs/CUs to allocate for communication kernels when
    # running DBO; the rest will be allocated to compute.
    # Default: 20 on CUDA (SMs), 64 on ROCm (CUs).
    "IRWKV_DBO_COMM_SMS": lambda: int(
        os.getenv(
            "IRWKV_DBO_COMM_SMS",
            "64"
            if hasattr(__import__("torch").version, "hip")
            and __import__("torch").version.hip is not None
            else "20",
        )
    ),
    # Enable max_autotune & coordinate_descent_tuning in inductor_config
    # to compile static shapes passed from compile_sizes in compilation_config
    # If set to 1, enable max_autotune; By default, this is enabled (1)
    "IRWKV_ENABLE_INDUCTOR_MAX_AUTOTUNE": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_INDUCTOR_MAX_AUTOTUNE", "1"))
    ),
    # If set to 1, enable coordinate_descent_tuning;
    # By default, this is enabled (1)
    "IRWKV_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING": lambda: bool(
        int(os.getenv("IRWKV_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING", "1"))
    ),
    # Flag to enable NCCL symmetric memory allocation and registration
    "IRWKV_USE_NCCL_SYMM_MEM": lambda: bool(
        int(os.getenv("IRWKV_USE_NCCL_SYMM_MEM", "0"))
    ),
    # NCCL header path
    "IRWKV_NCCL_INCLUDE_PATH": lambda: os.environ.get("IRWKV_NCCL_INCLUDE_PATH", None),
    # GC debug config
    # - IRWKV_GC_DEBUG=0: disable GC debugger
    # - IRWKV_GC_DEBUG=1: enable GC debugger with gc.collect elpased times
    # - IRWKV_GC_DEBUG='{"top_objects":5}': enable GC debugger with
    #                                      top 5 collected objects
    "IRWKV_GC_DEBUG": lambda: os.getenv("IRWKV_GC_DEBUG", ""),
    # Debug workspace allocations.
    # logging of workspace resize operations.
    "IRWKV_DEBUG_WORKSPACE": lambda: bool(int(os.getenv("IRWKV_DEBUG_WORKSPACE", "0"))),
    # Disables parallel execution of shared_experts via separate cuda stream
    "IRWKV_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(
        int(os.getenv("IRWKV_DISABLE_SHARED_EXPERTS_STREAM", "0"))
    ),
    # Limits when we run shared_experts in a separate stream.
    # We found out that for large batch sizes, the separate stream
    # execution is not beneficial (most likely because of the input clone)
    # TODO(alexm-redhat): Tune to be more dynamic based on GPU type
    "IRWKV_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": lambda: int(
        int(os.getenv("IRWKV_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", 256))
    ),
    # Format for saving torch.compile cache artifacts
    # - "binary": saves as binary file
    #     Safe for multiple irwkv serve processes using the same compile cache.
    # - "unpacked": saves as directory structure (for inspection/debugging)
    #     NOT multiprocess safe - race conditions may occur with multiple processes.
    #     Allows viewing and setting breakpoints in Inductor's code output files.
    "IRWKV_COMPILE_CACHE_SAVE_FORMAT": env_with_choices(
        "IRWKV_COMPILE_CACHE_SAVE_FORMAT", "binary", ["binary", "unpacked"]
    ),
    # Flag to control the v2 model runner. If unset, use config defaults.
    "IRWKV_USE_V2_MODEL_RUNNER": lambda: maybe_convert_bool(
        os.getenv("IRWKV_USE_V2_MODEL_RUNNER", None)
    ),
    # Log model inspection after loading.
    # If enabled, logs a transformers-style hierarchical view of the model
    # with quantization methods and attention backends.
    "IRWKV_LOG_MODEL_INSPECTION": lambda: bool(
        int(os.getenv("IRWKV_LOG_MODEL_INSPECTION", "0"))
    ),
    # Debug logging for --enable-mfu-metrics
    "IRWKV_DEBUG_MFU_METRICS": lambda: bool(
        int(os.getenv("IRWKV_DEBUG_MFU_METRICS", "0"))
    ),
    # Disable using pytorch's pin memory for CPU offloading.
    "IRWKV_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("IRWKV_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY", "0"))
    ),
    # Disable using UVA (Unified Virtual Addressing) for CPU offloading.
    "IRWKV_WEIGHT_OFFLOADING_DISABLE_UVA": lambda: bool(
        int(os.getenv("IRWKV_WEIGHT_OFFLOADING_DISABLE_UVA", "0"))
    ),
    # Max descriptors per CPU-KV-offload batch-memcpy call. 0 = platform default
    # (ROCm chunks at 8192, since hipMemcpyBatchAsync faults above that on
    # rocm 7.14/7.15 when batch copy is optimized; CUDA uncapped). Set >0 to override.
    "IRWKV_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS": lambda: int(
        os.getenv("IRWKV_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS", "0")
    ),
    # On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is
    # supported but disabled by default due to a small performance regression.
    # Set to 1 when pinned memory or UVA is required (e.g. CPU offloading
    # or v2 model runner).
    "IRWKV_WSL2_ENABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("IRWKV_WSL2_ENABLE_PIN_MEMORY", "0"))
    ),
    # Enable CUDA compatibility mode for datacenter GPUs with older
    # driver versions than the CUDA toolkit major version of irwkv.
    "IRWKV_ENABLE_CUDA_COMPATIBILITY": lambda: (
        os.environ.get("IRWKV_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower()
        in ("1", "true")
    ),
    # Path to the CUDA compatibility libraries when CUDA compatibility is enabled.
    "IRWKV_CUDA_COMPATIBILITY_PATH": lambda: os.environ.get(
        "IRWKV_CUDA_COMPATIBILITY_PATH", None
    ),
    # Skip model name validation in OpenAI API requests.
    # When set to 1, any model name will be accepted in the 'model' field
    # of API requests. This is useful for proxy/gateway scenarios where
    # the actual model is served but different names may be used in requests.
    "IRWKV_SKIP_MODEL_NAME_VALIDATION": lambda: (
        os.getenv("IRWKV_SKIP_MODEL_NAME_VALIDATION", "0").strip().lower()
        in ("1", "true")
    ),
    # If set to 1, enable CUDA graph memory estimation during memory profiling.
    # This profiles CUDA graph memory usage to provide more accurate KV cache
    # memory allocation. Enabled by default as of v0.21.0
    "IRWKV_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": lambda: bool(
        int(os.getenv("IRWKV_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1"))
    ),
    # Force N-contiguous weight layout for all XPU unquantized linears.
    "IRWKV_XPU_FORCE_N_CONTIG_WEIGHT": lambda: bool(
        int(os.getenv("IRWKV_XPU_FORCE_N_CONTIG_WEIGHT", "0"))
    ),
    # whether use xpu specific sample kernel
    "IRWKV_XPU_USE_SAMPLER_KERNEL": lambda: bool(
        int(os.getenv("IRWKV_XPU_USE_SAMPLER_KERNEL", "1"))
    ),
    # Enable simple KV offload.
    "IRWKV_USE_SIMPLE_KV_OFFLOAD": lambda: bool(
        int(os.getenv("IRWKV_USE_SIMPLE_KV_OFFLOAD", "0"))
    ),
    # If set to 1, use Python spinloop extension to poll in a more efficient
    # way when using the mp backend.
    # Comma-separated GPU_BDF=NIC_BDF pairs for RDMA NIC selection.
    # Must be set together with IRWKV_NIC_SELECTION_VARS.
    "IRWKV_GPU_NIC_PCIE_MAPPING": lambda: os.getenv("IRWKV_GPU_NIC_PCIE_MAPPING", ""),
    # Comma-separated list of env vars to set from the GPU-NIC mapping.
    # Each entry is VAR_NAME or VAR_NAME:<suffix> (suffix appended to
    # RDMA device name). Must be set together with IRWKV_GPU_NIC_PCIE_MAPPING.
    "IRWKV_NIC_SELECTION_VARS": lambda: os.getenv("IRWKV_NIC_SELECTION_VARS", ""),
    # If set to 1, enable the HPC fused kernels (requires the hpc package
    # (.so) and an sm100/sm103 device). Covers:
    #   * the HY V4 iHC ops -- each of the eager HYV4HCPreLayer /
    #     HYV4HCPostLayer / HYV4HCHeadLayer bodies becomes one kernel launch;
    #   * the gated-MLA output gating (attn_out * sigmoid(gate projection)),
    #     fused into gated_mla_gemm; elementwise gating only.
    # Each op additionally checks its own shape / dtype constraints and falls
    # back to the eager path when they do not hold.
    "IRWKV_ENABLE_HPC_OPS": lambda: bool(int(os.getenv("IRWKV_ENABLE_HPC_OPS", "0"))),
    # Whether to skip version suffix when building package
    "IRWKV_SKIP_VERSION_SUFFIX": lambda: bool(
        int(os.getenv("IRWKV_SKIP_VERSION_SUFFIX", "0"))
    ),
}


# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_envs_cache_enabled() -> bool:
    """Checked if __getattr__ is wrapped with functools.cache."""
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Enables caching of environment variables. This is useful for performance
    reasons, as it avoids the need to re-evaluate environment variables on
    every call.

    NOTE: Currently, it's invoked after service initialization to reduce
    runtime overhead. This also means that environment variables should NOT
    be updated after the service is initialized.
    """
    if _is_envs_cache_enabled():
        # Avoid wrapping functools.cache multiple times
        return
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Resets the environment variables cache. It could be used to isolate environments
    between unit tests.
    """
    global __getattr__
    # If __getattr__ is wrapped by functions.cache, unwrap the caching layer.
    if _is_envs_cache_enabled():
        assert hasattr(__getattr__, "__wrapped__")
        __getattr__ = __getattr__.__wrapped__


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compile_factors() -> dict[str, object]:
    """Return env vars used for torch.compile cache keys.

    Start with every known irwkv env var; drop entries in `ignored_factors`;
    hash everything else. This keeps the cache key aligned across workers."""
    ignored_factors: set[str] = {
        "MAX_JOBS",
        "IRWKV_TRITON_JIT_WARMUP_NUM_THREADS",
        "IRWKV_RPC_BASE_PATH",
        "IRWKV_USE_MODELSCOPE",
        "IRWKV_RINGBUFFER_WARNING_INTERVAL",
        "IRWKV_DEBUG_DUMP_PATH",
        "IRWKV_PORT",
        "IRWKV_CACHE_ROOT",
        # Runtime memory-plan persistence; does not affect compiled graphs.
        "IRWKV_ENABLE_STARTUP_PLAN",
        # Location-only derived paths: where a cache/config directory lives
        # cannot affect compiled artifacts, and hashing them means relocating
        # HOME or the XDG roots silently invalidates every compile cache
        # (IRWKV_CACHE_ROOT above and IRWKV_FLASHINFER_AUTOTUNE_CACHE_DIR below
        # are already ignored for the same reason).
        "IRWKV_XLA_CACHE_PATH",
        "IRWKV_CONFIG_ROOT",
        "LD_LIBRARY_PATH",
        "IRWKV_SERVER_DEV_MODE",
        "IRWKV_DP_MASTER_IP",
        "IRWKV_DP_MASTER_PORT",
        "IRWKV_NIXL_SIDE_CHANNEL_HOST",
        "IRWKV_RANDOMIZE_DP_DUMMY_INPUTS",
        "IRWKV_MODEL_REDIRECT_PATH",
        "IRWKV_HOST_IP",
        "IRWKV_FORCE_AOT_LOAD",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        # Credential; never affects compiled artifacts and must not be
        # persisted in cache_key_factors.json.
        "IRWKV_API_KEY",
        "IRWKV_LOGGING_LEVEL",
        "IRWKV_LOGGING_PREFIX",
        "IRWKV_LOGGING_STREAM",
        "IRWKV_LOGGING_CONFIG_PATH",
        "IRWKV_LOGGING_COLOR",
        "IRWKV_LOG_STATS_INTERVAL",
        "IRWKV_DEBUG_LOG_API_SERVER_RESPONSE",
        "IRWKV_TUNED_CONFIG_FOLDER",
        "IRWKV_FLASHINFER_AUTOTUNE_CACHE_DIR",
        "IRWKV_RWKV_TUNING_MODE",
        "IRWKV_RWKV_TUNING_LEGACY_DIR",
        "IRWKV_RWKV_SESSION_CACHE_SIZE",
        "IRWKV_ALLOW_RUNTIME_RWKV_UPDATING",
        "IRWKV_RWKV_UPLOAD_MAX_BYTES",
        "IRWKV_FLASHINFER_AUTOTUNE_SKIP_OPS",
        "IRWKV_HTTP_TIMEOUT_KEEP_ALIVE",
        "IRWKV_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
        "IRWKV_KEEP_ALIVE_ON_ENGINE_DEATH",
        "IRWKV_IMAGE_FETCH_TIMEOUT",
        "IRWKV_VIDEO_FETCH_TIMEOUT",
        "IRWKV_AUDIO_FETCH_TIMEOUT",
        "IRWKV_MEDIA_CACHE",
        "IRWKV_MEDIA_CACHE_MAX_SIZE_MB",
        "IRWKV_MEDIA_CACHE_TTL_HOURS",
        "IRWKV_MEDIA_FETCH_MAX_RETRIES",
        "IRWKV_MEDIA_URL_ALLOW_REDIRECTS",
        "IRWKV_MEDIA_LOADING_THREAD_COUNT",
        "IRWKV_MAX_AUDIO_CLIP_FILESIZE_MB",
        "IRWKV_MAX_AUDIO_DECODE_DURATION_S",
        "IRWKV_MAX_AUDIO_DECODE_BYTES",
        "IRWKV_MAX_AUDIO_PREPROCESS_WORKERS",
        "IRWKV_MAX_EMBED_DECODE_BYTES",
        "IRWKV_MAX_IMAGE_PIXELS",
        "IRWKV_VIDEO_LOADER_BACKEND",
        "IRWKV_MEDIA_CONNECTOR",
        "IRWKV_OBJECT_STORAGE_SHM_BUFFER_NAME",
        "IRWKV_ASSETS_CACHE",
        "IRWKV_ASSETS_CACHE_MODEL_CLEAN",
        "IRWKV_WORKER_MULTIPROC_METHOD",
        "IRWKV_ENABLE_V1_MULTIPROCESSING",
        "IRWKV_V1_OUTPUT_PROC_CHUNK_SIZE",
        "IRWKV_CPU_KVCACHE_SPACE",
        "IRWKV_ZENTORCH_WEIGHT_PREPACK",
        "IRWKV_TEST_FORCE_LOAD_FORMAT",
        "IRWKV_ENABLE_CUDA_COMPATIBILITY",
        "IRWKV_CUDA_COMPATIBILITY_PATH",
        "IRWKV_SKIP_MODEL_NAME_VALIDATION",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "NO_COLOR",
        "IRWKV_SKIP_VERSION_SUFFIX",
        "FORCE_COLOR",
    }

    from irwkv.config.utils import normalize_value

    factors: dict[str, object] = {}
    for factor, getter in environment_variables.items():
        if factor in ignored_factors:
            continue

        try:
            raw = getter()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping environment variable %s while hashing compile factors: %s",
                factor,
                exc,
            )
            continue

        factors[factor] = normalize_value(raw)

    ray_noset_env_vars = [
        # Refer to
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/nvidia_gpu.py#L11
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/amd_gpu.py#L11
        # https://github.com/ray-project/ray/blob/b97d21dab233c2bd8ed7db749a82a1e594222b5c/python/ray/_private/accelerators/amd_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/npu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/hpu.py#L12
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/neuron.py#L14
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/tpu.py#L38
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/intel_gpu.py#L10
        # https://github.com/ray-project/ray/blob/c584b1ea97b00793d1def71eaf81537d70efba42/python/ray/_private/accelerators/rbln.py#L10
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
        "RAY_EXPERIMENTAL_NOSET_RBLN_RT_VISIBLE_DEVICES",
    ]

    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))

    return factors
