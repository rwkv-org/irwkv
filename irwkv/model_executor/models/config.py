# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import TYPE_CHECKING

from irwkv.config.model import PROCESSED_LOGPROBS_MODES
from irwkv.logger import init_logger

if TYPE_CHECKING:
    from irwkv.config import IrwkvConfig, ModelConfig

logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(irwkv_config: "IrwkvConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, irwkv_config: "IrwkvConfig") -> None:
        """Perform early validation and setup for hybrid attention/mamba models.

        Block size alignment with mamba page sizes is handled later by
        Platform.update_block_size_for_backend(), which runs after model
        layers are constructed and the attention backend is known.

        Args:
            irwkv_config: irwkv Config

        """
        # Enable FULL_AND_PIECEWISE by default
        MambaModelConfig.verify_and_update_config(irwkv_config)


class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, irwkv_config: "IrwkvConfig") -> None:
        """Enable FULL_AND_PIECEWISE cuda graph mode by default (required
        to get good performance for mamba layers in V1).

        Args:
            irwkv_config: irwkv Config

        """
        model_config = irwkv_config.model_config
        cache_config = irwkv_config.cache_config

        if cache_config.enable_prefix_caching:
            if cache_config.mamba_cache_mode == "none":
                cache_config.mamba_cache_mode = "align"
                logger.info(
                    "Mamba cache mode is set to '%s' for %s by default "
                    "when prefix caching is enabled",
                    cache_config.mamba_cache_mode,
                    model_config.architecture,
                )
            assert irwkv_config.scheduler_config.enable_chunked_prefill, (
                "Chunked prefill is required for mamba cache mode 'align'."
            )
            # By default, mamba block size will be set to max_model_len (see
            # below). When enabling prefix caching, we align mamba block size
            # to the block size as the basic granularity for prefix caching.
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = cache_config.block_size
        else:
            if cache_config.mamba_cache_mode != "none":
                cache_config.mamba_cache_mode = "none"
                logger.warning(
                    "Mamba cache mode is set to 'none' when prefix caching is disabled"
                )
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = model_config.max_model_len


class RwkvForCausalLMConfig(VerifyAndUpdateConfig):
    LORA_KEYS = frozenset(("r", "lora_alpha", "use_nora", "target_modules"))
    LORA_TARGET_MODULES = frozenset(
        ("r_proj", "k_proj", "v_proj", "o_proj", "mlp.key", "mlp.value", "lm_head")
    )

    @staticmethod
    def verify_lora_config(lora: object) -> None:
        keys = RwkvForCausalLMConfig.LORA_KEYS
        targets = RwkvForCausalLMConfig.LORA_TARGET_MODULES
        if not isinstance(lora, dict) or lora.keys() != keys:
            raise ValueError(f"rwkv_lora requires exactly the keys {sorted(keys)}")
        rank = lora["r"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rwkv_lora.r must be a positive integer")
        alpha = lora["lora_alpha"]
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
        ):
            raise ValueError("rwkv_lora.lora_alpha must be a finite number")
        if not isinstance(lora["use_nora"], bool):
            raise ValueError("rwkv_lora.use_nora must be a bool")
        modules = lora["target_modules"]
        if (
            not isinstance(modules, (list, tuple))
            or not modules
            or not all(isinstance(module, str) for module in modules)
            or not set(modules) <= targets
        ):
            raise ValueError(
                "rwkv_lora.target_modules must be a non-empty list drawn from "
                f"{sorted(targets)}"
            )

    @staticmethod
    def verify_and_update_config(irwkv_config: "IrwkvConfig") -> None:
        import torch

        from irwkv.config.compilation import CompilationMode, CUDAGraphMode

        model_config = irwkv_config.model_config
        parallel_config = irwkv_config.parallel_config
        cache_config = irwkv_config.cache_config

        if model_config.dtype != torch.bfloat16:
            raise ValueError("RWKV inference requires --dtype bfloat16")
        if parallel_config.tensor_parallel_size != 1:
            raise ValueError("RWKV does not support tensor parallelism")
        if parallel_config.pipeline_parallel_size not in (1, 2):
            raise ValueError("RWKV supports pipeline parallel size 1 or 2")
        if parallel_config.decode_context_parallel_size != 1:
            raise ValueError("RWKV does not support decode context parallelism")
        if parallel_config.prefill_context_parallel_size != 1:
            raise ValueError("RWKV does not support prefill context parallelism")
        if model_config.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            raise ValueError("RWKV Rapid-Sampling supports only raw logprobs")
        if model_config.return_sampling_mask:
            raise ValueError("RWKV Rapid-Sampling does not return sampling masks")
        if model_config.enable_trace_replay:
            raise ValueError("RWKV Rapid-Sampling does not support trace replay")
        lora = getattr(model_config.hf_config, "rwkv_lora", None)
        if lora is not None:
            RwkvForCausalLMConfig.verify_lora_config(lora)
        if cache_config.kv_offloading_size is not None:
            raise ValueError("RWKV does not support KV cache offloading")
        if (
            irwkv_config.kv_transfer_config is not None
            and irwkv_config.kv_transfer_config.is_kv_transfer_instance
        ):
            raise ValueError("RWKV does not support KV cache connectors")
        if irwkv_config.mamba_config.enable_stochastic_rounding:
            raise ValueError("RWKV does not support stochastic cache rounding")
        if not irwkv_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("RWKV requires chunked prefill")
        if cache_config.mamba_cache_mode == "all":
            raise ValueError("RWKV does not support Mamba cache mode 'all'")

        head_dtype = getattr(model_config.hf_config, "head_dtype", None)
        if head_dtype not in (None, "float32", torch.float32):
            raise ValueError("RWKV Rapid-Sampling requires a float32 lm_head output")
        model_config.hf_config.head_dtype = "float32"

        if cache_config.mamba_cache_dtype == "auto":
            cache_config.mamba_cache_dtype = "bfloat16"
        elif cache_config.mamba_cache_dtype != "bfloat16":
            raise ValueError("RWKV shift state requires bfloat16 cache dtype")

        if cache_config.mamba_ssm_cache_dtype == "auto":
            cache_config.mamba_ssm_cache_dtype = "float32"
        elif cache_config.mamba_ssm_cache_dtype not in ("float16", "float32"):
            raise ValueError("RWKV WKV state supports float16 or float32")

        MambaModelConfig.verify_and_update_config(irwkv_config)

        compilation_config = irwkv_config.compilation_config
        if compilation_config.mode is None:
            compilation_config.mode = CompilationMode.NONE
        if compilation_config.cudagraph_mode is None:
            compilation_config.cudagraph_mode = (
                CUDAGraphMode.NONE if model_config.enforce_eager else CUDAGraphMode.FULL
            )
        if not model_config.enforce_eager:
            max_tokens = irwkv_config.scheduler_config.max_num_batched_tokens
            if compilation_config.cudagraph_capture_sizes is None:
                if compilation_config.max_cudagraph_capture_size is None:
                    compilation_config.max_cudagraph_capture_size = max_tokens
                max_capture = min(
                    compilation_config.max_cudagraph_capture_size,
                    max_tokens,
                )
                capture_sizes = [1, 2, 4]
                capture_sizes.extend(range(8, min(max_capture + 1, 256), 8))
                if max_capture >= 256:
                    capture_sizes.extend(range(256, min(max_capture + 1, 1024), 16))
                if max_capture >= 1024:
                    capture_sizes.extend(range(1024, max_capture + 1, 1024))
                if max_capture not in capture_sizes:
                    capture_sizes.append(max_capture)
                compilation_config.cudagraph_capture_sizes = capture_sizes


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "RwkvForCausalLM": RwkvForCausalLMConfig,
}
