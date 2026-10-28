# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import irwkv.envs as envs
from irwkv.config import IrwkvConfig, ModelConfig
from irwkv.config.load import LoadConfig
from irwkv.logger import init_logger
from irwkv.model_executor.model_loader.reload import finalize_layerwise_processing
from irwkv.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from irwkv.platforms import current_platform
from irwkv.tracing import instrument
from irwkv.utils.mem_utils import format_gib
from irwkv.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def get_external_weight_memory(self, irwkv_config: IrwkvConfig) -> int:
        """Get weights memory from external process;
        0 when the weights are not external."""
        return 0

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def create_model(
        self, irwkv_config: IrwkvConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Create a model with the given configurations."""
        model = initialize_model(
            irwkv_config=irwkv_config,
            model_config=model_config,
            prefix=prefix,
        )
        log_model_inspection(model)
        return model

    @instrument(span_name="Load model")
    def load_model(
        self, irwkv_config: IrwkvConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = irwkv_config.device_config
        load_config = irwkv_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                time_before_load = time.perf_counter()
                model = self.create_model(
                    irwkv_config=irwkv_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                time_after_load = time.perf_counter()
                logger.info_once(
                    "Initializing model took %.6f seconds",
                    time_after_load - time_before_load,
                )

            logger.debug("Loading weights on %s ...", load_device)
            self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if IRWKV_LOG_MODEL_INSPECTION=1."""
    if not envs.IRWKV_LOG_MODEL_INSPECTION:
        return

    from irwkv.model_inspection import format_model_inspection

    logger.info("irwkv model structure:\n%s", format_model_inspection(model))


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False
