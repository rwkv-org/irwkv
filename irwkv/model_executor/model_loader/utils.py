# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn

import irwkv.envs as envs
from irwkv.config import (
    IrwkvConfig,
    ModelConfig,
    set_current_irwkv_config,
)
from irwkv.logger import init_logger
from irwkv.model_executor.layers.attention import is_deferred_attention_layer
from irwkv.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from irwkv.model_executor.model_loader.reload import (
    record_metadata_for_reloading,
)
from irwkv.model_executor.model_loader.weight_tying import maybe_retie_word_embeddings
from irwkv.model_executor.utils import is_weights_pre_processed
from irwkv.tracing import instrument
from irwkv.utils.mem_utils import release_device_memory_under_pressure
from irwkv.utils.platform_utils import is_pin_memory_available
from irwkv.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


@instrument(span_name="Initialize model")
def initialize_model(
    irwkv_config: IrwkvConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    if model_config is None:
        model_config = irwkv_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    with set_current_irwkv_config(irwkv_config, check_compile=True, prefix=prefix):
        model = model_class(irwkv_config=irwkv_config, prefix=prefix)
        record_metadata_for_reloading(model)
    return model


def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    """Post-process loaded weights into runtime format.

    Under ``weights_already_processed`` (weight cache IPC loader), quant
    methods skip tensor transforms and must declare
    ``supports_pre_processed_weights``, otherwise this raises ``RuntimeError``.
    """
    # Reclaim memory when an explicit lm_head has been
    # loaded, but it is identical to the input embeddings.
    maybe_retie_word_embeddings(model, model_config)

    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            if (
                is_weights_pre_processed()
                and not quant_method.supports_pre_processed_weights
            ):
                raise RuntimeError(
                    f"layer {name or '<root>'}: {type(quant_method).__name__} "
                    "does not support pre-processed weights"
                )
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they typically expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            # Methods that can process weights in place (e.g. PLE scale
            # validation) set requires_device_loading=False to skip this move.
            loading_context = (
                device_loading_context(module, target_device)
                if quant_method.requires_device_loading
                else nullcontext()
            )
            with loading_context:
                quant_method.process_weights_after_loading(module)
            # process_weights_after_loading may swap in freshly-created
            # Parameters (e.g. FP8 requantization), which are stamped with the
            # global rank in BaseirwkvParameter.__init__. Re-reconcile their TP
            # state to the layer so a later weight reload / RL weight-refit
            # narrows replicated (disable_tp) weights at the correct offset.
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
            # Repacking transients above can leave large amounts of memory in
            # the caching allocator, which starves the OS on UMA devices.
            release_device_memory_under_pressure(target_device)

    # Initialize post-load attention weights for any attention layer and MM
    # encoder. NOTE: Happens after other modules so we can easily decompress
    # weights.
    for _, module in model.named_modules():
        if is_deferred_attention_layer(module):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # Model-level post-load hook, after the per-layer quant finalize.
    if hasattr(model, "process_weights_after_loading"):
        model.process_weights_after_loading()


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    cpu_params: set[str] = set()
    uva_offloaded_parameters: list[str] = []

    # Store which parameters are on CPU and move them to the GPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            cpu_params.add(name)
            p.data = p.data.to(target_device)
        if getattr(p, "_irwkv_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        use_pin_memory = (
            is_pin_memory_available()
            and not envs.IRWKV_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        # Restore the CPU-resident parameters, ignoring new parameters.
        for name, p in module.named_parameters():
            if name in cpu_params:
                p.data = torch.empty_like(
                    p.data, device="cpu", pin_memory=use_pin_memory
                ).copy_(p.data)

            # parameter is UVA offloaded, but was replaced with a new device tensor
            # re-offload it to CPU using UVA
            if name in uva_offloaded_parameters and not getattr(
                p, "_irwkv_is_uva_offloaded", False
            ):
                cpu_data = torch.empty_like(
                    p.data, device="cpu", pin_memory=use_pin_memory
                ).copy_(p.data)
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._irwkv_is_uva_offloaded = True


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    return model_config.registry.resolve_model_cls(
        model_config.architectures,
        model_config=model_config,
    )


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    model_cls_and_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_cls_and_arch
    return model_cls_and_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
