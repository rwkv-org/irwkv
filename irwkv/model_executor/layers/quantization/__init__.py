# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.logger import init_logger
from irwkv.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from irwkv.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = str
QUANTIZATION_METHODS: list[str] = []
DEPRECATED_QUANTIZATION_METHODS: list[str] = []

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized irwkv quantization config.

    When a quantization method is not supported by irwkv, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from irwkv.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from irwkv.model_executor.layers.quantization import get_quantization_config
        >>> from irwkv.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>

    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.debug(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
