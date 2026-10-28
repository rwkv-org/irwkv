# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
]
"""
Register a lazy module mapping.

Example:
    ReasoningParserManager.register_lazy_module(
        name="rwkv",
        module_path="irwkv.reasoning.rwkv_engine_reasoning_parser",
        class_name="RwkvReasoningParser",
    )
"""


_REASONING_PARSERS_TO_REGISTER = {
    "rwkv": ("rwkv_engine_reasoning_parser", "RwkvReasoningParser"),
    "hf": ("response_template_reasoning_parser", "ResponseTemplateReasoningParser"),
}


def register_lazy_reasoning_parsers():
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = f"irwkv.reasoning.{file_name}"
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_reasoning_parsers()
