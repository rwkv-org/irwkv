# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

__all__ = ["ToolParser", "ToolParserManager"]


_TOOL_PARSERS_TO_REGISTER = {
    "rwkv": ("rwkv_tool_parser", "RwkvToolParser"),
}


def register_lazy_tool_parsers():
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = f"irwkv.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
