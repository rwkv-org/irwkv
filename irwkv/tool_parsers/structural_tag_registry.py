# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.tool import Tool as ResponsesTool
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from openai.types.responses.tool_choice_function import ToolChoiceFunction
from pydantic import BaseModel
from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
from xgrammar.structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)

from irwkv.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from irwkv.tool_parsers.tool_strict_level import ToolStrictLevel
from irwkv.tool_parsers.utils import get_function_tools, require_function_tools

ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ResponsesToolChoice
    | None
)
AllowedToolRef: TypeAlias = dict[str, object]
SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
StructuralTagBuilder: TypeAlias = Callable[
    [
        list[FunctionToolParam],
        list[BuiltinToolParam],
        SimplifiedToolChoice,
        bool,
        str,
    ],
    StructuralTag,
]

# Keep this list in sync with xgrammar.builtin_structural_tag. It is used for
# irwkv-side validation and for documenting the xgrammar builtin surface that
# can be requested by tool parsers through ``structural_tag_model``.
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset(
    {
        "llama",
        "kimi",
        "deepseek_r1",
        "deepseek_v3_1",
        "qwen_3_5",
        "qwen_3_coder",
        "mimo",
        "qwen_3",
        "deepseek_v3_2",
        "deepseek_v4",
        "deepseek_v4_1",
    }
)
IRWKV_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"rwkv"})
SUPPORTED_STRUCTURAL_TAG_MODELS = (
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | IRWKV_BUILTIN_STRUCTURAL_TAG_MODELS
)

_IRWKV_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}
_BUILTIN_TOOL_STRUCTURAL_TAG_MODELS: set[str] = set()


def register_irwkv_structural_tag(
    model: str,
    *,
    builtin_tools: bool = False,
) -> Callable[[StructuralTagBuilder], StructuralTagBuilder]:
    """Register a irwkv-owned structural tag builder.

    Args:
        model: Structural tag model key.
        builtin_tools: Whether the model can call non-function tools, so its
            builder receives every request tool instead of only the function
            tools from ``get_function_tools``.

    """

    def decorator(func: StructuralTagBuilder) -> StructuralTagBuilder:
        _IRWKV_STRUCTURAL_TAG_REGISTRY[model] = func
        if builtin_tools:
            _BUILTIN_TOOL_STRUCTURAL_TAG_MODELS.add(model)
        return func

    return decorator


def get_structural_tag_tools(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
    tool_choice: ToolChoice,
) -> Sequence[ChatCompletionToolsParam | ResponsesTool]:
    """Return the tools a structural tag for ``model`` lets the model call.

    Most models are only shown function tools, so their grammar must cover
    exactly those: a tool missing from the prompt cannot be called, and one
    missing from the grammar is shown but blocked.

    Raises:
        IRWKVValidationError: ``tool_choice`` requires a call, but the model
            has no tool it can call.

    """
    if model in _BUILTIN_TOOL_STRUCTURAL_TAG_MODELS:
        return tools
    if (
        tool_choice is None
        or tool_choice == "auto"
        or getattr(tool_choice, "mode", None) == "auto"
    ):
        return get_function_tools(tools)
    return require_function_tools(tools)


def _tool_is_strict(tool: ChatCompletionToolsParam | ResponsesTool) -> bool:
    if isinstance(tool, FunctionTool):
        return tool.strict is True
    if isinstance(tool, ChatCompletionToolsParam):
        return tool.function.strict is True
    return False


def _any_tool_strict(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
) -> bool:
    return any(_tool_is_strict(tool) for tool in tools)


def _with_tool_strict(
    tool: ChatCompletionToolsParam | ResponsesTool,
    strict: bool,
) -> ChatCompletionToolsParam | ResponsesTool:
    """Return a copy of ``tool`` with ``strict`` set, leaving the request's alone."""
    if isinstance(tool, FunctionTool):
        return tool.model_copy(update={"strict": strict})
    if isinstance(tool, ChatCompletionToolsParam):
        return tool.model_copy(
            update={"function": tool.function.model_copy(update={"strict": strict})}
        )
    return tool


def resolve_tool_strictness(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
    tool_choice: ToolChoice,
    strict_level: ToolStrictLevel,
) -> Sequence[ChatCompletionToolsParam | ResponsesTool] | None:
    """Decide whether a structural tag applies and pin each tool's ``strict``.

    A tool without an explicit ``strict`` is treated as non-strict: its call
    envelope is still constrained, but its arguments stay free unless the
    server level is PARAMETER. ``None`` means no structural tag.
    """
    if (
        tool_choice == "auto"
        and strict_level == ToolStrictLevel.AUTO
        and not _any_tool_strict(tools)
    ):
        return None
    return [
        _with_tool_strict(
            tool, strict_level >= ToolStrictLevel.PARAMETER or _tool_is_strict(tool)
        )
        for tool in tools
    ]


def get_model_structural_tag(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
    token_suffix: str = "",
    strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""
    if not tools or tool_choice == "none":
        return None

    tools = get_structural_tag_tools(model, tools, tool_choice)
    if not tools:
        return None
    tools = resolve_tool_strictness(tools, tool_choice, strict_level)
    if tools is None:
        return None

    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

    if model in _IRWKV_STRUCTURAL_TAG_REGISTRY:
        function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
            dumped_tools,
            dumped_tool_choice,
        )
        return _IRWKV_STRUCTURAL_TAG_REGISTRY[model](
            function_tools,
            builtin_tools,
            simplified_tool_choice,
            reasoning,
            token_suffix,
        )

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        supported = sorted(SUPPORTED_STRUCTURAL_TAG_MODELS)
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    if token_suffix:
        raise ValueError(
            f"Structural tag model {model!r} is an xgrammar builtin with fixed "
            f"tokens and cannot apply token_suffix={token_suffix!r}"
        )

    return get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
    )


_TOOL_CALL_LIST_FORMATS = (
    TriggeredTagsFormat,
    TokenTriggeredTagsFormat,
    TagsWithSeparatorFormat,
)


def limit_to_single_tool_call(tag: StructuralTag) -> StructuralTag:
    """Stop every tool-call list in ``tag`` after its first call, in place.

    Enforces ``parallel_tool_calls=false`` in the grammar. In every registered
    tag, ``_TOOL_CALL_LIST_FORMATS`` repeat tool calls, never arguments.
    """

    def visit(node: Any) -> None:
        if isinstance(node, BaseModel):
            if isinstance(node, _TOOL_CALL_LIST_FORMATS):
                node.stop_after_first = True
            for name in type(node).model_fields:
                visit(getattr(node, name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(tag.format)
    return tag


def _dump_tool_for_xgrammar(
    tool: ChatCompletionToolsParam | ResponsesTool,
) -> dict[str, Any]:
    """Convert tool objects to xgrammar's Chat Completions tool protocol."""
    if isinstance(tool, FunctionTool):
        function: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            function["description"] = tool.description
        if tool.parameters is not None:
            function["parameters"] = tool.parameters
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}
    dumped_tool = tool.model_dump(mode="json", exclude_none=True)
    if isinstance(tool, ChatCompletionToolsParam):
        return dumped_tool
    return dict(dumped_tool)


def _dump_tool_choice_for_xgrammar(
    tool_choice: ToolChoice,
) -> dict[str, Any] | str | None:
    """Convert tool_choice objects to xgrammar's expected protocol."""
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)

    if isinstance(tool_choice, ToolChoiceFunction):
        return {
            "type": "function",
            "function": {"name": tool_choice.name},
        }

    if isinstance(tool_choice, ToolChoiceAllowed):
        return {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": tool_choice.mode,
                "tools": [
                    _dump_allowed_tool_ref_for_xgrammar(tool)
                    for tool in tool_choice.tools
                ],
            },
        }

    return tool_choice.model_dump(mode="json", exclude_none=True)


def _dump_allowed_tool_ref_for_xgrammar(tool_ref: AllowedToolRef) -> AllowedToolRef:
    if (
        tool_ref.get("type") == "function"
        and "function" not in tool_ref
        and "name" in tool_ref
    ):
        return {
            "type": "function",
            "function": {"name": tool_ref["name"]},
        }
    return tool_ref


def get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


_RWKV_TOOL_CALL_TRIGGER = "**Tool Call:**"


def _rwkv_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=(
                f'{_RWKV_TOOL_CALL_TRIGGER}\n```json\n{{"name": "'
                f'{tool.function.name}", "arguments": '
            ),
            content=JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function)
            ),
            end="}\n```",
        )
        for tool in tools
    ]


@register_irwkv_structural_tag("rwkv")
def get_rwkv_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    del builtin_tools, reasoning, token_suffix

    tags = _rwkv_tool_tags(tools)
    if tool_choice == "auto":
        output = (
            TriggeredTagsFormat(
                triggers=[_RWKV_TOOL_CALL_TRIGGER],
                tags=tags,
            )
            if tags
            else AnyTextFormat()
        )
    else:
        output = SequenceFormat(
            elements=[
                AnyTextFormat(excludes=[_RWKV_TOOL_CALL_TRIGGER]),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="\n",
                    at_least_one=True,
                    stop_after_first=tool_choice == "forced",
                ),
            ]
        )
    return StructuralTag(format=output)
