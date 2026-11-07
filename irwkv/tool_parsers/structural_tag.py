# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural output descriptions shared by the RWKV parser and guidance."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ConstStringFormat(BaseModel):
    type: Literal["const_string"] = "const_string"
    value: str


class JSONSchemaFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: dict[str, Any] | bool


class AnyTextFormat(BaseModel):
    type: Literal["any_text"] = "any_text"
    excludes: list[str] = Field(default_factory=list)


class GrammarFormat(BaseModel):
    type: Literal["grammar"] = "grammar"
    grammar: str


class RegexFormat(BaseModel):
    type: Literal["regex"] = "regex"
    pattern: str


class SequenceFormat(BaseModel):
    type: Literal["sequence"] = "sequence"
    elements: list[Format]


class OrFormat(BaseModel):
    type: Literal["or"] = "or"
    elements: list[Format]


class TagFormat(BaseModel):
    type: Literal["tag"] = "tag"
    begin: str
    content: Format
    end: str | list[str]


class TriggeredTagsFormat(BaseModel):
    type: Literal["triggered_tags"] = "triggered_tags"
    triggers: list[str]
    tags: list[TagFormat]
    at_least_one: bool = False
    stop_after_first: bool = False
    excludes: list[str] = Field(default_factory=list)


class TokenTriggeredTagsFormat(BaseModel):
    type: Literal["token_triggered_tags"] = "token_triggered_tags"
    trigger_tokens: list[int | str]
    tags: list[TagFormat]
    at_least_one: bool = False
    stop_after_first: bool = False


class TagsWithSeparatorFormat(BaseModel):
    type: Literal["tags_with_separator"] = "tags_with_separator"
    tags: list[TagFormat]
    separator: str
    at_least_one: bool = False
    stop_after_first: bool = False


Format = Annotated[
    ConstStringFormat
    | JSONSchemaFormat
    | AnyTextFormat
    | GrammarFormat
    | RegexFormat
    | SequenceFormat
    | OrFormat
    | TagFormat
    | TriggeredTagsFormat
    | TokenTriggeredTagsFormat
    | TagsWithSeparatorFormat,
    Field(discriminator="type"),
]

for _model in (SequenceFormat, OrFormat, TagFormat):
    _model.model_rebuild()


class StructuralTag(BaseModel):
    type: Literal["structural_tag"] = "structural_tag"
    format: Format


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | bool | None = None
    strict: bool | None = None


class FunctionToolParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


BuiltinToolParam = dict[str, Any]


def normalize_tool_choice(tools, tool_choice):
    """Resolve the function tools exposed by a RWKV request."""
    functions = [FunctionToolParam.model_validate(t) for t in tools]
    choice = tool_choice or "auto"
    if isinstance(choice, str):
        return functions, [], choice
    if choice["type"] == "allowed_tools":
        allowed = choice["allowed_tools"]
        names = {tool.get("function", tool).get("name") for tool in allowed["tools"]}
        return [t for t in functions if t.function.name in names], [], allowed["mode"]
    name = choice.get("function", choice)["name"]
    forced = [t for t in functions if t.function.name == name]
    if not forced:
        raise ValueError(f"Unknown function tool {name!r}")
    return forced, [], "forced"


def to_guidance_grammar(spec, schema_grammar):
    """Lower structural formats to llguidance's Lark grammar representation."""
    import llguidance

    rules = []
    grammars = []

    def external(grammar):
        parts = json.loads(grammar)["grammars"]
        if len(parts) != 1:
            raise ValueError("Nested multi-grammar structural content is unsupported")
        name = f"content_{len(grammars)}"
        grammars.append(parts[0] | {"name": name})
        return f"@{name}"

    def text(excludes):
        expression = r"/(?s:.*)/"
        for value in excludes:
            if not value:
                raise ValueError("An excluded text marker cannot be empty")
            expression += f" & ~(/(?s:.*)/ {json.dumps(value)} /(?s:.*)/)"
        name = f"TEXT_{len(rules)}"
        rules.append(f"{name}: {expression}")
        return name

    def lower(value):
        if isinstance(value, ConstStringFormat):
            return json.dumps(value.value)
        if isinstance(value, JSONSchemaFormat):
            return external(schema_grammar(value.json_schema))
        if isinstance(value, GrammarFormat):
            return external(llguidance.grammar_from("grammar", value.grammar))
        if isinstance(value, RegexFormat):
            return external(llguidance.grammar_from("regex", value.pattern))
        if isinstance(value, AnyTextFormat):
            return text(value.excludes)
        if isinstance(value, (SequenceFormat, OrFormat)):
            if not value.elements:
                raise ValueError("A structural sequence or alternative cannot be empty")
            separator = " " if isinstance(value, SequenceFormat) else " | "
            return "(" + separator.join(lower(v) for v in value.elements) + ")"
        if isinstance(value, TagFormat):
            ends = [value.end] if isinstance(value.end, str) else value.end
            if not ends:
                raise ValueError("A tag must define an end")
            end = " | ".join(json.dumps(v) for v in ends)
            return f"({json.dumps(value.begin)} {lower(value.content)} ({end}))"
        if isinstance(value, TagsWithSeparatorFormat):
            if not value.tags:
                raise ValueError("A structural format must contain tags")
            tags = "(" + " | ".join(lower(t) for t in value.tags) + ")"
            calls = tags
            if not value.stop_after_first:
                calls += f" ({json.dumps(value.separator)} {tags})*"
            return f"({calls})" + ("" if value.at_least_one else "?")
        if isinstance(value, TriggeredTagsFormat):
            if (
                not value.tags
                or not value.triggers
                or any(not t for t in value.triggers)
            ):
                raise ValueError("Triggered tags require nonempty tags and triggers")
            alternatives = []
            free = text(value.excludes)
            for tag in value.tags:
                trigger = next(
                    (t for t in value.triggers if tag.begin.startswith(t)), None
                )
                if trigger is None:
                    raise ValueError(f"No trigger matches tag {tag.begin!r}")
                head = f"head_{len(rules)}"
                rules.append(f"{head}[lazy]: {free} {json.dumps(trigger)}")
                suffix = tag.model_copy(update={"begin": tag.begin[len(trigger) :]})
                alternatives.append(f"({head} {lower(suffix)})")
            calls = "(" + " | ".join(alternatives) + ")"
            repeat = (
                ("" if value.at_least_one else "?")
                if value.stop_after_first
                else ("+" if value.at_least_one else "*")
            )
            return f"({calls}{repeat} {text(value.excludes + value.triggers)})"
        raise ValueError(f"Unsupported structural format {value.type!r}")

    root = lower(StructuralTag.model_validate(spec).format)
    return json.dumps(
        {
            "grammars": [
                {"lark_grammar": f"start: {root}\n" + "\n".join(rules)},
                *grammars,
            ]
        }
    )
