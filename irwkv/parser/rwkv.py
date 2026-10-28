# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV reasoning delimiters for the parser engine."""

from functools import cache
from typing import TYPE_CHECKING

from irwkv.parser.engine.events import EventType
from irwkv.parser.engine.parser_engine import ParserEngine
from irwkv.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from irwkv.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from irwkv.entrypoints.openai.responses.protocol import ResponsesRequest
    from irwkv.tokenizers import TokenizerLike
    from irwkv.tool_parsers.abstract_tool_parser import Tool


@cache
def rwkv_config(thinking: bool = True) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="rwkv",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals={"THINK_START": "<think>", "THINK_END": "</think>"},
        token_id_terminals={"THINK_START": "<think>", "THINK_END": "</think>"},
        transitions={
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING, ()
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT, (EventType.REASONING_END,)
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT, ()),
        },
    )


class RwkvParser(ParserEngine):
    def __init__(
        self, tokenizer: "TokenizerLike", tools: "list[Tool] | None" = None, **kwargs
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault("parser_engine_config", rwkv_config(self.thinking_enabled))
        super().__init__(tokenizer, tools, **kwargs)

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
