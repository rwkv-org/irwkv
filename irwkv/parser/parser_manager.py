# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from irwkv.logger import init_logger
from irwkv.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from irwkv.parser.abstract_parser import Parser
    from irwkv.reasoning import ReasoningParser
    from irwkv.tokenizers import TokenizerLike
    from irwkv.tool_parsers import ToolParser

logger = init_logger(__name__)

HF_PARSER = "hf"


class ParserManager:
    """Provides a unified Parser by composing reasoning and tool parser adapters."""

    @classmethod
    def get_tool_parser(
        cls,
        tool_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
    ) -> type[ToolParser] | None:
        """Get the tool parser based on the name."""
        from irwkv.tool_parsers import ToolParserManager

        parser: type[ToolParser] | None = None
        if not enable_auto_tools or tool_parser_name is None:
            return parser
        logger.info_once('"auto" tool choice has been enabled.')

        try:
            parser = ToolParserManager.get_tool_parser(tool_parser_name)
        except Exception as e:
            raise TypeError(
                "Error: --enable-auto-tool-choice requires "
                f"tool_parser:'{tool_parser_name}' which has not "
                "been registered"
            ) from e
        return parser

    @classmethod
    def get_reasoning_parser(
        cls,
        reasoning_parser_name: str | None,
    ) -> type[ReasoningParser] | None:
        """Get the reasoning parser based on the name."""
        from irwkv.reasoning import ReasoningParserManager

        parser: type[ReasoningParser] | None = None
        if not reasoning_parser_name:
            return None
        try:
            parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            assert parser is not None
        except Exception as e:
            raise TypeError(f"{reasoning_parser_name=} has not been registered") from e
        return parser

    @classmethod
    def validate_parser_tokenizer(
        cls,
        parser_cls: type[Parser],
        tokenizer: TokenizerLike,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Fail at startup if the tokenizer lacks what the parsers need."""
        try:
            parser_cls(tokenizer)
        except Exception as e:
            flags = []
            if parser_cls.tool_parser_cls is not None:
                flags.append(f"--tool-call-parser {tool_parser_name}")
            if parser_cls.reasoning_parser_cls is not None:
                flags.append(f"--reasoning-parser {reasoning_parser_name}")
            raise TypeError(
                f"{' and '.join(flags)} cannot be used with the tokenizer of "
                f"{model_name!r}: {e}"
            ) from e

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        tool_strict_level: str = "auto",
        tokenizer: TokenizerLike | None = None,
    ) -> type[Parser] | None:
        """Get a Parser that handles both reasoning and tool parsing.

        Composes the individual parsers into a ``DelegatingParser`` subclass.

        Args:
            tool_parser_name: The name of the tool parser.
            reasoning_parser_name: The name of the reasoning parser.
            enable_auto_tools: Whether auto tool choice is enabled.
            model_name: The model name for parser-specific warnings.
            tool_strict_level: Server-side floor for tool-call structural
                tags (``--tool-strict-level``).
            tokenizer: Tokenizer the composed parser is validated against
                at startup (see :meth:`validate_parser_tokenizer`). ``None``
                skips the check (e.g. ``skip_tokenizer_init``).

        Returns:
            A Parser class, or None if neither parser is specified.

        """
        parser_cls = cls._compose_parser(
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            tool_strict_level=tool_strict_level,
            tokenizer=tokenizer,
        )
        if parser_cls is not None and tokenizer is not None:
            cls.validate_parser_tokenizer(
                parser_cls,
                tokenizer,
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                model_name=model_name,
            )
        return parser_cls

    @classmethod
    def _compose_parser(
        cls,
        tool_parser_name: str | None,
        reasoning_parser_name: str | None,
        enable_auto_tools: bool,
        model_name: str | None,
        tool_strict_level: str,
        tokenizer: TokenizerLike | None,
    ) -> type[Parser] | None:
        if not tool_parser_name and not reasoning_parser_name:
            return None

        reasoning_parser_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_parser_cls = cls.get_tool_parser(
            tool_parser_name, enable_auto_tools, model_name
        )

        if reasoning_parser_cls is None and tool_parser_cls is None:
            return None

        strict_level = ToolStrictLevel.from_name(tool_strict_level)

        if HF_PARSER in (reasoning_parser_name, tool_parser_name):
            if {reasoning_parser_name, tool_parser_name} - {
                HF_PARSER,
                None,
                "",
            }:
                raise TypeError(
                    "The hf parser cannot be combined with other "
                    "reasoning or tool call parsers"
                )
            from irwkv.parser.response_template import (
                ResponseTemplateParser,
                validate_tokenizer_response_template,
            )

            if tokenizer is not None:
                validate_tokenizer_response_template(
                    tokenizer,
                    reasoning=reasoning_parser_cls is not None,
                    tools=tool_parser_cls is not None,
                )

            r_cls = reasoning_parser_cls
            t_cls = tool_parser_cls
            auto_tools = enable_auto_tools

            class _ResponseTemplateParser(ResponseTemplateParser):
                reasoning_parser_cls = r_cls
                tool_parser_cls = t_cls
                tool_strict_level = strict_level
                _enable_auto_tools = auto_tools

            return _ResponseTemplateParser

        from irwkv.parser.abstract_parser import DelegatingParser

        r_cls = reasoning_parser_cls
        t_cls = tool_parser_cls

        class _Parser(DelegatingParser):
            reasoning_parser_cls = r_cls
            tool_parser_cls = t_cls
            tool_strict_level = strict_level

            def set_prompt_token_ids(self, prompt_token_ids):
                if self._reasoning_parser is not None:
                    self._reasoning_parser.adjust_initial_state_from_prompt(
                        prompt_token_ids
                    )

        return _Parser
