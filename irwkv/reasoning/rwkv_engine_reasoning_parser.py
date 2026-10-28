# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import cast

from irwkv.parser.engine.adapters import ParserEngineReasoningAdapter
from irwkv.parser.rwkv import RwkvParser


class RwkvReasoningParser(ParserEngineReasoningAdapter):
    _parser_engine_cls = RwkvParser

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._prompt_fragment = ""
        self._prompt_reasoning_ended = False
        self._prompt_prepared = False

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        if self._prompt_prepared:
            return
        self._prompt_prepared = True
        text = self.model_tokenizer.decode(prompt_token_ids)
        self._prompt_reasoning_ended = self.is_reasoning_end(prompt_token_ids)
        for marker in ("<think>", "</think>"):
            for length in range(len(marker) - 1, 0, -1):
                if text.endswith(marker[:length]) and length > len(
                    self._prompt_fragment
                ):
                    self._prompt_fragment = marker[:length]
                    break
        if self._prompt_fragment:
            self._parser_engine.initialize_streaming()
            self._parser_engine._feed(self._prompt_fragment, [])

    def extract_reasoning(self, model_output, request):
        if self._prompt_reasoning_ended:
            return None, model_output
        return super().extract_reasoning(self._prompt_fragment + model_output, request)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if self._prompt_reasoning_ended:
            return 0
        if self._streaming_count_valid or not self._prompt_fragment:
            return super().count_reasoning_tokens(token_ids)
        if self._counting_parser_engine is None:
            self._counting_parser_engine = self._parser_engine_cls(
                self.model_tokenizer, **self._parser_engine_kwargs
            )
        engine = self._counting_parser_engine
        engine._reset()
        engine._feed(self._prompt_fragment, [])
        engine._feed(self.model_tokenizer.decode(token_ids), token_ids)
        engine._engine.finish()
        return engine.count_reasoning_tokens(token_ids)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(input_ids)
        start = text.rfind("<think>")
        end = text.rfind("</think>")
        for length in range(len("<think>") - 1, 1, -1):
            if text.endswith("<think>"[:length]):
                start = max(start, len(text) - length)
                break
        if start < 0 and end < 0:
            return not cast(RwkvParser, self._parser_engine).thinking_enabled
        return end > start

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        delta_length = len(list(delta_ids))
        if delta_length == 0:
            return False
        start = max(0, len(input_ids) - delta_length - len("</think>") + 1)
        prefix_end = len(input_ids) - delta_length
        prefix = self.model_tokenizer.decode(input_ids[start:prefix_end])
        text = self.model_tokenizer.decode(input_ids[start:])
        end = text.rfind("</think>")
        return end >= 0 and end + len("</think>") > len(prefix)

    def reasoning_end_remainder(self, input_ids: Sequence[int]) -> list[int]:
        text = self.model_tokenizer.decode(input_ids[-len("</think>") :])
        _, marker, remainder = text.rpartition("</think>")
        if not marker or not remainder:
            return []
        return self.model_tokenizer.encode(remainder, add_special_tokens=False)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        for end in range(len(input_ids), 0, -1):
            start = max(0, end - len("</think>"))
            text = self.model_tokenizer.decode(input_ids[start:end])
            if text.endswith("</think>"):
                return input_ids[end:]
        return input_ids
