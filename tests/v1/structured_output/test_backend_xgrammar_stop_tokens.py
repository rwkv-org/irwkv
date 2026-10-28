# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for issue #42403.

A request's stop-token set (the generation_config eos list plus any
user-supplied ``stop_token_ids``) is invisible to xgrammar, which only knows
the tokenizer's single eos. Such tokens can therefore escape the grammar
bitmask while the FSM is still mid-object and truncate structured output.
``compile_grammar`` now forwards ``all_stop_token_ids`` to the matcher as
``override_stop_tokens`` so xgrammar masks them until the grammar completes.
"""

import os

import pytest
from transformers import AutoTokenizer

from irwkv.config import IrwkvConfig, StructuredOutputsConfig
from irwkv.v1.structured_output.backend_types import StructuredOutputOptions
from irwkv.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = os.environ.get("IRWKV_RWKV_TEST_MODEL", "rwkv-rs/rwkv7-g1-st")


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    irwkv_config = IrwkvConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar")
    )
    return XgrammarBackend(
        irwkv_config,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
    )


def test_request_stop_tokens_gated_to_grammar_terminal(backend: XgrammarBackend):
    eos = backend.tokenizer.eos_token_id
    (quote,) = backend.tokenizer.encode('"', add_special_tokens=False)
    (letter,) = backend.tokenizer.encode("X", add_special_tokens=False)
    schema = '{"type": "string"}'
    default = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    override = backend.compile_grammar(
        StructuredOutputOptions.JSON, schema, stop_token_ids={eos, letter}
    )

    # Open the string: both grammars are now in a non-terminal state.
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [quote])

    bm_default = backend.allocate_token_bitmask(1)
    bm_override = backend.allocate_token_bitmask(1)
    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # Mid-string, the plain token is valid content, so the default grammar
    # leaves it samplable -- this is the leak. Registering it as a stop token
    # masks it until the grammar can terminate.
    assert _token_allowed(bm_default[0], letter)
    assert not _token_allowed(bm_override[0], letter)

    # Close the string -> accepting state (grammar complete, not yet terminated).
    for grammar in (default, override):
        assert grammar.accept_tokens("req", [quote])
        assert not grammar.is_terminated()

    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    # The extra stop token may now terminate under the override, never under
    # the default grammar -- and the tokenizer's own eos still terminates both,
    # so default termination is preserved.
    assert not _token_allowed(bm_default[0], letter)
    assert _token_allowed(bm_override[0], letter)
    assert _token_allowed(bm_default[0], eos)
    assert _token_allowed(bm_override[0], eos)
