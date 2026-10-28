# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive end-to-end tests for `min_tokens` in the V1 engine.

Addresses #21950: verify and add CI coverage.

Covers:
1) Basic functionality
2) Stop strings with `min_tokens`
3) EOS behavior with `min_tokens`
4) Edge cases (min_tokens == max_tokens, min_tokens == 0)
5) Multiple stop conditions
"""

import json

import pytest

from irwkv import LLM, SamplingParams
from irwkv.exceptions import IRWKVValidationError
from irwkv.outputs import RequestOutput
from irwkv.sampling_params import StructuredOutputsParams

# Test configuration
TEST_MODEL = "rwkv-rs/rwkv7-g1-st"
GREEDY = 0.0  # Deterministic generation for consistent testing


class MinTokensTestCase:
    """Data class for min_tokens test scenarios."""

    def __init__(
        self,
        name: str,
        min_tokens: int,
        max_tokens: int,
        stop: str | list[str] | None = None,
        expected_min_len: int | None = None,
        expected_exact_len: int | None = None,
    ):
        self.name = name
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.stop = stop
        self.expected_min_len = expected_min_len or min_tokens
        self.expected_exact_len = expected_exact_len

    def __str__(self):
        return (
            f"{self.name}: min={self.min_tokens}, "
            f"max={self.max_tokens}, stop={self.stop}"
        )


# Test scenarios covering all critical cases
MIN_TOKENS_TEST_CASES = [
    # === BASIC FUNCTIONALITY (should work) ===
    MinTokensTestCase(
        name="basic_min_tokens_no_stop",
        min_tokens=8,
        max_tokens=20,
        stop=None,
        expected_min_len=8,
    ),
    MinTokensTestCase(
        name="min_tokens_zero",
        min_tokens=0,
        max_tokens=10,
        stop=None,
        expected_min_len=0,
    ),
    MinTokensTestCase(
        name="min_equals_max_no_stop",
        min_tokens=15,
        max_tokens=15,
        stop=None,
        expected_exact_len=15,
    ),
    MinTokensTestCase(
        name="min_tokens_with_comprehensive_stops",
        min_tokens=5,
        max_tokens=20,
        stop=["a", "e", "i", "o", "u", "t", "n", "s", "r", "l", " "],
        expected_min_len=5,
    ),
    MinTokensTestCase(
        name="min_tokens_with_simple_char_stop",
        min_tokens=3,
        max_tokens=15,
        stop=["e", "a", " "],
        expected_min_len=3,
    ),
    MinTokensTestCase(
        name="min_equals_max_eos_only",
        min_tokens=20,
        max_tokens=20,
        stop=None,
        expected_exact_len=20,
    ),
    # === EDGE CASES ===
    MinTokensTestCase(
        name="large_min_tokens",
        min_tokens=50,
        max_tokens=60,
        stop=None,
        expected_min_len=50,
    ),
    MinTokensTestCase(
        name="min_tokens_with_empty_stop_list",
        min_tokens=5,
        max_tokens=15,
        stop=[],  # Empty stop list
        expected_min_len=5,
    ),
]


@pytest.fixture(scope="module")
def llm_v1():
    """Create V1 LLM instance for testing."""
    llm = LLM(
        model=TEST_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=1024,  # Small context for fast testing
        max_num_seqs=4,
        enforce_eager=True,  # Avoid graph compilation overhead
        generation_config="irwkv",
    )
    return llm


def test_min_tokens_with_terminal_structured_output(llm_v1: LLM):
    """A grammar whose only legal continuation is EOS must stop under
    min_tokens instead of sampling outside the grammar."""
    params = SamplingParams(
        temperature=GREEDY,
        min_tokens=20,
        max_tokens=40,
        structured_outputs=StructuredOutputsParams(
            json=json.dumps({"type": "boolean"})
        ),
    )

    output = llm_v1.generate("Answer:", params)[0].outputs[0]
    eos_token_id = llm_v1.get_tokenizer().eos_token_id

    assert output.finish_reason == "stop"
    assert isinstance(json.loads(output.text), bool)
    assert output.token_ids[-1] == eos_token_id
    assert len(output.token_ids) < params.min_tokens


def get_token_count(output: RequestOutput) -> int:
    """Extract token count from LLM output."""
    if not output.outputs:
        return 0
    return len(output.outputs[0].token_ids)


def assert_min_tokens_satisfied(
    output: RequestOutput, test_case: MinTokensTestCase
) -> None:
    """Assert that min_tokens requirement is satisfied."""
    token_count = get_token_count(output)
    stop_reason = output.outputs[0].stop_reason if output.outputs else "no output"

    if test_case.expected_exact_len is not None:
        # Exact length requirement
        assert token_count == test_case.expected_exact_len, (
            f"Expected exactly {test_case.expected_exact_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {stop_reason}"
        )
    else:
        # Minimum length requirement
        assert token_count >= (test_case.expected_min_len or 0), (
            f"Expected at least {test_case.expected_min_len} tokens, "
            f"got {token_count} tokens. "
            f"Stop reason: {stop_reason}"
        )


@pytest.mark.parametrize(
    "test_case",
    MIN_TOKENS_TEST_CASES,
    ids=lambda tc: tc.name,
)
def test_min_tokens_comprehensive(llm_v1: LLM, test_case: MinTokensTestCase):
    """Comprehensive test for min_tokens functionality in V1 engine.

    This test covers all critical scenarios for min_tokens:
    - Basic functionality
    - Stop strings with min_tokens
    - EOS tokens with min_tokens
    - Edge cases

    Args:
        llm_v1: V1 LLM instance
        test_case: Test scenario parameters

    """
    # Create sampling parameters
    sampling_params = SamplingParams(
        min_tokens=test_case.min_tokens,
        max_tokens=test_case.max_tokens,
        stop=test_case.stop,
        temperature=GREEDY,
        include_stop_str_in_output=True,  # Include stop strings for debugging
    )

    # Use simple prompt. Comprehensive stop lists should catch any generation
    prompt = "Hello"

    # Generate output
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1, "Expected exactly one output"
    output = outputs[0]

    # Debug information
    token_count = get_token_count(output)
    generated_text = output.outputs[0].text if output.outputs else ""
    stop_reason = output.outputs[0].stop_reason if output.outputs else "unknown"

    print(f"\nTest: {test_case.name}")
    print(f"Generated {token_count} tokens")
    print(f"Stop reason: {stop_reason}")
    print(f"Generated text: {repr(generated_text)}")
    print(f"Expected min: {test_case.expected_min_len}")
    if test_case.expected_exact_len:
        print(f"Expected exact: {test_case.expected_exact_len}")

    # Validate min_tokens requirement
    assert_min_tokens_satisfied(output, test_case)


def test_min_tokens_basic_functionality(llm_v1: LLM):
    """Test basic min_tokens functionality without stop conditions.

    This is a baseline test that should always pass and validates
    that min_tokens works correctly in the simple case.
    """
    sampling_params = SamplingParams(min_tokens=10, max_tokens=20, temperature=GREEDY)

    prompt = "Once upon a time"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])

    assert token_count >= 10, f"Expected at least 10 tokens, got {token_count}"
    assert token_count <= 20, f"Expected at most 20 tokens, got {token_count}"


def test_min_tokens_delays_stop_string(llm_v1: LLM):
    """Stop strings do not end generation before the minimum."""
    sampling_params = SamplingParams(
        min_tokens=15,
        max_tokens=50,
        # Common letter; likely appears early
        stop=["e"],
        temperature=GREEDY,
        include_stop_str_in_output=True,
    )

    # Simple prompt that will generate text containing "e"
    prompt = "The quick brown fox"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    generated_text = outputs[0].outputs[0].text if outputs[0].outputs else ""

    # Debug info to understand what happened
    print(f"Generated text: {repr(generated_text)}")
    print(f"Token count: {token_count}")
    print(f"Contains 'e': {'e' in generated_text}")

    stop_reason = (
        outputs[0].outputs[0].stop_reason if outputs[0].outputs else "no output"
    )
    assert token_count >= 15, (
        f"{token_count} tokens < min_tokens=15. "
        f"Reason: {stop_reason}. "
        f"Text: {repr(generated_text)}"
    )


def test_min_tokens_delays_multiple_stop_strings(llm_v1: LLM):
    """Multiple stop strings do not end generation before the minimum."""
    sampling_params = SamplingParams(
        min_tokens=50,  # Set high min_tokens to ensure bug detection
        max_tokens=200,
        # Use multiple very common patterns - at least one will appear
        stop=["e", "a", "i", "o", "u", " ", "t", "n", "s", "r"],
        temperature=GREEDY,
        include_stop_str_in_output=True,
    )

    # Simple prompt that will generate some text
    prompt = "The cat"
    outputs = llm_v1.generate([prompt], sampling_params)

    assert len(outputs) == 1
    token_count = get_token_count(outputs[0])
    generated_text = outputs[0].outputs[0].text if outputs[0].outputs else ""
    stop_reason = outputs[0].outputs[0].stop_reason if outputs[0].outputs else "unknown"

    print(f"Generated text: {repr(generated_text)}")
    print(f"Token count: {token_count}")
    print(f"Stop reason: {stop_reason}")

    finish_reason = (
        outputs[0].outputs[0].finish_reason if outputs[0].outputs else "unknown"
    )

    print(f"Finish reason: {finish_reason}")

    if finish_reason == "stop":
        assert token_count >= 50, (
            f"{token_count} tokens < min_tokens=50. "
            f"Reason: {finish_reason}. "
            f"Text: {repr(generated_text)}"
        )


def test_min_tokens_eos_behavior(llm_v1: LLM):
    """EOS is blocked until the requested minimum has been generated."""
    eos = llm_v1.get_tokenizer().eos_token_id
    fallback = 1 if eos != 1 else 2
    common = {
        "max_tokens": 8,
        "temperature": GREEDY,
        "allowed_token_ids": [eos, fallback],
        "logit_bias": {eos: 100},
    }

    without_min = llm_v1.generate("Hello", SamplingParams(**common))[0].outputs[0]
    with_min = llm_v1.generate("Hello", SamplingParams(min_tokens=8, **common))[
        0
    ].outputs[0]

    assert without_min.finish_reason == "stop"
    assert without_min.token_ids == [eos]
    assert with_min.finish_reason == "length"
    assert with_min.token_ids == [fallback] * 8


def test_min_tokens_validation():
    """Test that SamplingParams correctly validates min_tokens parameters.

    This tests the parameter validation logic in SamplingParams.
    """
    # Valid cases
    SamplingParams(min_tokens=0, max_tokens=10)
    SamplingParams(min_tokens=5, max_tokens=10)
    SamplingParams(min_tokens=10, max_tokens=10)

    # Invalid cases
    with pytest.raises(
        IRWKVValidationError,
        match="min_tokens must be greater than or equal to 0",
    ):
        SamplingParams(min_tokens=-1, max_tokens=10)

    with pytest.raises(
        IRWKVValidationError,
        match="min_tokens must be less than or equal to max_tokens",
    ):
        SamplingParams(min_tokens=15, max_tokens=10)


if __name__ == "__main__":
    """
    Run tests locally for development.
    
    Usage:
        cd irwkv/
        python -m pytest tests/v1/e2e/general/test_min_tokens.py -v
    """
    pytest.main([__file__, "-v"])
