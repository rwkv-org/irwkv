# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from irwkv import SamplingParams
from irwkv.exceptions import IRWKVValidationError
from irwkv.platforms import current_platform
from tests.utils import large_gpu_mark


def _model_config(vocab_size: int = 10, max_logprobs: int = 20):
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type="rwkv"),
        max_logprobs=max_logprobs,
        max_model_len=16,
        logits_processors=None,
        get_vocab_size=lambda: vocab_size,
    )


@pytest.mark.parametrize("token_ids", [[0], [0, 9]])
def test_logprob_token_ids_validate_vocab_bounds_valid(token_ids: list[int]):
    SamplingParams(logprob_token_ids=token_ids).verify(
        _model_config(), structured_outputs_config=None, tokenizer=None
    )


@pytest.mark.parametrize("token_ids", [[-1], [10], [-35, 1873042417]])
def test_logprob_token_ids_validate_vocab_bounds_invalid(token_ids: list[int]):
    with pytest.raises(IRWKVValidationError, match="logprob_token_ids"):
        SamplingParams(logprob_token_ids=token_ids).verify(
            _model_config(), structured_outputs_config=None, tokenizer=None
        )


def test_prompt_logprob_token_ids_validation():
    """Rows may be ragged (-1 padded); the width is bounded by max_logprobs."""
    model_config = _model_config(vocab_size=100, max_logprobs=4)

    def verify(**kwargs):
        params = SamplingParams(**kwargs)
        params.verify(model_config, structured_outputs_config=None, tokenizer=None)
        return params.prompt_logprob_token_ids

    assert verify(prompt_logprob_token_ids=[[], [1], [4, 3, 2, 1]]).tolist() == [
        [-1] * 4,
        [1, -1, -1, -1],
        [4, 3, 2, 1],
    ]
    # Any integer layout is normalized to what the worker uploads:
    # C-contiguous native int32.
    odd = np.asfortranarray(np.array([[1, 2], [3, -1]], dtype=">i8"))
    ids = verify(prompt_logprob_token_ids=odd)
    assert ids.dtype == np.int32 and ids.flags.c_contiguous
    assert ids.tolist() == [[1, 2], [3, -1]]
    # The pydantic path (token-in/token-out API, /docs) takes strict integer lists.
    adapter = TypeAdapter(SamplingParams)
    for bad in ("[[1.5]]", '[["3"]]', "[[true]]", "[1, 2]"):
        with pytest.raises(ValidationError):
            adapter.validate_json(f'{{"prompt_logprob_token_ids": {bad}}}')

    with pytest.raises(IRWKVValidationError, match=r"shape \[num_rows, num_ids\]"):
        verify(prompt_logprob_token_ids=np.array([1, 2]))
    with pytest.raises(IRWKVValidationError, match=r"shape \[num_rows, num_ids\]"):
        verify(prompt_logprob_token_ids=[1, 2])
    for bad_ids in ([[None]], [[2**63]], [[[1]]], [[[1], 2]]):
        with pytest.raises(IRWKVValidationError, match="integer token ids"):
            verify(prompt_logprob_token_ids=bad_ids)
    with pytest.raises(IRWKVValidationError, match="out-of-vocab"):
        verify(prompt_logprob_token_ids=np.array([[2**64 - 1]], dtype=np.uint64))
    # The error names the value to raise max_logprobs to.
    with pytest.raises(IRWKVValidationError, match=r"max_logprobs.*at least 5"):
        verify(prompt_logprob_token_ids=[[1], [1, 2, 3, 4, 5]])
    with pytest.raises(IRWKVValidationError, match="out-of-vocab"):
        verify(prompt_logprob_token_ids=[[1], [100]])
    with pytest.raises(IRWKVValidationError, match="out-of-vocab"):
        verify(prompt_logprob_token_ids=[[1], [-2]])
    with pytest.raises(IRWKVValidationError, match="non-empty"):
        verify(prompt_logprob_token_ids=[[], []])
    # Rows are bounded by max_model_len before the table is padded.
    with pytest.raises(IRWKVValidationError, match="at most 15 scored rows"):
        verify(prompt_logprob_token_ids=[[]] * 15 + [[1]])
    # prompt_logprob_start alone is a caller mistake, not a silent no-op.
    with pytest.raises(IRWKVValidationError, match="requires prompt_logprob_token_ids"):
        verify(prompt_logprob_start=3)


class TestCorrectDecodedToken:
    """Unit tests for _correct_decoded_token method in LogprobsProcessor.

    This method handles UTF-8 decoding issues where incomplete byte sequences
    result in the Unicode replacement character "�" (U+FFFD). This commonly
    happens with byte-fallback tokenization when multi-byte UTF-8 characters
    are split across tokens.

    The method signature is _correct_decoded_token(token_id, context_token_ids)
    where token_id is the single token to correct and context_token_ids are
    the preceding sampled tokens in sequential order.
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        from unittest.mock import Mock

        tokenizer = Mock()
        return tokenizer

    @pytest.fixture
    def processor(self, mock_tokenizer):
        """Create a LogprobsProcessor."""
        from irwkv.v1.engine.logprobs import LogprobsProcessor

        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )
        return processor

    def test_correction_with_context(self, processor):
        """Test correction using context from preceding sampled tokens.

        Scenario: A byte-fallback token that completes a multi-byte
        UTF-8 sequence when decoded with context.
        """

        # Context is [101] (a preceding sampled token)
        # Token 102 individually decodes to "�"
        # decode([101, 102]) returns "valid" (complete sequence)
        def mock_decode(ids):
            if ids == [101, 102]:
                return "hello valid"
            if ids == [101]:
                return "hello "
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(102, [101])
        assert result == "valid"

    def test_correction_with_context_from_logprobs(self, processor):
        """Test correction using context from previous logprob entries.

        Scenario: Token decoded with context from previously sampled
        tokens completes a UTF-8 sequence.
        """

        # Token 123 was previously sampled (in context)
        def mock_decode(ids):
            if ids == [123, 100]:
                return 'hello "polarized"'
            if ids == [123]:
                return "hello "
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(100, [123])
        assert result == '"polarized"'

    def test_correction_no_context(self, processor):
        """Test correction with no context available.

        Should return empty string as fallback.
        """
        processor.tokenizer.decode.return_value = "�"

        result = processor._correct_decoded_token(100, [])
        assert result == ""

    def test_correction_with_context_succeeds(self, processor):
        """Test correction with context from previously sampled tokens."""

        def mock_decode(ids):
            if ids == [123, 200]:
                return "hello corrected"
            if ids == [123]:
                return "hello "
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        result = processor._correct_decoded_token(200, [123])
        assert result == "corrected"

    def test_fallback_when_all_attempts_fail(self, processor):
        """Test fallback to empty string when no correction works."""
        processor.tokenizer.decode.return_value = "still�"

        result = processor._correct_decoded_token(102, [100, 101])
        assert result == ""

    def test_increasing_context_window(self, processor):
        """Test that increasing context window finds the correction.

        Scenario: 3-byte UTF-8 char. With 1 context token, still
        incomplete. With 2 context tokens, completes the sequence.
        """

        def mock_decode(ids):
            # 1 context token: still incomplete
            if ids == [81, 82]:
                return "�"
            # 2 context tokens: complete
            if ids == [80, 81, 82]:
                return "\u201c"
            # Context-only decodes
            if ids == [81]:
                return "�"
            if ids == [80, 81]:
                return "�"
            return "�"

        processor.tokenizer.decode.side_effect = mock_decode

        # Context has 2 preceding tokens [80, 81]
        result = processor._correct_decoded_token(82, [80, 81])
        assert result == "\u201c"

    def test_multiple_consecutive_replacement_chars(self, processor):
        """Test handling of multiple consecutive replacement characters.

        Scenario: Multi-byte sequence where intermediate bytes return
        empty string and the final byte returns the complete character.
        """
        processor.tokenizer.decode.return_value = "still�"

        # First byte with no useful context: returns ""
        result1 = processor._correct_decoded_token(100, [50])
        assert result1 == ""

        # Second byte with same context: still returns ""
        result2 = processor._correct_decoded_token(101, [50])
        assert result2 == ""

    def test_correction_with_multibyte_utf8(self, processor):
        """Test correction involving multi-byte UTF-8 characters.

        Scenario: Byte-fallback tokenization splits curly quotes.
        The last byte token should produce the complete character.
        """

        def mock_decode(ids):
            # Context [123] + first byte: completes to left curly quote
            if ids == [123, 200]:
                return "hello \u201c"
            if ids == [123]:
                return "hello "
            # Context [123] + second byte: completes to right curly quote
            if ids == [123, 201]:
                return "hello \u201d"
            return "\ufffd"

        processor.tokenizer.decode.side_effect = mock_decode

        # Each top-k token is corrected independently with same context
        result1 = processor._correct_decoded_token(200, [123])
        assert result1 == "\u201c"

        result2 = processor._correct_decoded_token(201, [123])
        assert result2 == "\u201d"

    def test_topk_tokens_corrected_independently(self, processor):
        """Test that top-k alternatives at the same position are each
        corrected independently using only sequential context, not
        each other.

        This is the core fix for issue #27300: when logprobs > 0,
        alternative tokens must not be combined with each other.
        """
        # Context: previously sampled token 50
        context = [50]

        def mock_decode(ids):
            # Token 100 (sampled) with context
            if ids == [50, 100]:
                return "prefix \u201c"
            # Token 200 (top-k alternative) with context
            if ids == [50, 200]:
                return "prefix \u2014"
            # Context alone
            if ids == [50]:
                return "prefix "
            return "\ufffd"

        processor.tokenizer.decode.side_effect = mock_decode

        # Both tokens at the same position use the SAME context [50]
        result_sampled = processor._correct_decoded_token(100, context)
        assert result_sampled == "\u201c"

        result_alt = processor._correct_decoded_token(200, context)
        assert result_alt == "\u2014"

    def test_real_world_opt125m_scenario(self, mock_tokenizer):
        """Test the real-world scenario from the bug report.

        Simulates the OPT-125m sequence where curly quotes are split
        into byte-fallback tokens. Each token is corrected using only
        the preceding sampled tokens as context.
        """
        from irwkv.v1.engine.logprobs import LogprobsProcessor

        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )

        # Simulating: byte tokens 3, 4 form left curly quote "\u201c"
        # byte tokens 8, 9 form right curly quote "\u201d"
        def mock_decode(ids):
            # Context decodes
            if ids == [2]:
                return " term"
            if ids == [1, 2]:
                return " the term"
            if ids == [3]:
                return "\ufffd"
            if ids == [2, 3]:
                return " term\ufffd"
            if ids == [1, 2, 3]:
                return " the term\ufffd"
            # Token 4 with context [2, 3] -> completes left curly quote
            if ids == [3, 4]:
                return "\u201c"
            if ids == [2, 3, 4]:
                return " term\u201c"
            # Context for right curly quote
            if ids == [7]:
                return "ized"
            if ids == [7, 8]:
                return "ized\ufffd"
            if ids == [8, 9]:
                return "\u201d"
            if ids == [7, 8, 9]:
                return "ized\u201d"
            return "normal_text"

        mock_tokenizer.decode.side_effect = mock_decode

        # First byte (token 3) of left curly quote with no context
        result = processor._correct_decoded_token(3, [])
        assert result == ""

        # First byte (token 3) with context [2] -> still incomplete
        result = processor._correct_decoded_token(3, [2])
        assert result == ""

        # Second byte (token 4) of left curly quote with context [2, 3]
        # Token 3 is byte-fallback, so clean context is [2] only.
        # decode([2, 3, 4]) = " term\u201c", decode([2]) = " term"
        # result = "\u201c"
        result = processor._correct_decoded_token(4, [2, 3])
        assert result == "\u201c"

        # Second byte (token 9) of right curly quote with context [7, 8]
        result = processor._correct_decoded_token(9, [7, 8])
        assert result == "\u201d"

    def test_byte_fallback_context_preserves_space(self, mock_tokenizer):
        """Test that text from byte-fallback context tokens is preserved.

        In OPT-125m, token 44 = space + 2 bytes of curly quote.
        When token 44 returns "" (incomplete), the space it carried
        must be attributed to the completing token (48).
        """
        from irwkv.v1.engine.logprobs import LogprobsProcessor

        processor = LogprobsProcessor(
            tokenizer=mock_tokenizer,
            logprobs=[],
            prompt_logprobs=None,
            cumulative_logprob=0.0,
            num_logprobs=1,
            num_prompt_logprobs=None,
        )

        def mock_decode(ids):
            # Token 44 = space + 2 bytes (like OPT-125m's \u0120\u00e2\u0080)
            if ids == [44]:
                return " \ufffd"
            if ids == [48]:
                return "\ufffd"
            # Together they form: space + left curly quote
            if ids == [44, 48]:
                return " \u201c"
            # With preceding clean context
            if ids == [1385]:
                return " term"
            if ids == [1385, 44]:
                return " term \ufffd"
            if ids == [1385, 44, 48]:
                return " term \u201c"
            return "\ufffd"

        mock_tokenizer.decode.side_effect = mock_decode

        # Token 44 with context [1385] -> still ends with replacement
        result = processor._correct_decoded_token(44, [1385])
        assert result == ""

        # Token 48 with context [1385, 44]:
        # Token 44 is byte-fallback, so clean context is [1385].
        # decode([1385, 44, 48]) = " term \u201c"
        # decode([1385]) = " term"
        # result = " \u201c" (space preserved from token 44!)
        result = processor._correct_decoded_token(48, [1385, 44])
        assert result == " \u201c"


@large_gpu_mark(min_gb=24)
def test_token_logprobs_large_batch_int64_row_offset():
    """Regression: logprob kernel row offset (row * vocab_size) must use int64.

    The rejection-sampler logprobs path runs the logprob kernels over the
    spec-expanded logits batch, so batch_size * vocab_size can exceed 2**31
    (e.g. DFlash drafts K tokens per request). With int32 offset arithmetic the
    per-row pointer wraps to a negative address and the kernel hits a CUDA
    illegal memory access. Run over a batch where batch_size * vocab_size > 2**31
    and check the highest-offset row matches a reference log-softmax.
    """
    if not current_platform.is_cuda():
        pytest.skip("int32 row-offset overflow is a CUDA kernel issue")
    from irwkv.v1.worker.gpu.sample.logprob import compute_token_logprobs

    device = torch.device("cuda")
    vocab_size = 131072
    batch_size = 2**31 // vocab_size + 64  # batch_size * vocab_size > 2**31
    # logits (the large input) plus small logprob/rank outputs; ~1 GB headroom.
    required_bytes = batch_size * vocab_size * 4 + (1 << 30)
    if torch.accelerator.get_memory_info()[0] < required_bytes:
        pytest.skip(f"needs ~{required_bytes / 1e9:.0f} GB of free GPU memory")

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    token_ids = torch.full((batch_size, 1), 7, device=device, dtype=torch.int64)
    logprobs = compute_token_logprobs(logits, token_ids)
    torch.accelerator.synchronize()  # surface any async illegal memory access
    last = batch_size - 1
    ref = torch.log_softmax(logits[last].float(), dim=-1)[7]
    assert torch.allclose(logprobs[last, 0], ref, atol=1e-2), (
        f"logprob {logprobs[last, 0].item()} != ref {ref.item()}"
    )
