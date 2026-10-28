// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared helpers for streaming parsers.

use winnow::error::{ContextError, ErrMode, ModalResult, Needed};
use winnow::stream::{Offset, Partial, Stream};

use crate::tool::{Result, ToolParserError};

/// Return the byte length of the longest proper prefix of `token` that is also
/// a suffix of `buffer`.
///
/// Streaming parsers use this to keep only the trailing fragment that might
/// still grow into a full marker after the next decoded chunk arrives.
///
/// The returned length is always a valid UTF-8 boundary in `token`, so callers
/// can safely slice `&token[..len]` even when markers contain non-ASCII
/// characters such as DeepSeek's DSML delimiters.
pub fn partial_prefix_len(buffer: &str, token: &str) -> usize {
    let Some(first_byte) = token.as_bytes().first().copied() else {
        return 0;
    };

    let max_len = buffer.len().min(token.len().saturating_sub(1));
    let tail_start = buffer.len() - max_len;
    let buffer_bytes = buffer.as_bytes();
    let token_bytes = token.as_bytes();

    // Scan from the longest possible suffix to preserve overlapping prefixes.
    for index in tail_start..buffer.len() {
        if buffer_bytes[index] != first_byte {
            continue;
        }

        let len = buffer.len() - index;
        if buffer.is_char_boundary(index)
            && token.is_char_boundary(len)
            && token_bytes[..len] == buffer_bytes[index..]
        {
            return len;
        }
    }

    0
}

/// Parse text before the first complete or partial marker.
pub fn safe_text_len_mul(input: &mut Partial<&str>, markers: &[&str]) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }
    let found = markers.iter().filter_map(|marker| text.find(marker)).min();
    let stop = found.unwrap_or_else(|| {
        let keep = markers.iter().map(|marker| partial_prefix_len(text, marker)).max().unwrap_or(0);
        text.len() - keep
    });
    if stop == 0 {
        return if found.is_some() {
            Err(ErrMode::Backtrack(ContextError::new()))
        } else {
            incomplete()
        };
    }
    input.next_slice(stop);
    Ok(stop)
}

/// Parse one event from a buffered streaming input.
///
/// Returns:
/// - `Ok(Some((event, consumed_len)))` if an event was successfully parsed, along with the number
///   of bytes consumed from the buffer.
/// - `Ok(None)` if the buffer does not contain a full event yet, and more data is needed.
/// - `Err` if a parsing error occurred.
pub fn parse_buffered_event<E>(
    buffer: &str,
    parse: impl FnOnce(&mut Partial<&str>) -> ModalResult<E>,
) -> Result<Option<(E, usize)>> {
    let mut input = Partial::new(buffer);
    let checkpoint = input.checkpoint();
    let event = match parse(&mut input) {
        Ok(event) => event,
        Err(ErrMode::Incomplete(_)) => return Ok(None),
        Err(ErrMode::Backtrack(e) | ErrMode::Cut(e)) => {
            let snippet = buffer.char_indices().nth(80).map_or(buffer, |(i, _)| &buffer[..i]);
            return Err(ToolParserError::ParsingFailed {
                message: format!("near {snippet:?}: {e}"),
            });
        }
    };
    let consumed_len = input.offset_from(&checkpoint);
    if consumed_len == 0 {
        return Ok(None);
    }

    Ok(Some((event, consumed_len)))
}

/// Returns an error indicating that we need more data to continue parsing.
pub fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}

#[cfg(test)]
mod tests {

    use winnow::error::ErrMode;
    use winnow::stream::{Offset, Partial, Stream};

    use super::{parse_buffered_event, partial_prefix_len, safe_text_len_mul};

    #[test]
    fn partial_prefix_len_handles_ascii_markers() {
        assert_eq!(
            partial_prefix_len("hello<|tool", "<|tool_call>"),
            "<|tool".len()
        );
        assert_eq!(partial_prefix_len("hello world", "<|tool_call>"), 0);
    }

    #[test]
    fn partial_prefix_len_prefers_longest_overlapping_prefix() {
        assert_eq!(partial_prefix_len("chunk ending in aba", "ababa"), 3);
    }

    #[test]
    fn partial_prefix_len_handles_unicode_markers() {
        let token = "<｜DSML｜function_calls>";
        assert_eq!(
            partial_prefix_len("prefix <｜DSML｜fun", token),
            "<｜DSML｜fun".len()
        );
        assert_eq!(partial_prefix_len("prefix <｜DSML", token), "<｜DSML".len());
    }

    #[test]
    fn safe_text_len_mul_stops_before_earliest_marker() {
        let mut input = Partial::new("hello<channel|><|tool_call>");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<channel|>"]).unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
        assert_eq!(*input, "<channel|><|tool_call>");
    }

    #[test]
    fn safe_text_len_mul_holds_back_longest_partial_marker() {
        let mut input = Partial::new("hello<|tool");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
        assert_eq!(*input, "<|tool");
    }

    #[test]
    fn safe_text_len_mul_skips_false_same_prefix_candidate() {
        let mut input = Partial::new("hello<not_marker><|tool_call>");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap();

        assert_eq!(len, "hello<not_marker>".len());
        assert_eq!(input.offset_from(&checkpoint), "hello<not_marker>".len());
        assert_eq!(*input, "<|tool_call>");
    }

    #[test]
    fn safe_text_len_mul_reports_incomplete_for_only_partial_marker() {
        let mut input = Partial::new("<|channel>thought");

        let error =
            safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }

    #[test]
    fn parse_buffered_event_error_includes_input_snippet() {
        let result = parse_buffered_event::<()>(" {\"x\":1}", |_| {
            Err(ErrMode::Backtrack(winnow::error::ContextError::new()))
        });
        let err = result.unwrap_err().to_string();
        assert!(err.contains("near \""), "error must include snippet");
    }

    #[test]
    fn parse_buffered_event_error_truncates_long_input() {
        let long_input = format!(" {}", "x".repeat(100));
        let result = parse_buffered_event::<()>(&long_input, |_| {
            Err(ErrMode::Backtrack(winnow::error::ContextError::new()))
        });
        let err = result.unwrap_err().to_string();
        assert!(err.contains("near \""), "error must include snippet");
        assert!(
            !err.contains(&long_input),
            "snippet must be truncated for long input"
        );
    }
}
