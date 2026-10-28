// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Streaming reasoning parsers for chat completions.
//!
//! The key design choice here is that parser initialization prefers the
//! *actual rendered prompt state* over model-family conventions. When a stream
//! starts, each parser receives the prompt token IDs and inspects the last
//! reasoning boundary that is already present in the prompt. In practice this
//! is a more faithful signal than hardcoding assumptions such as "this model
//! always starts in reasoning" or "this model always emits `<think>` itself".
//!
//! That prompt-first initialization lets multiple model families share the
//! same incremental parser implementation even when older Python parsers split
//! them apart. If two families use the same textual delimiters and differ
//! mostly in how their chat templates prefill `<think>` / `</think>`, they can
//! usually reuse one parser here because the prompt token IDs already tell us
//! which state the stream is entering with.

mod delimited;
mod rwkv {
    use irwkv_tokenizer::{DecodedText, DynTokenizer};
    use xgrammar_structural_tag::format::Format;

    use crate::output_grammar::{self, OutputGrammarContext};

    use super::{
        DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
        Result,
    };

    /// Reasoning parser for the RWKV.
    ///
    /// This parser uses standard `<think>...</think>` delimiters.
    pub struct RwkvReasoningParser {
        inner: DelimitedReasoningParser,
    }

    impl RwkvReasoningParser {
        /// Create a RWKV parser backed by the shared delimited state machine.
        pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
            Ok(Self {
                inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                    .with_after_start("\n")
                    .with_before_end("\n")
                    .with_after_end("\n\n")
                    .build()?,
            })
        }
    }

    impl ReasoningParser for RwkvReasoningParser {
        fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self::new(tokenizer)?))
        }

        fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
            self.inner.initialize(prompt_token_ids)
        }

        fn wrap_visible_format(
            &self,
            _ctx: &OutputGrammarContext<'_>,
            visible: &Format,
        ) -> output_grammar::Result<Option<Format>> {
            Ok(Some(self.inner.wrap_visible_format(visible)))
        }

        fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
            Ok(self.inner.push(delta))
        }

        fn finish(&mut self) -> Result<ReasoningDelta> {
            Ok(self.inner.finish())
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::Arc;

        use irwkv_tokenizer::test_utils::TestTokenizer;

        use super::RwkvReasoningParser;
        use crate::reasoning::ReasoningParser;
        use crate::reasoning::tests::{
            SPECIAL_BOUNDARY_ID, THINK_END_ID, THINK_START_ID, content_str, fake_tokenizer,
            push_str, reasoning_str,
        };

        #[test]
        fn rwkv_without_prompt_markers_expects_start_token() {
            let tokenizer = Arc::new(fake_tokenizer());
            let mut parser = RwkvReasoningParser::new(tokenizer).unwrap();

            let delta = push_str(&mut parser, "reason</think>answer");
            assert_eq!(delta.reasoning, None);
            assert_eq!(content_str(&delta), Some("reason</think>answer"));
        }

        #[test]
        fn rwkv_prompt_end_marker_starts_in_content() {
            let tokenizer = Arc::new(fake_tokenizer());
            let mut parser = RwkvReasoningParser::new(tokenizer).unwrap();
            parser.initialize(&[THINK_END_ID]).unwrap();

            let delta = push_str(&mut parser, "answer");
            assert_eq!(delta.reasoning, None);
            assert_eq!(content_str(&delta), Some("answer"));
        }

        #[test]
        fn rwkv_tolerates_old_and_new_formats() {
            let tokenizer = Arc::new(fake_tokenizer());

            let mut old_parser = RwkvReasoningParser::new(tokenizer.clone()).unwrap();
            let old = push_str(&mut old_parser, "<think>reason</think>answer");
            assert_eq!(reasoning_str(&old), Some("reason"));
            assert_eq!(content_str(&old), Some("answer"));

            let mut new_parser = RwkvReasoningParser::new(tokenizer).unwrap();
            new_parser.initialize(&[THINK_START_ID]).unwrap();
            let new = push_str(&mut new_parser, "reason</think>answer");
            assert_eq!(reasoning_str(&new), Some("reason"));
            assert_eq!(content_str(&new), Some("answer"));
        }

        #[test]
        fn rwkv_stops_scanning_at_last_special_token() {
            let tokenizer = Arc::new(fake_tokenizer());
            let mut parser = RwkvReasoningParser::new(tokenizer).unwrap();

            parser.initialize(&[THINK_START_ID, SPECIAL_BOUNDARY_ID]).unwrap();

            let delta = push_str(&mut parser, "answer");
            assert_eq!(delta.reasoning, None);
            assert_eq!(content_str(&delta), Some("answer"));
        }

        #[test]
        fn reasoning_markers_can_span_tokens_and_continue_from_prompt() {
            let tokenizer = Arc::new(
                TestTokenizer::new()
                    .with_regular_token("<think", 300)
                    .with_regular_token("</think", 301),
            );
            let mut parser = RwkvReasoningParser::new(tokenizer).unwrap();

            parser.initialize(&[300]).unwrap();
            let delta = push_str(&mut parser, ">work</think>answer");
            assert_eq!(reasoning_str(&delta), Some("work"));
            assert_eq!(content_str(&delta), Some("answer"));

            parser.initialize(&[300, u32::from(b'>'), 301]).unwrap();
            let delta = push_str(&mut parser, ">answer");
            assert_eq!(delta.reasoning, None);
            assert_eq!(content_str(&delta), Some("answer"));
        }
    }
}

use irwkv_tokenizer::{DecodedText, DynTokenizer};
use thiserror::Error;
use xgrammar_structural_tag::format::Format;

use crate::output_grammar::{self, OutputGrammarContext};

pub(crate) use self::delimited::{DelimitedReasoningParser, DelimitedReasoningParserBuilder};
pub use self::rwkv::RwkvReasoningParser;

/// Result alias for reasoning parser operations.
pub type Result<T> = std::result::Result<T, ReasoningError>;

/// One parsed streaming delta split into reasoning and visible content.
///
/// Each portion carries the attributions of the generated tokens that produced
/// it, so downstream consumers can count reasoning tokens exactly. Marker
/// spans are dropped by the parsers, keeping marker tokens out of any count.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: Option<DecodedText>,
    pub content: Option<DecodedText>,
}

impl ReasoningDelta {
    /// Return true when this delta carries neither reasoning nor content text.
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }

    /// Append text to the reasoning portion, creating it on first use.
    ///
    /// A piece with empty text but non-empty attributions (zero-width tokens
    /// only) is still kept: the tokens are attributed to the current state.
    pub(crate) fn push_reasoning(&mut self, piece: DecodedText) {
        if piece.is_empty() {
            return;
        }
        match &mut self.reasoning {
            Some(existing) => existing.append(piece),
            None => self.reasoning = Some(piece),
        }
    }

    /// Append text to the visible content portion, creating it on first use.
    ///
    /// A piece with empty text but non-empty attributions (zero-width tokens
    /// only) is still kept: the tokens are attributed to the current state.
    pub(crate) fn push_content(&mut self, piece: DecodedText) {
        if piece.is_empty() {
            return;
        }
        match &mut self.content {
            Some(existing) => existing.append(piece),
            None => self.content = Some(piece),
        }
    }
}

/// Incremental parser that splits decoded text deltas into reasoning and
/// content.
pub trait ReasoningParser: Send {
    /// Construct a boxed parser instance for one request stream.
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static;

    /// Initialize parser state from prompt token IDs before output deltas
    /// arrive.
    fn initialize(&mut self, _prompt_token_ids: &[u32]) -> Result<()> {
        Ok(())
    }

    /// Return whether decoded output must preserve tokenizer special tokens.
    ///
    /// Some model families emit reasoning sentinels as special tokens. Those
    /// parsers need `skip_special_tokens = false` while parsing is enabled.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Prepend the reasoning phase implied by the initialized parser state.
    fn wrap_visible_format(
        &self,
        _ctx: &OutputGrammarContext<'_>,
        _visible: &Format,
    ) -> output_grammar::Result<Option<Format>> {
        Ok(None)
    }

    /// Feed one decoded text delta into the parser.
    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta>;

    /// Flush any buffered partial delimiter state at end of stream.
    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(ReasoningDelta::default())
    }
}

/// Errors produced while creating or running reasoning parsers.
#[derive(Debug, Error)]
pub enum ReasoningError {
    #[error(transparent)]
    Tokenizer(#[from] irwkv_tokenizer::TokenizerError),
    #[error("tokenizer is missing reasoning delimiter token `{token}`")]
    MissingToken { token: String },
    #[error(
        "`{name}` only provides a unified parser; the same reasoning parser and tool parser should be specified together"
    )]
    DummyUnifiedParser { name: String },
}

#[cfg(test)]
mod tests;
