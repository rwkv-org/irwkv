// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Reasoning parser registration and selection boundary for `irwkv-chat`.

use std::sync::{Arc, LazyLock};

pub use irwkv_parser::reasoning::{
    ReasoningDelta, ReasoningError, ReasoningParser, RwkvReasoningParser,
};
use irwkv_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;

/// Canonical public names for registered reasoning parsers.
pub mod names {
    pub const RWKV: &str = "rwkv";
}

/// Constructor signature for one registered reasoning parser implementation.
type ReasoningParserCreator = Arc<
    dyn Fn(DynTokenizer) -> irwkv_parser::reasoning::Result<Box<dyn ReasoningParser>> + Send + Sync,
>;

/// Registry and model matcher for reasoning parsers.
pub type ReasoningParserFactory = ParserFactory<ReasoningParserCreator>;

impl ReasoningParserFactory {
    /// Get the global reasoning parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ReasoningParserFactory> =
            LazyLock::new(ReasoningParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory.register_parser::<RwkvReasoningParser>(names::RWKV);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ReasoningParser + 'static,
    {
        self.register_creator(name, Arc::new(T::create))
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn ReasoningParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "reasoning",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator.as_ref()(tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "reasoning",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

#[cfg(test)]
mod tests;
