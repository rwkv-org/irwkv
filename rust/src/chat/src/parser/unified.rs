// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `irwkv-chat`.

use std::sync::LazyLock;

pub use irwkv_parser::unified::{
    HfTemplateError, HfUnifiedParser, ResponseTemplate, UnifiedParser,
};
use irwkv_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered unified parsers.
pub mod names {
    pub const HF: &str = "hf";
}

/// Constructor signature for one registered unified parser implementation.
type UnifiedParserCreator =
    fn(&[ChatTool], DynTokenizer) -> irwkv_parser::unified::Result<Box<dyn UnifiedParser>>;

/// Registry and model matcher for unified parsers.
pub type UnifiedParserFactory = ParserFactory<UnifiedParserCreator>;

impl UnifiedParserFactory {
    /// Get the global unified parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<UnifiedParserFactory> = LazyLock::new(UnifiedParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory.register_parser::<HfUnifiedParser>(names::HF);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: UnifiedParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tools: &[ChatTool],
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn UnifiedParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "unified",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools, tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "unified",
            name: name.to_string(),
            error: error.into(),
        })
    }
}
