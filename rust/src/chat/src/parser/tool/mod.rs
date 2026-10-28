// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tool parser registration and selection boundary for `irwkv-chat`.

use std::sync::{Arc, LazyLock};

pub use irwkv_parser::tool::{RwkvToolParser, ToolParser, ToolParserError};

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered tool parsers.
pub mod names {
    pub const RWKV: &str = "rwkv";
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator =
    Arc<dyn Fn(&[ChatTool]) -> irwkv_parser::tool::Result<Box<dyn ToolParser>> + Send + Sync>;

/// Registry and model matcher for tool parsers.
pub type ToolParserFactory = ParserFactory<ToolParserCreator>;

impl ToolParserFactory {
    /// Get the global tool parser factory with built-in registrations and model
    /// mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ToolParserFactory> = LazyLock::new(ToolParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<RwkvToolParser>(names::RWKV)
            .register_pattern("rwkv", names::RWKV);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, Arc::new(T::create))
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str, tools: &[ChatTool]) -> crate::Result<Box<dyn ToolParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "tool",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator.as_ref()(tools).map_err(|error| crate::Error::ParserInitialization {
            kind: "tool",
            name: name.to_string(),
            error: error.into(),
        })
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(
        &self,
        model_id: &str,
        tools: &[ChatTool],
    ) -> crate::Result<Box<dyn ToolParser>> {
        let name = self.resolve_name_for_model(model_id).ok_or_else(|| {
            crate::Error::ParserUnavailableForModel {
                kind: "tool",
                model_id: model_id.to_string(),
            }
        })?;
        self.create(name, tools)
    }
}

#[cfg(test)]
mod tests;
