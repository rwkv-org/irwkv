// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use irwkv_parser::tool::{Result, ToolParserOutput};

use super::{ToolParser, ToolParserFactory, names};
use crate::Error;
use crate::request::ChatTool;

struct FakeToolParser;

impl ToolParser for FakeToolParser {
    fn create(_tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, _chunk: &str, _output: &mut ToolParserOutput) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        Ok(ToolParserOutput::default())
    }

    fn reset(&mut self) -> String {
        String::new()
    }
}

#[test]
fn default_factory_starts_empty() {
    let factory = ToolParserFactory::default();
    assert!(factory.list().is_empty());
}

#[test]
fn factory_contains_and_creates_registered_parsers() {
    let mut factory = ToolParserFactory::default();
    factory.register_parser::<FakeToolParser>("fake");

    assert!(factory.contains("fake"));
    assert!(factory.list().contains(&"fake".to_string()));
    factory.create("fake", &[]).unwrap();
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let factory = ToolParserFactory::default();
    let error = match factory.create("missing", &[]) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ParserUnavailableByName { .. }));
}

#[test]
fn factory_rejects_unknown_models() {
    let factory = ToolParserFactory::default();
    let error = match factory.create_for_model("definitely-unknown-model", &[]) {
        Ok(_) => panic!("expected model lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ParserUnavailableForModel { .. }));
}

#[test]
fn factory_creates_registered_parser_for_model() {
    let mut factory = ToolParserFactory::default();
    factory
        .register_parser::<FakeToolParser>("fake")
        .register_pattern("fake-model", "fake");

    factory.create_for_model("my-fake-model-v1", &[]).unwrap();
}

#[test]
fn factory_registers_rwkv_parser() {
    let factory = ToolParserFactory::new();
    assert_eq!(factory.list(), vec![names::RWKV.to_string()]);
    assert_eq!(
        factory.resolve_name_for_model("rwkv-rs/rwkv7"),
        Some(names::RWKV)
    );
    factory.create_for_model("rwkv-rs/rwkv7", &[]).unwrap();
    assert_eq!(factory.resolve_name_for_model("Qwen/Qwen3"), None);
}
