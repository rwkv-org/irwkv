// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use irwkv_tokenizer::test_utils::TestTokenizer;

use super::{ReasoningParserFactory, names};

#[test]
fn factory_registers_thinking_parsers() {
    let factory = ReasoningParserFactory::new();
    assert!(factory.contains(names::RWKV));
    assert_eq!(factory.resolve_name_for_model("rwkv-rs/rwkv7"), None);
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let tokenizer = Arc::new(TestTokenizer::new());
    let factory = ReasoningParserFactory::new();
    let error = match factory.create("missing", tokenizer) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("choose from"));
}
