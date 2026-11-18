// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{
    Result, RwkvToolParser, Tool, ToolCallDelta, ToolParser, ToolParserEvent, ToolParserOutput,
};
use crate::tool::ToolParserTestExt as _;

#[test]
fn rwkv_parser_accepts_split_markers_and_parallel_calls() {
    use super::test_utils::{collect_stream, split_by_chars, test_tools};

    let input = "Searching. **Tool Call:**\n```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n```\n**Tool Call:**\n```json\n{\"name\":\"add\",\"arguments\":{\"x\":1}}\n```";
    let mut parser = RwkvToolParser::create(&test_tools()).unwrap();
    let output = collect_stream(parser.as_mut(), &split_by_chars(input, 2));

    assert_eq!(output.normal_text(), "Searching. ");
    assert_eq!(output.calls().len(), 2);
    assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    assert_eq!(output.calls()[0].arguments, "{\"city\":\"Paris\"}");
    assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
}

#[test]
fn rwkv_parser_preserves_unrecognized_call_as_text() {
    use super::test_utils::test_tools;

    let text = "hello **Tool Call:**\n```json\n{\"name\":\"unknown\",\"arguments\":{}}\n```";
    let mut parser = RwkvToolParser::create(&test_tools()).unwrap();
    let output = parser.parse_complete(text).unwrap();
    assert_eq!(output.normal_text(), text);
    assert!(output.calls().is_empty());
}

#[test]
fn rwkv_parser_ignores_unclosed_fence_after_complete_json_payload() {
    use super::test_utils::{collect_stream, split_by_chars, test_tools};

    let complete = "**Tool Call:**\n```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n```";
    let unclosed = "**Tool Call:**\n```json\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}\n``";

    let mut single = RwkvToolParser::create(&test_tools()).unwrap();
    let single_out = collect_stream(single.as_mut(), &split_by_chars(unclosed, 1));
    assert!(single_out.calls().is_empty());
    assert_eq!(single_out.normal_text(), unclosed);

    let joined = format!("{complete}\n{unclosed}");
    let mut after_valid = RwkvToolParser::create(&test_tools()).unwrap();
    let after_valid_out = collect_stream(after_valid.as_mut(), &split_by_chars(&joined, 1));
    assert_eq!(after_valid_out.calls().len(), 1);
    assert_eq!(after_valid_out.calls()[0].arguments, "{\"city\":\"Paris\"}");
    assert!(after_valid_out.normal_text().is_empty());
}

#[test]
fn rwkv_visible_grammar_preserves_native_turn_stops_and_parallel_flag() {
    use expect_test::expect;
    use xgrammar_structural_tag::ToolChoice;

    use super::test_utils::test_tools;
    use crate::output_grammar::test_utils::outline;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};

    let tools = &test_tools()[..1];
    let parser = RwkvToolParser::create(tools).unwrap();

    let required_choice = ToolChoice::required();
    let required = parser
        .build_visible_format(&OutputGrammarContext {
            tools,
            tool_choice: &required_choice,
            tool_strict_level: ToolStrictLevel::Parameter,
            parallel_tool_calls: true,
        })
        .unwrap()
        .unwrap();
    expect![[r#"
        sequence
          text excluding [`**Tool Call:**`, `✿`, `\nUser:`, `\n### User`]
          tags_with_separator `\n` at_least_one
            tag `**Tool Call:**\n\`\`\`json\n{"name": "get_weather", "arguments": ` json({ location?: string, city?: string, state?: string, unit?: string, date?: string, days?: integer }) `}\n\`\`\``
          optional
            or
              `✿`
              `\nUser:`
              `\n### User`
    "#]]
    .assert_eq(&outline(&required));

    let non_parallel = parser
        .build_visible_format(&OutputGrammarContext {
            tools,
            tool_choice: &required_choice,
            tool_strict_level: ToolStrictLevel::Parameter,
            parallel_tool_calls: false,
        })
        .unwrap()
        .unwrap();
    assert!(
        outline(&non_parallel).contains("tags_with_separator `\\n` at_least_one stop_after_first")
    );
}

struct DefaultParser;

impl ToolParser for DefaultParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
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
fn tool_parser_does_not_preserve_special_tokens_by_default() {
    let parser = DefaultParser;

    assert!(!parser.preserve_special_tokens());
}

#[test]
fn tool_parser_output_coalesces_adjacent_text_events() {
    let mut output = ToolParserOutput::default();
    output.push_text("hello");
    output.push_text(" ");
    output.push_text("world");
    output.push_call(ToolCallDelta {
        tool_index: 0,
        name: Some("lookup".to_string()),
        arguments: "{}".to_string(),
    });
    output.push_text("!");

    assert_eq!(
        output.events,
        vec![
            ToolParserEvent::Text("hello world".to_string()),
            ToolParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("lookup".to_string()),
                arguments: "{}".to_string(),
            }),
            ToolParserEvent::Text("!".to_string()),
        ]
    );
}

#[test]
fn tool_parser_output_append_coalesces_adjacent_text_events() {
    let mut output = ToolParserOutput::default();
    output.push_text("hello");

    let mut other = ToolParserOutput::default();
    other.push_text(" ");
    other.push_text("world");
    output.append(other);

    let mut after_call = ToolParserOutput::default();
    after_call.push_call(ToolCallDelta {
        tool_index: 0,
        name: Some("lookup".to_string()),
        arguments: "{}".to_string(),
    });
    after_call.push_text("!");
    output.append(after_call);

    assert_eq!(
        output.events,
        vec![
            ToolParserEvent::Text("hello world".to_string()),
            ToolParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("lookup".to_string()),
                arguments: "{}".to_string(),
            }),
            ToolParserEvent::Text("!".to_string()),
        ]
    );
}

#[test]
fn default_parse_complete_delegates_through_parse_chunk_and_finish() {
    struct StreamingParser;

    impl ToolParser for StreamingParser {
        fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            output.push_text("prefix ");
            for call in [
                ToolCallDelta {
                    tool_index: 0,
                    name: Some("weather".to_string()),
                    arguments: "{\"location\":".to_string(),
                },
                ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: "\"Paris\"".to_string(),
                },
                ToolCallDelta {
                    tool_index: 1,
                    name: Some("time".to_string()),
                    arguments: "{\"timezone\":".to_string(),
                },
            ] {
                output.push_call(call);
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            let mut output = ToolParserOutput::default();
            output.push_text("suffix");
            output.push_call(ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "}".to_string(),
            });
            output.push_call(ToolCallDelta {
                tool_index: 1,
                name: None,
                arguments: "\"UTC\"}".to_string(),
            });
            Ok(output)
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    let mut parser = StreamingParser;
    let output = parser.parse_complete("ignored").unwrap();
    assert_eq!(output.normal_text(), "prefix suffix");
    assert_eq!(
        output.calls().into_iter().cloned().collect::<Vec<_>>(),
        vec![
            ToolCallDelta {
                tool_index: 0,
                name: Some("weather".to_string()),
                arguments: "{\"location\":\"Paris\"}".to_string(),
            },
            ToolCallDelta {
                tool_index: 1,
                name: Some("time".to_string()),
                arguments: "{\"timezone\":\"UTC\"}".to_string(),
            },
        ]
    );
}
