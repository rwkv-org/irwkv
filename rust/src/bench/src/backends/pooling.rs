// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI embeddings backends: non-streaming HTTP POST to `/v1/embeddings`.
//!
//! Supported variants:
//! - `openai-embeddings`: Standard OpenAI `/v1/embeddings` with text input
//! - `openai-embeddings-chat`: OpenAI `/v1/embeddings` with chat message format (supports
//!   multimodal)

use std::time::Instant;

use crate::backends::{RequestFuncInput, RequestFuncOutput, build_headers};
use crate::cli::BackendKind;
use crate::error::Result;

/// Response from the embeddings API, with the usage fields used by the benchmark.
#[derive(serde::Deserialize)]
struct PoolingResponse {
    usage: Option<PoolingUsage>,
}

#[derive(serde::Deserialize)]
struct PoolingUsage {
    prompt_tokens: Option<u64>,
}

#[derive(Clone)]
pub struct PoolingBackend {
    pub kind: BackendKind,
}

impl PoolingBackend {
    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        let payload = self.build_payload(input);

        // Preserve client-side prompt_len as fallback if server doesn't report usage.
        let mut output = RequestFuncOutput {
            prompt_len: input.prompt_len,
            num_input_sequences: match self.kind {
                BackendKind::OpenaiEmbeddings => {
                    input.prompt_list.as_ref().map_or(1, |list| list.len())
                }
                BackendKind::OpenaiEmbeddingsChat => 1,
                _ => unreachable!("PoolingBackend with non-pooling kind"),
            },
            ..Default::default()
        };

        let headers = build_headers(
            Some("application/json"),
            &input.extra_headers,
            &input.request_id,
        );

        let mut request = client.post(&input.api_url);
        for (k, v) in &headers {
            request = request.header(k, v);
        }

        let st = Instant::now();

        let response = match request.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                output.error = format!("Request failed: {e}");
                return Ok(output);
            }
        };

        if response.status().is_success() {
            let latency = st.elapsed().as_secs_f64();
            output.latency = latency;
            output.ttft = latency;
            output.success = true;

            // Parse usage from response; keep client-side prompt_len as fallback.
            match response.json::<PoolingResponse>().await {
                Ok(data) => {
                    if let Some(usage) = data.usage
                        && let Some(tokens) = usage.prompt_tokens
                    {
                        output.prompt_len = tokens as usize;
                    }
                }
                Err(_) => {
                    // Response parsed but no usage — keep client-side prompt_len
                }
            }
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            output.error = format!("HTTP {status}: {body}");
        }

        Ok(output)
    }

    fn build_payload(&self, input: &RequestFuncInput) -> serde_json::Value {
        let model = input.model_name.as_deref().unwrap_or(&input.model);

        // A batched request
        // (--random-batch-size) sends the text list; otherwise prefer prompt_token_ids
        // when available. The random dataset sets prompt="" and relies on token IDs;
        // the OpenAI embeddings API accepts both text strings and token ID arrays.
        // Note: embeddings-chat uses text in messages.
        let input_value = if let Some(ref list) = input.prompt_list {
            serde_json::json!(list.iter().map(|s| s.as_ref()).collect::<Vec<&str>>())
        } else if let Some(ref token_ids) = input.prompt_token_ids {
            serde_json::json!(token_ids.as_ref())
        } else {
            serde_json::json!(input.prompt.as_ref())
        };

        let mut payload = match self.kind {
            BackendKind::OpenaiEmbeddings => {
                serde_json::json!({
                    "model": model,
                    "input": input_value,
                })
            }
            BackendKind::OpenaiEmbeddingsChat => {
                // Chat format: uses text prompt in messages array (for multimodal support).
                // Python's _get_chat_content always returns a content array.
                // Use raw string concatenation for multimodal fragments (zero-copy,
                // avoids re-parsing ~200KB+ base64 per image).
                let content_json = build_chat_content_json(input);

                serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": content_json}],
                })
            }
            _ => unreachable!("PoolingBackend with non-pooling kind"),
        };

        // Merge extra_body fields into payload
        if let Some(ref extra) = input.extra_body
            && let (Some(base), Some(extra_obj)) = (payload.as_object_mut(), extra.as_object())
        {
            for (k, v) in extra_obj {
                base.insert(k.clone(), v.clone());
            }
        }

        payload
    }
}

/// Build the chat content JSON array for embeddings-chat.
/// Uses raw string concatenation for multimodal fragments to avoid
/// re-parsing large base64 image data (matching openai_chat.rs approach).
fn build_chat_content_json(input: &RequestFuncInput) -> serde_json::Value {
    if input.multi_modal_content.is_none() {
        // Text-only: return content array with single text element
        return serde_json::json!([{
            "type": "text",
            "text": input.prompt.as_ref(),
        }]);
    }

    // Multimodal: build JSON string manually for zero-copy fragment embedding
    let mm = input.multi_modal_content.as_ref().unwrap();
    let prompt = input.prompt.as_ref();

    let mm_total: usize = mm.iter().map(|f| f.len() + 1).sum();
    let mut json = String::with_capacity(64 + prompt.len() * 2 + mm_total);

    // [{"type":"text","text":"<prompt>"}
    json.push_str(r#"[{"type":"text","text":""#);
    push_json_escaped_str(&mut json, prompt);
    json.push_str(r#""}"#);

    // ,<mm fragment 1>,<mm fragment 2>,...
    for fragment in mm.iter() {
        json.push(',');
        json.push_str(fragment);
    }

    json.push(']');

    // Parse the assembled string into a Value for embedding in the payload.
    // This parse is O(n) but operates on the pre-built string once, not per-fragment.
    serde_json::from_str(&json).unwrap_or_else(|_| {
        serde_json::json!([{
            "type": "text",
            "text": input.prompt.as_ref(),
        }])
    })
}

/// Escape a string for safe JSON embedding (matching openai_chat.rs).
fn push_json_escaped_str(buf: &mut String, s: &str) {
    use std::fmt::Write;
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str(r#"\""#),
            '\\' => buf.push_str(r"\\"),
            '\n' => buf.push_str(r"\n"),
            '\r' => buf.push_str(r"\r"),
            '\t' => buf.push_str(r"\t"),
            c if c < '\x20' => {
                let _ = write!(buf, "\\u{:04x}", c as u32);
            }
            c => buf.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn list(items: &[&str]) -> Option<Arc<[Arc<str>]>> {
        Some(items.iter().map(|s| Arc::from(*s)).collect())
    }

    #[test]
    fn test_embeddings_payload_batched_input() {
        let backend = PoolingBackend {
            kind: BackendKind::OpenaiEmbeddings,
        };
        let input = RequestFuncInput {
            model: "bge".to_string(),
            prompt_list: list(&["t1", "t2", "t3"]),
            ..Default::default()
        };
        let payload = backend.build_payload(&input);
        assert_eq!(payload["input"], serde_json::json!(["t1", "t2", "t3"]));
        assert_eq!(payload["model"], "bge");
        // The generic OpenAI embeddings endpoint does not use irwkv-specific fields.
        assert!(payload.get("truncate_prompt_tokens").is_none());
    }
}
