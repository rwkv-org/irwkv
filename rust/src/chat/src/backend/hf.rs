// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use irwkv_text::backend::hf::{
    HfTextBackend, ResolvedModelFiles, RwkvSamplingProfile, load_model_config,
    load_rwkv_sampling_profile, load_tokenizer_config,
};
use irwkv_text::tokenizer::DynTokenizer;
use irwkv_text::{DynTextBackend, TextBackend as _};
use thiserror_ext::AsReport as _;
use tracing::{info, warn};

use crate::DynChatOutputProcessor;
use crate::backend::{
    ChatBackend, DynChatBackend, LoadModelBackendsOptions, LoadedModelBackends,
    NewChatOutputProcessorOptions,
};
use crate::error::Result;
use crate::multimodal::{MultimodalConfigFiles, MultimodalModelInfo};
use crate::output::DefaultChatOutputProcessor;
use crate::parser::unified::{HfTemplateError, ResponseTemplate};
use crate::renderer::DynChatRenderer;
use crate::renderer::hf::{HfChatRenderer, MultimodalRenderInfo};
use crate::request::ChatRequest;

/// [`ChatBackend`] implementation built on Hugging Face model files.
pub struct HfChatBackend {
    model_id: String,
    rwkv_sampling_profiles: Option<[RwkvSamplingProfile; 3]>,
    tokenizer: DynTokenizer,
    chat_renderer: DynChatRenderer,
    multimodal_model_info: Option<MultimodalModelInfo>,
    /// `response_template` from `tokenizer_config.json` for the `hf` parser, or
    /// why it is unavailable.
    response_template: std::result::Result<Arc<ResponseTemplate>, HfTemplateError>,
}

impl HfChatBackend {
    /// Load the chat backend from resolved Hugging Face model files.
    pub fn from_resolved_model_files(
        files: ResolvedModelFiles,
        model_id: String,
        options: LoadModelBackendsOptions,
        tokenizer: DynTokenizer,
    ) -> Result<Self> {
        let model_config = load_model_config(files.config_path.as_deref())?;
        let model_type = model_config.model_type().unwrap_or_default();
        let rwkv_sampling_profiles = if model_type == "rwkv"
            && options.generation_config == irwkv_text::GenerationConfigMode::Auto
        {
            Some([
                load_rwkv_sampling_profile(files.generation_config_path.as_deref())?,
                load_rwkv_sampling_profile(files.fake_think_generation_config_path.as_deref())?,
                load_rwkv_sampling_profile(files.tools_generation_config_path.as_deref())?,
            ])
        } else {
            None
        };
        let multimodal_model_info = if options.language_model_only {
            None
        } else {
            MultimodalModelInfo::from_paths(
                model_id.clone(),
                (!model_type.is_empty()).then_some(model_type.to_string()),
                MultimodalConfigFiles {
                    config: files.config_path.as_deref(),
                    preprocessor_config: files.preprocessor_config_path.as_deref(),
                    video_preprocessor_config: files.video_preprocessor_config_path.as_deref(),
                    processor_config: files.processor_config_path.as_deref(),
                },
                tokenizer.clone(),
                options.limit_mm_per_prompt.clone(),
            )?
        };
        let multimodal_render_info = resolve_multimodal_render_info(multimodal_model_info.as_ref());
        let response_template = load_response_template(&files)?;

        let renderer = options.renderer.resolve(model_type);
        let chat_renderer: DynChatRenderer = Arc::new(HfChatRenderer::load(
            &files,
            options,
            multimodal_render_info,
        )?);

        info!(
            model_id,
            model_type,
            %renderer,
            "loaded chat backend with Hugging Face model files"
        );

        Ok(Self {
            model_id,
            rwkv_sampling_profiles,
            tokenizer,
            chat_renderer,
            multimodal_model_info,
            response_template,
        })
    }
}

/// Load and compile the checkpoint's `response_template`, if any.
///
/// A missing or unusable template is kept as an error, so that only requests
/// selecting the `hf` parser fail.
fn load_response_template(
    files: &ResolvedModelFiles,
) -> Result<std::result::Result<Arc<ResponseTemplate>, HfTemplateError>> {
    let Some(template) =
        load_tokenizer_config(files.tokenizer_config_path.as_deref())?.response_template
    else {
        return Ok(Err(HfTemplateError::Missing));
    };
    Ok(match ResponseTemplate::from_json(&template) {
        Ok(template) => {
            info!("loaded response_template for the `hf` parser");
            Ok(Arc::new(template))
        }
        Err(error) => {
            warn!(error = %error.as_report(), "the `hf` parser cannot execute the model's response_template");
            Err(error)
        }
    })
}

impl ChatBackend for HfChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        self.chat_renderer.clone()
    }

    fn apply_sampling_defaults(
        &self,
        request: &mut ChatRequest,
        template_kwargs: &std::collections::HashMap<String, serde_json::Value>,
    ) -> bool {
        let Some(profiles) = &self.rwkv_sampling_profiles else {
            return false;
        };
        let profile = if !request.tool_context.initial_tools.is_empty() {
            &profiles[2]
        } else if template_kwargs.get("fake_cot_mode").and_then(|v| v.as_bool()) == Some(true) {
            &profiles[1]
        } else {
            &profiles[0]
        };
        request.decode_options.add_default_stop_strings(&profile.stop_strings);
        let profile = &profile.params;
        let params = &mut request.sampling_params;
        params.temperature = params.temperature.or(profile.temperature);
        params.top_p = params.top_p.or(profile.top_p);
        params.top_k = params.top_k.or(profile.top_k);
        params.frequency_penalty = params.frequency_penalty.or(profile.frequency_penalty);
        params.presence_penalty = params.presence_penalty.or(profile.presence_penalty);
        params.penalty_decay = params.penalty_decay.or(profile.penalty_decay);
        true
    }

    fn multimodal_model_info(&self) -> Option<&MultimodalModelInfo> {
        self.multimodal_model_info.as_ref()
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor> {
        Ok(Box::new(
            DefaultChatOutputProcessor::with_response_template(
                request,
                &self.model_id,
                self.tokenizer.clone(),
                &self.response_template,
                options.tool_call_parser,
                options.reasoning_parser,
                options.tool_strict_level,
            )?,
        ))
    }
}

/// Load the Hugging Face text and chat backends for the given model id.
pub(super) async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    let mut files = ResolvedModelFiles::new(model_id, options.revision.as_deref()).await?;
    files.apply_overrides(&options.hf_overrides)?;
    let text_backend = HfTextBackend::from_resolved_model_files(
        files.clone(),
        model_id.to_string(),
        options.generation_config,
    )?;
    let tokenizer = text_backend.tokenizer();
    let text_backend: DynTextBackend = Arc::new(text_backend);

    let chat_backend: DynChatBackend = Arc::new(HfChatBackend::from_resolved_model_files(
        files,
        model_id.to_string(),
        options,
        tokenizer,
    )?);

    Ok(LoadedModelBackends {
        text_backend,
        chat_backend,
    })
}

fn resolve_multimodal_render_info(
    info: Option<&MultimodalModelInfo>,
) -> Option<MultimodalRenderInfo> {
    use llm_multimodal::Modality;

    info.map(|info| MultimodalRenderInfo {
        image_token: info.placeholder_token(Modality::Image).map(str::to_string),
        video_token: info.placeholder_token(Modality::Video).map(str::to_string),
        audio_token: info.placeholder_token(Modality::Audio).map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use irwkv_text::tokenizer::DynTokenizer;
    use irwkv_tokenizer::test_utils::TestTokenizer;
    use tempfile::tempdir;

    use super::HfChatBackend;
    use crate::RendererSelection;
    use crate::backend::{ChatBackend, LoadModelBackendsOptions};
    use crate::request::{ChatContent, ChatMessage, ChatRequest};

    fn request_with_user_text(text: &str) -> ChatRequest {
        ChatRequest {
            request_id: "renderer-selection-test".to_string(),
            messages: vec![ChatMessage::User {
                content: ChatContent::Text(text.to_string()),
            }],
            ..ChatRequest::for_test()
        }
    }

    fn write_json(path: &std::path::Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    fn resolved_files(
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> irwkv_text::backend::hf::ResolvedModelFiles {
        let dir = tempdir().unwrap();
        let root = dir.keep();
        let config_path = root.join("config.json");
        let tokenizer_config_path = root.join("tokenizer_config.json");
        write_json(&config_path, config_json);
        write_json(&tokenizer_config_path, tokenizer_config_json);

        write_json(&root.join("tokenizer.json"), "{}");
        futures::executor::block_on(irwkv_text::backend::hf::ResolvedModelFiles::new(
            root.to_str().unwrap(),
            None,
        ))
        .unwrap()
    }

    fn test_tokenizer() -> DynTokenizer {
        Arc::new(TestTokenizer::new())
    }

    fn backend_for_selection(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> HfChatBackend {
        HfChatBackend::from_resolved_model_files(
            resolved_files(config_json, tokenizer_config_json),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                revision: None,
                hf_overrides: Default::default(),
                generation_config: Default::default(),
                renderer,
                language_model_only: false,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                limit_mm_per_prompt: HashMap::new(),
            },
            test_tokenizer(),
        )
        .unwrap()
    }

    fn render_prompt(
        renderer: RendererSelection,
        config_json: &str,
        tokenizer_config_json: &str,
    ) -> String {
        backend_for_selection(renderer, config_json, tokenizer_config_json)
            .chat_renderer()
            .render(&request_with_user_text("hello"))
            .unwrap()
            .prompt
            .into_text()
            .expect("renderer should return text prompt")
    }

    #[test]
    fn rwkv_chat_selects_profile_and_preserves_request_overrides() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        write_json(&root.join("config.json"), r#"{"model_type":"rwkv"}"#);
        write_json(&root.join("tokenizer.json"), "{}");
        write_json(
            &root.join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages }}"}"#,
        );
        for (name, temperature, decay) in [
            ("generation_config.json", 0.96, 0.988),
            ("fake_think_generation_config.json", 1.0, 0.996),
            ("tools_generation_config.json", 0.8, 0.997),
        ] {
            write_json(
                &root.join(name),
                &format!(
                    r#"{{"temperature":{temperature},"top_p":0.76,"top_k":32,"presence_penalty":1.0,"frequency_penalty":0.1,"penalty_decay":{decay},"stop_strings":["{name}"]}}"#
                ),
            );
        }
        let files = futures::executor::block_on(irwkv_text::backend::hf::ResolvedModelFiles::new(
            root.to_str().unwrap(),
            None,
        ))
        .unwrap();
        let backend = HfChatBackend::from_resolved_model_files(
            files,
            "rwkv".into(),
            LoadModelBackendsOptions {
                language_model_only: true,
                ..Default::default()
            },
            test_tokenizer(),
        )
        .unwrap();
        let mut request = request_with_user_text("hello");
        request.sampling_params.temperature = Some(0.2);
        request.decode_options.stop_strings = Some(vec!["user".into()]);
        let kwargs = HashMap::from([("fake_cot_mode".to_string(), serde_json::json!(true))]);
        backend.apply_sampling_defaults(&mut request, &kwargs);
        assert_eq!(request.sampling_params.temperature, Some(0.2));
        assert_eq!(request.sampling_params.penalty_decay, Some(0.996));
        assert_eq!(
            request.decode_options.stop_strings,
            Some(vec![
                "user".into(),
                "fake_think_generation_config.json".into()
            ])
        );
        request.tool_context.initial_tools.push(crate::request::ChatTool {
            name: "lookup".into(),
            description: None,
            parameters: serde_json::json!({}),
            strict: None,
            defer_loading: None,
        });
        request.sampling_params = Default::default();
        request.decode_options.stop_strings = None;
        backend.apply_sampling_defaults(&mut request, &kwargs);
        assert_eq!(request.sampling_params.temperature, Some(0.8));
        assert_eq!(request.sampling_params.penalty_decay, Some(0.997));
        assert_eq!(
            request.decode_options.stop_strings,
            Some(vec!["tools_generation_config.json".into()])
        );
    }

    #[test]
    fn auto_uses_hf_renderer_for_other_model_types() {
        let prompt = render_prompt(
            RendererSelection::Auto,
            r#"{"model_type":"qwen2"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );

        assert_eq!(prompt, "hello");
    }

    #[test]
    fn language_model_only_skips_multimodal_preprocessor_config() {
        let mut files = resolved_files(
            r#"{"model_type":"deepseek_v0_vl"}"#,
            r#"{"chat_template":"{{ messages[0].content }}"}"#,
        );
        let preprocessor_config_path = files
            .config_path
            .as_ref()
            .unwrap()
            .parent()
            .unwrap()
            .join("preprocessor_config.json");
        write_json(&preprocessor_config_path, r#"{"size":[672,672]}"#);
        files.preprocessor_config_path = Some(preprocessor_config_path.clone());

        let backend = HfChatBackend::from_resolved_model_files(
            files.clone(),
            "test-model".to_string(),
            LoadModelBackendsOptions {
                language_model_only: true,
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                ..Default::default()
            },
            test_tokenizer(),
        )
        .unwrap();

        assert!(backend.multimodal_model_info().is_none());

        let invalid_preprocessor_config = r#"{"size":[672,672]"#;
        write_json(&preprocessor_config_path, invalid_preprocessor_config);

        let error = HfChatBackend::from_resolved_model_files(
            files,
            "test-model".to_string(),
            LoadModelBackendsOptions {
                chat_template_content_format: Default::default(),
                chat_template: None,
                default_chat_template_kwargs: HashMap::new(),
                ..Default::default()
            },
            test_tokenizer(),
        )
        .err()
        .expect("invalid preprocessor config should fail without language_model_only");
        assert!(error.to_string().contains("failed to parse preprocessor_config.json"));
    }
}
