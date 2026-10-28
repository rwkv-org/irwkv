// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;

use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject};
use crate::state::AppState;

// Frontend marker; Python uses "irwkv".
const OWNED_BY: &str = "irwkv-frontend-rs";

/// List the base models served by this frontend.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let max_model_len = state.chat.engine_core_client().max_model_len();
    let model_path = state.model_path().map(str::to_string);

    let base_cards = state.served_model_names().iter().map(|name| ModelObject {
        id: name.clone(),
        object: "model".to_string(),
        created,
        owned_by: OWNED_BY.to_string(),
        root: Some(model_path.clone().unwrap_or_else(|| name.clone())),
        parent: None,
        max_model_len: Some(max_model_len),
    });

    Json(ListModelsResponse {
        object: "list".to_string(),
        data: base_cards.collect(),
    })
}
