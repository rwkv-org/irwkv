// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use rmpv::Value;
use serde_json::{Value as JsonValue, json};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

async fn operate(state: &AppState, operation: &str, value: Value) -> Result<Value, ApiError> {
    let mut results = state
        .engine_core_client()
        .call_utility::<Value, _>("rwkv_state", (operation, value))
        .await
        .map_err(|error| utility_call_error("rwkv_state", error))?;
    let result = results
        .pop()
        .ok_or_else(|| ApiError::server_error("Missing RWKV state result".to_string()))?;
    if let Some(error) = result.as_map().and_then(|entries| {
        entries.iter().find_map(|(k, v)| (k.as_str() == Some("error")).then_some(v))
    }) {
        return Err(ApiError::invalid_request(
            error.as_str().unwrap_or("Invalid RWKV state").to_string(),
            None,
        ));
    }
    Ok(result)
}

fn as_json(value: Value) -> Result<JsonValue, ApiError> {
    serde_json::to_value(value).map_err(|error| ApiError::server_error(error.to_string()))
}

pub async fn list_states(State(state): State<Arc<AppState>>) -> Result<Json<JsonValue>, ApiError> {
    let entries = operate(&state, "list", Value::Nil).await?;
    Ok(Json(json!({"data": as_json(entries)?})))
}

pub async fn upload_state(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<(StatusCode, Json<JsonValue>), ApiError> {
    let result = operate(&state, "import", Value::Binary(body.to_vec())).await?;
    Ok((StatusCode::CREATED, Json(as_json(result)?)))
}

pub async fn download_state(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    match operate(&state, "export", Value::from(id)).await? {
        Value::Binary(data) => Ok((
            [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"rwkv.state\"",
                ),
            ],
            data,
        )
            .into_response()),
        _ => Err(ApiError::server_error(
            "Invalid binary RWKV state response".to_string(),
        )),
    }
}

pub async fn delete_state(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    operate(&state, "delete", Value::from(id)).await?;
    Ok(StatusCode::NO_CONTENT)
}
