// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

use irwkv_engine_core_client::Error as EngineCoreError;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("generate request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("engine-core error")]
    EngineCoreClient(#[source] EngineCoreError),
    #[error("{message}")]
    RequestValidation { message: String },
    #[error(
        "sampling mask for request `{request_id}` has {row_count} rows for {token_count} generated tokens"
    )]
    SamplingMaskTokenCountMismatch {
        request_id: String,
        token_count: usize,
        row_count: usize,
    },
}

impl From<EngineCoreError> for Error {
    fn from(error: EngineCoreError) -> Self {
        match error {
            EngineCoreError::RequestValidation { message } => Self::RequestValidation { message },
            other => Self::EngineCoreClient(other),
        }
    }
}
