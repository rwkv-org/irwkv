// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::str::FromStr;

use itertools::Itertools;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{EnumIter, IntoEnumIterator};

/// Specify which chat renderer implementation to use.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay, EnumIter,
)]
pub enum RendererSelection {
    /// Use model-based auto-detection.
    #[default]
    Auto,
    /// Force the generic Hugging Face chat-template renderer.
    Hf,
}

impl RendererSelection {
    pub const AUTO_LITERAL: &str = "auto";
    pub const HF_LITERAL: &str = "hf";

    /// Resolve the renderer selection using the given model type string, if
    /// it's `Auto`.
    pub fn resolve(self, _model_type: &str) -> Self {
        match self {
            Self::Auto => Self::Hf,
            selection => selection,
        }
    }
}

impl FromStr for RendererSelection {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::HF_LITERAL) {
            Ok(Self::Hf)
        } else {
            Err(format!(
                "unknown renderer `{value}` (expected one of: {})",
                Self::iter().join(", ")
            ))
        }
    }
}

impl fmt::Display for RendererSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::Hf => f.write_str(Self::HF_LITERAL),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use strum::IntoEnumIterator;

    use super::RendererSelection;

    #[test]
    fn renderer_selection_display_round_trips() {
        for selection in RendererSelection::iter() {
            assert_eq!(
                selection.to_string().parse::<RendererSelection>().unwrap(),
                selection
            );
        }
    }

    #[test]
    fn renderer_selection_expected_error_message() {
        let err = RendererSelection::from_str("unknown").unwrap_err();
        expect_test::expect!["unknown renderer `unknown` (expected one of: auto, hf)"]
            .assert_eq(&err);
    }
}
