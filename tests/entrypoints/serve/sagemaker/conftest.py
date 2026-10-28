# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared fixtures and utilities for SageMaker tests."""

# Model name constants used across tests
MODEL_NAME_SMOLLM = "HuggingFaceTB/SmolLM2-135M-Instruct"

# SageMaker header constants
HEADER_SAGEMAKER_CLOSED_SESSION_ID = "X-Amzn-SageMaker-Closed-Session-Id"
HEADER_SAGEMAKER_SESSION_ID = "X-Amzn-SageMaker-Session-Id"
HEADER_SAGEMAKER_NEW_SESSION_ID = "X-Amzn-SageMaker-New-Session-Id"
