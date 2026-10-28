# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum


class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"
