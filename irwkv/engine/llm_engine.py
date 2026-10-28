# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.v1.engine.llm_engine import LLMEngine as V1LLMEngine

LLMEngine = V1LLMEngine  # type: ignore
"""The `LLMEngine` class is an alias of [irwkv.v1.engine.llm_engine.LLMEngine][]."""
