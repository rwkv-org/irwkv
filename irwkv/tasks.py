# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal, get_args

GenerationTask = Literal["generate", "transcription", "realtime"]
GENERATION_TASKS: tuple[GenerationTask, ...] = get_args(GenerationTask)

FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = get_args(FrontendTask)

SupportedTask = Literal[GenerationTask, FrontendTask]
FALLBACK_SUPPORTED_TASKS: tuple[SupportedTask, ...] = ("generate",)
