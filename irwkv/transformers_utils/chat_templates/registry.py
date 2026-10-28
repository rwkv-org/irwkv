# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from irwkv.logger import init_logger

logger = init_logger(__file__)

ChatTemplatePath: TypeAlias = Path | Callable[[str], Path | None]

_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {}


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.",
            model_type,
            chat_template,
        )

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = chat_template


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Path | None:
    chat_template = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template
