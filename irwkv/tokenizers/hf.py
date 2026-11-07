# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility names for tokenizer artifacts in Hugging Face format."""

from pathlib import Path
from typing import TypeAlias, TypeVar

from .protocol import TokenizerLike
from .rwkv import RwkvTokenizer

HfTokenizer: TypeAlias = RwkvTokenizer
ThreadSafeHFTokenizerMixin = RwkvTokenizer
_T = TypeVar("_T", bound=TokenizerLike)


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1) -> _T:
    # Native encoding does not mutate shared truncation or padding settings.
    return tokenizer


def get_cached_tokenizer(tokenizer: _T) -> _T:
    return tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, path_or_repo_id: str | Path, *args, **kwargs):
        return RwkvTokenizer.from_pretrained(path_or_repo_id, *args, **kwargs)
