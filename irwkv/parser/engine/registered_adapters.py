# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from irwkv.parser.engine.adapters import make_adapters
from irwkv.parser.rwkv import RwkvParser

RwkvParserReasoningAdapter, _ = make_adapters(RwkvParser)
