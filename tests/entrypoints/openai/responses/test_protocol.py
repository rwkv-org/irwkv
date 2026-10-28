# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from irwkv.entrypoints.openai.responses.protocol import (
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    assert serialize_messages([dict_value]) == [dict_value]
