# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from typing_extensions import TypedDict


class ServerConfig(TypedDict, total=False):
    model: str
    arguments: list[str]
    system_prompt: str | None
    supports_parallel: bool | None
    supports_rocm: bool | None
    extended: bool | None  # tests do not run in CI automatically


def patch_system_prompt(
    messages: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    new_messages = deepcopy(messages)
    if new_messages[0]["role"] == "system":
        new_messages[0]["content"] = system_prompt
    else:
        new_messages.insert(0, {"role": "system", "content": system_prompt})
    return new_messages


def ensure_system_prompt(
    messages: list[dict[str, Any]], config: ServerConfig
) -> list[dict[str, Any]]:
    prompt = config.get("system_prompt")
    if prompt:
        return patch_system_prompt(messages, prompt)
    else:
        return messages


# universal args for all models go here. also good if you need to test locally
# and change type or KV cache quantization or something.
SEED = 42

ARGS: list[str] = [
    "--enable-auto-tool-choice",
    "--max-model-len",
    "1024",
    "--max-num-seqs",
    "256",
]

CONFIGS: dict[str, ServerConfig] = {
    "rwkv": {
        "model": "rwkv-rs/rwkv7-g1-st",
        "arguments": [
            "--enforce-eager",
            "--tool-call-parser",
            "rwkv",
            "--reasoning-parser",
            "rwkv",
        ],
        "supports_parallel": True,
    },
}

WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, "
                    "e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "must the two-letter abbreviation for the state "
                    "that the city is in, e.g. 'CA' which would "
                    "mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
        },
    },
}

SEARCH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet and get a summary of the top "
        "10 webpages. Should only be used if you don't know "
        "the answer to a user query, and the results are likely"
        "to be able to be found with a web search",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The term to use in the search. This should"
                    "ideally be keywords to search for, not a"
                    "natural-language question",
                }
            },
            "required": ["search_term"],
        },
    },
}

MESSAGES_WITHOUT_TOOLS: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Hi! How are you?"},
    {"role": "assistant", "content": "I'm doing great! How can I assist you?"},
    {"role": "user", "content": "Can you tell me a joke please?"},
]

MESSAGES_ASKING_FOR_TOOLS: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"}
]

MESSAGES_WITH_TOOL_RESPONSE: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL["function"]["name"],
                    "arguments": '{"city": "Dallas", "state": "TX", '
                    '"unit": "fahrenheit"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
        "content": "The weather in Dallas is 98 degrees fahrenheit, with partly"
        "cloudy skies and a low chance of rain.",
    },
]

MESSAGES_ASKING_FOR_PARALLEL_TOOLS: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "What is the weather in Dallas, Texas and Orlando, Florida in "
        "Fahrenheit?",
    }
]

MESSAGES_WITH_PARALLEL_TOOL_RESPONSE: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "What is the weather in Dallas, Texas and Orlando, Florida in "
        "Fahrenheit?",
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL["function"]["name"],
                    "arguments": '{"city": "Dallas", "state": "TX", '
                    '"unit": "fahrenheit"}',
                },
            },
            {
                "id": "chatcmpl-tool-d027061e1bd21cda48bee7da829c1f5b",
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL["function"]["name"],
                    "arguments": '{"city": "Orlando", "state": "Fl", '
                    '"unit": "fahrenheit"}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
        "content": "The weather in Dallas TX is 98 degrees fahrenheit with mostly "
        "cloudy skies and a chance of rain in the evening.",
    },
    {
        "role": "tool",
        "tool_call_id": "chatcmpl-tool-d027061e1bd21cda48bee7da829c1f5b",
        "content": "The weather in Orlando FL is 78 degrees fahrenheit with clear"
        "skies.",
    },
]
