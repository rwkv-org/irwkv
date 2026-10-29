# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example client that summarizes older chat turns to fit a token budget.
NOTE: start a supported chat completion model server with `irwkv serve`, e.g.
    irwkv serve /path/to/rwkv-model --reasoning-parser rwkv

Tokens are counted with the server's /tokenize endpoint. Leading system
messages and the most recent turns are kept verbatim; older turns are replaced
by one model-written summary. Summarization is lossy, so keep the original
history. The summary request uses the default template so the model can think
first; the reasoning parser keeps that reasoning out of the summary. Token
counts use the same template kwargs as the answer request.
"""

import argparse
import json

import requests
from openai import OpenAI

SUMMARY_PROMPT = (
    "Summarize the earlier conversation below for a later continuation. "
    "Preserve exact numbers, names, constraints and unresolved tasks. "
    "Do not answer it or add facts.\n\n"
)


def count_tokens(base_url, model, messages, template_kwargs):
    response = requests.post(
        f"{base_url}/tokenize",
        json={
            "model": model,
            "messages": messages,
            "chat_template_kwargs": template_kwargs,
        },
    )
    response.raise_for_status()
    return response.json()["count"]


def compact(
    client,
    base_url,
    model,
    messages,
    budget,
    keep_last_turns,
    max_tokens,
    template_kwargs,
):
    if count_tokens(base_url, model, messages, template_kwargs) <= budget:
        return messages

    num_system = next(
        (i for i, m in enumerate(messages) if m["role"] != "system"), len(messages)
    )
    user_starts = [
        i for i, m in enumerate(messages) if i >= num_system and m["role"] == "user"
    ]
    if len(user_starts) <= keep_last_turns:
        raise ValueError("The most recent turns alone exceed the budget")
    split = user_starts[-keep_last_turns]
    older = messages[num_system:split]

    transcript = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in older)
    summary = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT + transcript}],
        max_tokens=max_tokens,
        temperature=0,
    ).choices[0]
    if summary.finish_reason != "stop" or not summary.message.content:
        raise ValueError(f"Summary did not finish: {summary.finish_reason}")

    # Merge into the first kept user message so roles still alternate.
    first_kept = dict(messages[split])
    first_kept["content"] = (
        f"Summary of the earlier conversation:\n{summary.message.content.strip()}"
        f"\n\n{first_kept['content']}"
    )
    compacted = messages[:num_system] + [first_kept] + messages[split + 1 :]
    if count_tokens(base_url, model, compacted, template_kwargs) > budget:
        raise ValueError("The compacted history still exceeds the budget")
    return compacted


def parse_args():
    parser = argparse.ArgumentParser(description="Chat client with compaction")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--messages", required=True, help="JSON list of messages")
    parser.add_argument("--budget", type=int, default=8192, help="Input tokens")
    parser.add_argument("--keep-last-turns", type=int, default=2)
    parser.add_argument("--summary-max-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={})
    return parser.parse_args()


def main(args):
    client = OpenAI(api_key="EMPTY", base_url=f"{args.base_url}/v1")
    model = client.models.list().data[0].id

    with open(args.messages) as f:
        messages = json.load(f)
    compacted = compact(
        client,
        args.base_url,
        model,
        messages,
        args.budget,
        args.keep_last_turns,
        args.summary_max_tokens,
        args.chat_template_kwargs,
    )
    before, after = (
        count_tokens(args.base_url, model, m, args.chat_template_kwargs)
        for m in (messages, compacted)
    )
    print(f"Input tokens: {before} -> {after}")

    chat_completion = client.chat.completions.create(
        model=model,
        messages=compacted,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": args.chat_template_kwargs},
    )
    print("-" * 50)
    print("Chat completion results:")
    print(chat_completion.choices[0].message.content)
    print("-" * 50)


if __name__ == "__main__":
    main(parse_args())
