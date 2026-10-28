# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from irwkv.entrypoints.openai.responses.context import (
    SimpleContext,
)
from irwkv.outputs import CompletionOutput, RequestOutput

# ==================== SimpleContext Tests ====================


def create_simple_context_output(
    text="",
    token_ids=None,
    prompt="Test prompt",
    prompt_token_ids=None,
    num_cached_tokens=0,
    num_cache_creation_tokens=None,
    logprobs=None,
    finished=True,
):
    """Helper to create a RequestOutput with customizable text for
    SimpleContext tests."""
    if token_ids is None:
        token_ids = []
    return RequestOutput(
        request_id="test-id",
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=logprobs,
                finish_reason=None,
                stop_reason=None,
            )
        ],
        finished=finished,
        num_cached_tokens=num_cached_tokens,
        num_cache_creation_tokens=num_cache_creation_tokens,
    )


def test_simple_context_output_messages_empty():
    """output_messages should be empty before any output is appended."""
    context = SimpleContext()
    assert context.output_messages == []


def test_simple_context_output_messages_single_call():
    """Non-streaming: single append_output produces a single output message."""
    context = SimpleContext()
    output = create_simple_context_output(
        text="Hello world",
        token_ids=[10, 20, 30],
        prompt_token_ids=[1, 2, 3],
    )
    context.append_output(output)

    messages = context.output_messages
    assert len(messages) == 1
    assert messages[0].message == "Hello world"
    assert messages[0].tokens == [10, 20, 30]
    assert messages[0].type == "raw_message_tokens"


def test_simple_context_output_messages_streaming_consolidation():
    """Streaming: multiple append_output calls consolidate into one message."""
    context = SimpleContext()

    # Simulate 3 streaming deltas
    context.append_output(
        create_simple_context_output(
            text="Hello",
            token_ids=[10],
            prompt_token_ids=[1, 2, 3],
        )
    )
    context.append_output(
        create_simple_context_output(
            text=" world",
            token_ids=[20],
            prompt_token_ids=[1, 2, 3],
        )
    )
    context.append_output(
        create_simple_context_output(
            text="!",
            token_ids=[30],
            prompt_token_ids=[1, 2, 3],
        )
    )

    messages = context.output_messages
    assert len(messages) == 1
    assert messages[0].message == "Hello world!"
    assert messages[0].tokens == [10, 20, 30]


def test_simple_context_output_messages_many_deltas():
    """Streaming with many small deltas still produces a single message."""
    context = SimpleContext()

    words = ["The", " quick", " brown", " fox", " jumps"]
    for i, word in enumerate(words):
        context.append_output(
            create_simple_context_output(
                text=word,
                token_ids=[100 + i],
                prompt_token_ids=[1, 2],
            )
        )

    messages = context.output_messages
    assert len(messages) == 1
    assert messages[0].message == "The quick brown fox jumps"
    assert messages[0].tokens == [100, 101, 102, 103, 104]


def test_simple_context_input_messages():
    """input_messages is populated on the first append_output call."""
    context = SimpleContext()
    assert context.input_messages == []

    context.append_output(
        create_simple_context_output(
            text="Hi",
            token_ids=[10],
            prompt="My prompt text",
            prompt_token_ids=[1, 2, 3],
        )
    )

    assert len(context.input_messages) == 1
    assert context.input_messages[0].message == "My prompt text"
    assert context.input_messages[0].tokens == [1, 2, 3]

    # Second call should not add another input message
    context.append_output(
        create_simple_context_output(
            text=" there",
            token_ids=[20],
            prompt="My prompt text",
            prompt_token_ids=[1, 2, 3],
        )
    )

    assert len(context.input_messages) == 1


def test_simple_context_token_counting():
    """Token counting accumulates across streaming deltas."""
    context = SimpleContext()

    context.append_output(
        create_simple_context_output(
            text="a",
            token_ids=[10, 11],
            prompt_token_ids=[1, 2, 3, 4, 5],
            num_cached_tokens=2,
            num_cache_creation_tokens=3,
        )
    )
    context.append_output(
        create_simple_context_output(
            text="b",
            token_ids=[12],
            prompt_token_ids=[1, 2, 3, 4, 5],
            num_cached_tokens=2,
            num_cache_creation_tokens=3,
        )
    )

    assert context.num_prompt_tokens == 5
    assert context.num_output_tokens == 3  # 2 + 1
    assert context.num_cached_tokens == 2
    assert context.num_cache_creation_tokens == 3


def test_simple_context_final_output():
    """final_output reconstructs accumulated text and token_ids."""
    context = SimpleContext()

    context.append_output(
        create_simple_context_output(
            text="foo",
            token_ids=[1, 2],
            prompt_token_ids=[10],
        )
    )
    context.append_output(
        create_simple_context_output(
            text="bar",
            token_ids=[3],
            prompt_token_ids=[10],
        )
    )

    final = context.final_output
    assert final is not None
    assert final.outputs[0].text == "foobar"
    assert final.outputs[0].token_ids == (1, 2, 3)


def test_simple_context_output_messages_empty_text_with_tokens():
    """output_messages should be returned when tokens exist even if text is
    empty (e.g. special tokens)."""
    context = SimpleContext()
    context.append_output(
        create_simple_context_output(
            text="",
            token_ids=[99],
            prompt_token_ids=[1],
        )
    )

    messages = context.output_messages
    assert len(messages) == 1
    assert messages[0].message == ""
    assert messages[0].tokens == [99]


def test_simple_context_output_messages_no_mutation():
    """Each call to output_messages returns a fresh list; callers can't
    corrupt internal state."""
    context = SimpleContext()
    context.append_output(
        create_simple_context_output(
            text="hello",
            token_ids=[1],
            prompt_token_ids=[10],
        )
    )

    msgs1 = context.output_messages
    msgs2 = context.output_messages
    assert msgs1 is not msgs2
    assert msgs1[0].message == msgs2[0].message

    # Appending more output updates the property
    context.append_output(
        create_simple_context_output(
            text=" world",
            token_ids=[2],
            prompt_token_ids=[10],
        )
    )

    msgs3 = context.output_messages
    assert len(msgs3) == 1
    assert msgs3[0].message == "hello world"
    assert msgs3[0].tokens == [1, 2]
