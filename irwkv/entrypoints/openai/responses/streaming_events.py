# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming SSE event builders for the Responses API.

Pure functions that translate streaming state + delta data into
OpenAI Response API SSE events. Used by the streaming event
processors in serving.py.

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, NamedTuple

from openai.types.responses import (
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallItem,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    response_output_text,
    response_text_delta_event,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool

from irwkv.entrypoints.chat_utils import make_tool_call_id
from irwkv.entrypoints.generate.base.protocol import DeltaMessage, DeltaToolCall
from irwkv.entrypoints.openai.responses.protocol import (
    StreamingResponsesResponse,
)
from irwkv.entrypoints.openai.responses.utils import (
    build_responses_tool_call_name_map,
    resolve_responses_tool_call_name,
)
from irwkv.outputs import CompletionOutput
from irwkv.utils import random_uuid


class _StateType(Enum):
    NONE = auto()
    CONTENT = auto()
    REASONING = auto()
    TOOL_CALL = auto()


@dataclass
class SimpleStreamingState:
    output_index: int = 0
    current_item_id: str = ""
    content_index: int = 0
    accumulated_text: str = ""
    accumulated_logprobs: list[response_output_text.Logprob] = field(
        default_factory=list
    )
    tool_call_id: str = ""
    tool_call_name: str = ""
    tool_call_namespace: str | None = None
    tool_call_index: int | None = None
    current_state: _StateType = field(default_factory=lambda: _StateType.NONE)


def emit_simple_content_open(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.CONTENT
    state.current_item_id = f"msg_{random_uuid()}"
    state.content_index = 0
    state.accumulated_text = ""
    state.accumulated_logprobs = []
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseOutputMessage(
                id=state.current_item_id,
                type="message",
                role="assistant",
                content=[],
                status="in_progress",
            ),
        ),
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            content_index=state.content_index,
            part=ResponseOutputText(
                type="output_text",
                text="",
                annotations=[],
                logprobs=[],
            ),
        ),
    ]


def _to_stream_logprobs(
    logprobs: list[response_output_text.Logprob],
) -> list[response_text_delta_event.Logprob]:
    return [
        response_text_delta_event.Logprob(
            token=lp.token,
            logprob=lp.logprob,
            top_logprobs=[
                response_text_delta_event.LogprobTopLogprob(
                    token=tl.token, logprob=tl.logprob
                )
                for tl in lp.top_logprobs
            ],
        )
        for lp in logprobs
    ]


def emit_simple_content_delta(
    state: SimpleStreamingState,
    delta: str,
    logprobs: list[response_output_text.Logprob] | None = None,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    if logprobs:
        state.accumulated_logprobs.extend(logprobs)
    return [
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=-1,
            content_index=state.content_index,
            output_index=state.output_index,
            item_id=state.current_item_id,
            delta=delta,
            logprobs=_to_stream_logprobs(logprobs or []),
        )
    ]


def emit_simple_content_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    part = ResponseOutputText(
        type="output_text",
        text=state.accumulated_text,
        annotations=[],
        logprobs=state.accumulated_logprobs or None,
    )
    events: list[StreamingResponsesResponse] = [
        ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=-1,
            output_index=state.output_index,
            content_index=state.content_index,
            text=state.accumulated_text,
            logprobs=[],
            item_id=state.current_item_id,
        ),
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=part,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseOutputMessage(
                id=state.current_item_id,
                type="message",
                role="assistant",
                content=[part] if state.accumulated_text else [],
                status="completed",
            ),
        ),
    ]
    state.output_index += 1
    state.current_state = _StateType.NONE
    return events


def emit_simple_reasoning_open(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.REASONING
    state.current_item_id = f"rs_{random_uuid()}"
    state.content_index = 0
    state.accumulated_text = ""
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseReasoningItem(
                type="reasoning",
                id=state.current_item_id,
                summary=[],
                status="in_progress",
            ),
        ),
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            content_index=state.content_index,
            part={"text": "", "type": "reasoning_text"},
        ),
    ]


def emit_simple_reasoning_delta(
    state: SimpleStreamingState,
    delta: str,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    return [
        ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id=state.current_item_id,
            sequence_number=-1,
            output_index=state.output_index,
            content_index=state.content_index,
            delta=delta,
        )
    ]


def emit_simple_reasoning_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    part = ResponseReasoningTextContent(
        text=state.accumulated_text,
        type="reasoning_text",
    )
    events: list[StreamingResponsesResponse] = [
        ResponseReasoningTextDoneEvent(
            type="response.reasoning_text.done",
            item_id=state.current_item_id,
            sequence_number=-1,
            output_index=state.output_index,
            content_index=state.content_index,
            text=state.accumulated_text,
        ),
        ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=state.content_index,
            part=part.model_dump(),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseReasoningItem(
                type="reasoning",
                content=[part],
                status="completed",
                id=state.current_item_id,
                summary=[],
            ),
        ),
    ]
    state.output_index += 1
    state.current_state = _StateType.NONE
    return events


def emit_simple_tool_call_open(
    state: SimpleStreamingState,
    name: str,
    index: int | None,
    namespace: str | None = None,
    call_id: str | None = None,
) -> list[StreamingResponsesResponse]:
    state.current_state = _StateType.TOOL_CALL
    state.current_item_id = f"fc_{random_uuid()}"
    state.tool_call_id = call_id or make_tool_call_id()
    state.tool_call_name = name
    state.tool_call_namespace = namespace
    state.tool_call_index = index
    state.accumulated_text = ""
    return [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseFunctionToolCallItem(
                type="function_call",
                id=state.current_item_id,
                call_id=state.tool_call_id,
                name=name,
                namespace=namespace,
                arguments="",
                status="in_progress",
            ),
        ),
    ]


def emit_simple_tool_call_delta(
    state: SimpleStreamingState,
    delta: str,
) -> list[StreamingResponsesResponse]:
    state.accumulated_text += delta
    return [
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            delta=delta,
        )
    ]


def emit_simple_tool_call_done(
    state: SimpleStreamingState,
) -> list[StreamingResponsesResponse]:
    events: list[StreamingResponsesResponse] = [
        ResponseFunctionCallArgumentsDoneEvent(
            type="response.function_call_arguments.done",
            sequence_number=-1,
            output_index=state.output_index,
            item_id=state.current_item_id,
            arguments=state.accumulated_text,
            name=state.tool_call_name,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=state.output_index,
            item=ResponseFunctionToolCall(
                type="function_call",
                name=state.tool_call_name,
                namespace=state.tool_call_namespace,
                arguments=state.accumulated_text,
                status="completed",
                id=state.current_item_id,
                call_id=state.tool_call_id,
            ),
        ),
    ]
    state.output_index += 1
    state.tool_call_namespace = None
    state.current_state = _StateType.NONE
    return events


class _StateHandlers(NamedTuple):
    """Tuple for each state: open(start), delta(chunk), done(finish)."""

    open_fn: Callable[..., list[StreamingResponsesResponse]]
    delta_fn: Callable[..., list[StreamingResponsesResponse]]
    done_fn: Callable[..., list[StreamingResponsesResponse]]


def split_delta(delta: DeltaMessage) -> list[DeltaMessage]:
    """Decompose a DeltaMessage with multiple fields into atomic deltas.

    The Responses API emits typed SSE events (one type per event), so a
    compound DeltaMessage must be split before entering the state machine.
    Order: reasoning -> content -> tool_calls (grouped by index).
    """
    has_reasoning = delta.reasoning is not None
    has_content = delta.content is not None
    has_tools = bool(delta.tool_calls)
    parts = int(has_reasoning) + int(has_content) + int(has_tools)

    if parts <= 1 and (
        not has_tools
        or len({tc.index for tc in delta.tool_calls if tc.index is not None}) <= 1
    ):
        return [delta]

    deltas: list[DeltaMessage] = []
    if has_reasoning:
        deltas.append(DeltaMessage(reasoning=delta.reasoning))
    if has_content:
        deltas.append(DeltaMessage(content=delta.content))
    if has_tools:
        groups: dict[int | None, list[DeltaToolCall]] = {}
        for tc in delta.tool_calls:
            groups.setdefault(tc.index, []).append(tc)
        for tcs in groups.values():
            deltas.append(DeltaMessage(tool_calls=tcs))
    return deltas or [delta]


class SimpleStreamingEventProcessor:
    """State-machine processor for the streaming path.

    Core flow:
      1. Resolve the target state from the delta_message
         (CONTENT / REASONING / TOOL_CALL).
      2. If the target state differs from the current one,
         close_current() then open() the new state.
      3. emit_delta() produces the incremental events for the state.

    State lifecycle:
      open()  ->  repeated emit_delta()  ->  close_current()
    """

    _STATE_HANDLERS: ClassVar[dict[_StateType, _StateHandlers]] = {
        _StateType.CONTENT: _StateHandlers(
            emit_simple_content_open,
            emit_simple_content_delta,
            emit_simple_content_done,
        ),
        _StateType.REASONING: _StateHandlers(
            emit_simple_reasoning_open,
            emit_simple_reasoning_delta,
            emit_simple_reasoning_done,
        ),
        _StateType.TOOL_CALL: _StateHandlers(
            emit_simple_tool_call_open,
            emit_simple_tool_call_delta,
            emit_simple_tool_call_done,
        ),
    }

    def __init__(
        self,
        state: SimpleStreamingState | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.state = state or SimpleStreamingState()
        self.tool_call_name_map = build_responses_tool_call_name_map(tools)
        self.output_items: list[ResponseOutputItem] = []

    def resolve_target_state(
        self, delta_message: DeltaMessage
    ) -> tuple[_StateType, Any]:
        """Decide which state the next delta belongs to.

        Priority: TOOL_CALL > REASONING > CONTENT, fallback to NONE.
        For TOOL_CALL the first tool_call object is also returned so
        callers can detect a switch between consecutive tools.
        """
        if (
            delta_message.tool_calls
            and delta_message.tool_calls[0].function is not None
        ):
            return _StateType.TOOL_CALL, delta_message.tool_calls[0]
        if delta_message.reasoning is not None:
            return _StateType.REASONING, None
        if delta_message.content:
            return _StateType.CONTENT, None
        return _StateType.NONE, None

    def needs_transition(self, target_state: _StateType, tool_call: Any) -> bool:
        """Return True when we must close the current state and open a new one.

        Two cases trigger a transition:
          1. The target state differs from the current state
             (e.g. CONTENT -> TOOL_CALL).
          2. We are already in TOOL_CALL but the next tool_call has a
             different index (multiple consecutive tool calls).
        """
        if self.state.current_state != target_state:
            return True
        return (
            target_state == _StateType.TOOL_CALL
            and tool_call is not None
            and self.state.tool_call_index is not None
            and tool_call.index is not None
            and self.state.tool_call_index != tool_call.index
        )

    def close_current(self) -> list[StreamingResponsesResponse]:
        """Close the current state and emit its 'done' event sequence."""
        handlers = self._STATE_HANDLERS.get(self.state.current_state)
        if handlers is None:
            return []
        events = handlers.done_fn(self.state)
        self.output_items.extend(
            event.item
            for event in events
            if isinstance(event, ResponseOutputItemDoneEvent)
        )
        return events

    def open(
        self, target_state: _StateType, tool_call: Any = None
    ) -> list[StreamingResponsesResponse]:
        """Open a new state and emit its 'added' / 'open' event sequence."""
        handlers = self._STATE_HANDLERS[target_state]
        if target_state == _StateType.TOOL_CALL:
            assert tool_call is not None
            call_name = resolve_responses_tool_call_name(
                tool_call.function.name,
                tool_call_name_map=self.tool_call_name_map,
            )
            return handlers.open_fn(
                self.state,
                call_name.name,
                tool_call.index,
                call_name.namespace,
                tool_call.id,
            )
        return handlers.open_fn(self.state)

    def emit_delta(
        self,
        delta_message: DeltaMessage,
        output: CompletionOutput,
        get_logprobs: Callable[[CompletionOutput], list[response_output_text.Logprob]]
        | None = None,
    ) -> list[StreamingResponsesResponse]:
        """Emit incremental events for the current state from the delta."""
        handlers = self._STATE_HANDLERS[self.state.current_state]

        if self.state.current_state == _StateType.TOOL_CALL:
            assert delta_message.tool_calls is not None
            combined_args = ""
            for tc in delta_message.tool_calls:
                if tc.function is not None and tc.function.arguments:
                    combined_args += tc.function.arguments
            if combined_args:
                return handlers.delta_fn(self.state, combined_args)
            return []
        elif self.state.current_state == _StateType.REASONING:
            assert delta_message.reasoning is not None
            return handlers.delta_fn(self.state, delta_message.reasoning)
        elif self.state.current_state == _StateType.CONTENT:
            assert delta_message.content is not None
            logprobs = get_logprobs(output) if get_logprobs else []
            return handlers.delta_fn(self.state, delta_message.content, logprobs)
        return []
