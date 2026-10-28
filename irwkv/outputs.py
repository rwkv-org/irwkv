# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import MutableSequence
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from irwkv.logger import init_logger
from irwkv.logprobs import PromptLogprobs, SampleLogprobs
from irwkv.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


@dataclass
class SamplingMask:
    """Per-token sampling support sets aligned with completion token IDs.

    Each inner list contains the vocabulary token IDs that survived
    top-k / top-p / min-p filtering for the corresponding generated token.
    """

    token_ids: list[list[int]]


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        sampling_mask: The post-processing token support set for each generated
            token, if requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.

    """

    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    sampling_mask: SamplingMask | None = None

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"sampling_mask={self.sampling_mask}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        prompt_token_id_logprobs: Logprobs of prompt_logprob_token_ids, shaped
            [prompt_len - 1 - prompt_logprob_start, num_ids]; row i holds the
            logprobs of its own IDs as predictions of prompt token
            prompt_logprob_start + i + 1, with -inf in padded columns.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
        num_cache_creation_tokens: Prompt tokens currently counted as local
            prefix-cache writes for this request.
        kv_transfer_params: The params for remote K/V transfer.
        ec_transfer_params: The params for remote encoder-cache transfer.

    """

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestStateStats | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_cache_creation_tokens: int | None = None,
        *,
        prompt_token_id_logprobs: np.ndarray | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning_once(
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)
            )
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.prompt_token_id_logprobs = prompt_token_id_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens
        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one"""
        self.finished |= next_output.finished
        self.kv_transfer_params = next_output.kv_transfer_params
        self.ec_transfer_params = next_output.ec_transfer_params

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    if aggregate:
                        # Merge outputs with same index
                        completion.text += next_completion.text
                        if not isinstance(completion.token_ids, MutableSequence):
                            completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]
                        if next_completion.sampling_mask is not None:
                            if completion.sampling_mask is None:
                                completion.sampling_mask = SamplingMask([])
                            completion.sampling_mask.token_ids.extend(
                                next_completion.sampling_mask.token_ids
                            )
                        completion.cumulative_logprob = (
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    else:
                        # Replace the output with the new one
                        self.outputs[i] = next_completion
                    break
            else:
                self.outputs.append(next_completion)

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"encoder_prompt={self.encoder_prompt!r}, "
            f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"prompt_token_id_logprobs={self.prompt_token_id_logprobs}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished}, "
            f"metrics={self.metrics}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"num_cache_creation_tokens={self.num_cache_creation_tokens})"
        )


# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt=None,
    prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[],
    finished=True,
)
