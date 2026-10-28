# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

import irwkv.envs as envs
from irwkv.config import IrwkvConfig
from irwkv.exceptions import IRWKVValidationError
from irwkv.inputs import (
    EngineInput,
    PromptType,
    SingletonInput,
    split_enc_dec_input,
)
from irwkv.logger import init_logger
from irwkv.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from irwkv.multimodal.inputs import MultiModalFeatureSpec
from irwkv.multimodal.utils import argsort_mm_positions
from irwkv.platforms import current_platform
from irwkv.renderers import BaseRenderer, renderer_from_config
from irwkv.renderers.inputs.preprocess import parse_model_prompt
from irwkv.sampling_params import BeamSearchParams, SamplingParams
from irwkv.tasks import GENERATION_TASKS, SupportedTask
from irwkv.tokenizers import TokenizerLike
from irwkv.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from irwkv.utils.async_utils import make_async
from irwkv.utils.jsontree import json_iter_leaves
from irwkv.v1.engine import EngineCoreRequest
from irwkv.v1.kv_hints import KvHintsEnvelope

logger = init_logger(__name__)


class InputProcessor:
    def __init__(
        self,
        irwkv_config: IrwkvConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.irwkv_config = irwkv_config
        self.model_config = model_config = irwkv_config.model_config
        self.cache_config = irwkv_config.cache_config
        self.scheduler_config = irwkv_config.scheduler_config
        self.structured_outputs_config = irwkv_config.structured_outputs_config
        self.observability_config = irwkv_config.observability_config
        # Load the custom logits processor classes once; the returned callable
        # runs their validate_params hooks per request at admission.
        self.validate_logits_processors_params = (
            self._build_logits_processors_params_validator()
        )

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(irwkv_config)

        self.supports_mm_inputs = False
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False

        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers should run it on the renderer's
        # thread pool to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _build_logits_processors_params_validator(
        self,
    ) -> Callable[[SamplingParams], None]:
        """Load the custom logits processor classes and return the per-request
        params validator for the active model runner."""
        custom_logitsprocs = self.model_config.logits_processors

        from irwkv.v1.worker.gpu.sample.logits_processor import (
            build_custom_logits_processors_params_validator,
        )

        return build_custom_logits_processors_params_validator(custom_logitsprocs)

    def resolve_watermarking(self, params: SamplingParams | BeamSearchParams) -> bool:
        """Return whether watermarking will be applied to this request.

        ``None`` and ``True`` request watermarking, while ``False`` opts out.
        If watermarking is unavailable or incompatible with the request, emit
        a warning and return ``False``.
        """
        return self.irwkv_config._check_supports_watermarking(params)

    def apply_watermarking(self, params: SamplingParams, watermarking: bool) -> None:
        """Store the resolved `watermarking` on request-local params.

        Also records whether watermarking was requested on a watermarking
        engine but could not be applied, as opposed to an explicit opt-out.
        """
        # Sticky, so resolving an already resolved request keeps the flag.
        params._watermarking_skipped = params._watermarking_skipped or (
            self.irwkv_config.watermark_config is not None
            and params.watermarking is not False
            and not watermarking
        )
        params.watermarking = watermarking

    def _validate_params(
        self,
        params: SamplingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> bool:
        if not isinstance(params, SamplingParams):
            raise TypeError(
                f"params must be a SamplingParams, but got {type(params).__name__}"
            )

        if not any(task in GENERATION_TASKS for task in supported_tasks):
            raise IRWKVValidationError("This model does not support generation")

        params.verify(self.model_config, self.structured_outputs_config, self.tokenizer)
        if (
            params.prompt_logprob_token_ids is not None
            and self.irwkv_config.cache_config.kv_sharing_fast_prefill
        ):
            raise IRWKVValidationError(
                "prompt_logprob_token_ids is incorrect with "
                "--kv-sharing-fast-prefill; disable it for scoring.",
                parameter="prompt_logprob_token_ids",
            )

        self.validate_logits_processors_params(params)

        if self.model_config.return_sampling_mask:
            if params.temperature <= 0:
                raise ValueError(
                    "sampling distribution replay requires temperature > 0"
                )
            if params.top_k <= 0:
                raise ValueError(
                    "sampling distribution replay requires top_k > 0 to "
                    "bound sampling mask size, reduce transfer overhead, "
                    "and avoid potential OOMs"
                )
        if params.thinking_token_budget is not None and (
            self.irwkv_config.reasoning_config is None
            or not self.irwkv_config.reasoning_config.enabled
        ):
            raise IRWKVValidationError(
                "thinking_token_budget is set but reasoning_config is "
                "not configured. Please set --reasoning-parser "
                "and/or --reasoning-config to use thinking_token_budget."
            )
        if params.trace_decode_token_ids and not self.model_config.enable_trace_replay:
            raise IRWKVValidationError(
                "trace_decode_token_ids is set but trace replay is not "
                "enabled. Start the engine with --enable-trace-replay "
                "to use it."
            )

        return self.resolve_watermarking(params)

    def _normalize_trace_replay_params(
        self, sampling_params: SamplingParams, prompt_len: int
    ) -> None:
        """Apply trace replay's generation semantics to request-local params."""
        trace_token_ids = sampling_params.trace_decode_token_ids
        assert trace_token_ids
        assert sampling_params.max_tokens is not None

        max_trace_len = max(self.model_config.max_model_len - prompt_len, 1)
        trace_token_ids = trace_token_ids[:max_trace_len]
        sampling_params.trace_decode_token_ids = trace_token_ids

        # Apply this after the generation config so its EOS token cannot stop
        # replay before the trace is exhausted.
        sampling_params.max_tokens = min(
            len(trace_token_ids), sampling_params.max_tokens
        )
        sampling_params.min_tokens = 0
        sampling_params.ignore_eos = True
        sampling_params._eos_token_id = None
        sampling_params.stop = []
        sampling_params.stop_token_ids = []
        sampling_params._all_stop_token_ids = set()

    def _get_mm_identifier(self, mm_hash: str) -> str:
        """Return the multimodal cache key."""
        return mm_hash

    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],
        mm_kwargs: dict[str, list],
    ) -> None:
        """Inject pre-processed mm_kwargs into the processor cache.

        Call this when mm_kwargs have already been through the HF processor
        externally (e.g. by a frontend that transfers pre-processed tensors
        to the backend).  This ensures MM cache hit rate metrics are reported
        accurately and avoids redundant processing on subsequent requests
        with the same images.

        Uses ``get_and_update_item()`` with an empty prompt_updates list,
        since token expansion has already been handled externally.
        """
        cache = self.renderer.mm_processor_cache
        if cache is None:
            return
        try:
            for modality, hashes in mm_hashes.items():
                items = mm_kwargs.get(modality, [])
                for i, mm_hash in enumerate(hashes):
                    if i < len(items) and items[i] is not None:
                        # Insert into cache via get_and_update_item.
                        # Use the returned item (may be an address for SHM
                        # cache or the original item for LRU cache).
                        items[i], _ = cache.get_and_update_item(
                            (items[i], []),
                            mm_hash,
                        )
            # Update cache stats to reflect the externally processed items
            self.renderer.update_mm_cache_stats()
        except Exception:
            logger.warning(
                "Failed to inject mm_kwargs into processor cache",
                exc_info=True,
            )

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to irwkv; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.IRWKV_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "IRWKV_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
    ) -> EngineCoreRequest:
        watermarking = self._validate_params(params, supported_tasks)
        parallel_config = self.irwkv_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise IRWKVValidationError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            engine_input: EngineInput = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in the future. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            renderer = self.renderer
            model_config = self.model_config

            parsed_prompt = parse_model_prompt(model_config, prompt)
            tok_params = renderer.default_cmpl_tok_params.with_kwargs(
                **(tokenization_kwargs or {})
            )

            (engine_input,) = renderer.render_cmpl(
                [parsed_prompt],
                tok_params,
            )

        current_platform.validate_request(engine_input, params)

        encoder_input, decoder_input = split_enc_dec_input(engine_input)
        self._validate_model_inputs(encoder_input, decoder_input)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_input["type"] == "embeds":
            prompt_embeds = decoder_input["prompt_embeds"]
            prompt_token_ids = decoder_input.get("prompt_token_ids")
            prompt_is_token_ids = decoder_input.get("is_token_ids")
        else:
            prompt_token_ids = decoder_input["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        # TODO: can we avoid cloning here in multiproc case?
        sampling_params = params.clone()
        # Resolve on the request-local copy: `params` may be shared across
        # prompts, requests and even engines by the caller.
        self.apply_watermarking(sampling_params, watermarking)
        prompt_len = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        rows = sampling_params.prompt_logprob_token_ids
        if rows is not None:
            start = sampling_params.prompt_logprob_start or 0
            num_rows = max(prompt_len - 1 - start, 0)
            if len(rows) != num_rows:
                raise IRWKVValidationError(
                    f"prompt_logprob_token_ids has {len(rows)} rows, but the "
                    f"prompt has {num_rows} scored rows "
                    f"(prompt_len - 1 - prompt_logprob_start).",
                    parameter="prompt_logprob_token_ids",
                    value=len(rows),
                )
        # If unset max tokens, then generate up to the max_model_len.
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = self.model_config.max_model_len - prompt_len
            # min_tokens is not checked while max_tokens is unset.
            if sampling_params.min_tokens > sampling_params.max_tokens:
                raise IRWKVValidationError(
                    f"min_tokens must be less than or equal to "
                    f"max_tokens={sampling_params.max_tokens}, got "
                    f"{sampling_params.min_tokens}."
                )

        sampling_params.update_from_generation_config(
            self.generation_config_fields,
            self.renderer.get_eos_token_id(),
        )
        if self.tokenizer is not None:
            sampling_params.update_from_tokenizer(self.tokenizer)
        if sampling_params.trace_decode_token_ids:
            self._normalize_trace_replay_params(sampling_params, prompt_len)

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_input["type"] == "multimodal":
            decoder_mm_inputs = decoder_input["mm_kwargs"]
            decoder_mm_positions = decoder_input["mm_placeholders"]
            decoder_mm_hashes = decoder_input["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(base_mm_hash),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            arrival_time=arrival_time,
            cache_salt=decoder_input.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            session_id=session_id,
            kv_hints=kv_hints,
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        if self.skip_prompt_length_check and prompt_type == "encoder":
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise IRWKVValidationError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens."
            )

            raise IRWKVValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise IRWKVValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "embeds":
            is_token_ids = prompt_input.get("is_token_ids")
            if is_token_ids is not None and len(is_token_ids) != prompt_len:
                raise IRWKVValidationError(
                    "prompt_is_token_ids must have the same length as prompt_embeds "
                    f"(expected {prompt_len}, got {len(is_token_ids)}).",
                    parameter="prompt_is_token_ids",
                )

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        raise IRWKVValidationError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids:
            self.renderer.validate_token_ids(prompt_ids)

    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
