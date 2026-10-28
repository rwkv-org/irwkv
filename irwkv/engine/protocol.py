# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from irwkv.config import IrwkvConfig, ModelConfig
from irwkv.config.kv_events import KVEventsConfig
from irwkv.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from irwkv.inputs import EngineInput, PromptType
from irwkv.outputs import RequestOutput
from irwkv.renderers import BaseRenderer
from irwkv.sampling_params import SamplingParams
from irwkv.tasks import SupportedTask
from irwkv.v1.engine import EngineCoreRequest
from irwkv.v1.engine.input_processor import InputProcessor
from irwkv.v1.kv_hints import KvHintsEnvelope

if TYPE_CHECKING:
    from irwkv.v1.engine import PauseMode


@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: EngineInput
    sampling_params: SamplingParams | None = None


class EngineClient(ABC):
    """Protocol class for Clients to Engine."""

    irwkv_config: IrwkvConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: InputProcessor

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    def get_kv_event_sources(self) -> dict[int, KVEventsConfig]:
        """KV-event publisher config of each engine, keyed by DP rank."""
        return {}

    def check_admission(  # noqa: B027
        self, n: int = 1, request_id: str | None = None
    ) -> None:
        """Reject the request up front if it would exceed queue limits.

        Called before a response is started so that overload rejections can
        carry an HTTP status, which is not possible once a streaming response
        has begun. Engines without admission control accept everything.

        Args:
            n: Number of sequences the request will occupy.
            request_id: Request id, used for logging only.

        Raises:
            GracefulHTTPError: If the request cannot be admitted.

        """

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.

        """
        ...

    @abstractmethod
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify the engine that a KV-transfer request was rejected before
        engine admission, so connector-side cleanup can run (e.g. free
        prefill blocks pinned on the P node).
        """
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy."""
        ...

    @abstractmethod
    async def start_profile(
        self,
        profile_prefix: str | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Start profiling with optional per-session overrides."""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine."""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache."""
        ...

    @abstractmethod
    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache."""
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache."""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """Sleep the engine."""
        ...

    @abstractmethod
    async def release_kv_cache_memory(self) -> None:
        """Discard KV cache physical GPU memory. Requires a completed pause."""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> bool:
        """Wake the tagged resources; return whether every engine is fully awake."""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping."""
        ...

    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: "PauseMode" = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation requests.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  and return partial results with "abort" reason (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
            clear_cache: DEPRECATED. Whether to clear KV and prefix caches
                after draining.

        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def compute_weight_checksums(self) -> list[dict[str, str]]:
        """Return rank-qualified weight digests from every worker of every engine."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks."""
        raise NotImplementedError

    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest
    ) -> None:
        """Initialize weight transfer for RL training."""
        raise NotImplementedError

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        raise NotImplementedError

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """Batched weight update for RL training."""
        raise NotImplementedError

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        raise NotImplementedError

    async def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        raise NotImplementedError

    async def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        raise NotImplementedError
