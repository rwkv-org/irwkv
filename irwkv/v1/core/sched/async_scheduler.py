# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import irwkv.envs as envs
from irwkv.logger import init_logger
from irwkv.platforms import current_platform
from irwkv.v1.core.sched.output import SchedulerOutput
from irwkv.v1.core.sched.scheduler import Scheduler
from irwkv.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pp_size = self.parallel_config.pipeline_parallel_size
        # Only XPU can disable PP decode microbatching via IRWKV_XPU_PP_MICROBATCH.
        self.decode_stagger = self.pp_size
        if current_platform.is_xpu() and not envs.IRWKV_XPU_PP_MICROBATCH:
            self.decode_stagger = 1

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            request.num_output_placeholders += self.num_sampled_tokens_per_step

            request.next_decode_eligible_step = self.current_step + self.decode_stagger

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
