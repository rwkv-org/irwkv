# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from irwkv.engine.arg_utils import EngineArgs
from irwkv.sampling_params import SamplingParams
from irwkv.v1.core.sched.scheduler import Scheduler
from irwkv.v1.engine.llm_engine import LLMEngine


class DummyV1Scheduler(Scheduler):
    def schedule(self, throttle_prefills: bool = False):
        raise Exception("Exception raised by DummyV1Scheduler")


def test_scheduler_plugins_v1(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("IRWKV_ENABLE_V1_MULTIPROCESSING", "0")

        with pytest.raises(Exception) as exception_info:
            engine_args = EngineArgs(
                model="rwkv-rs/rwkv7-g1-st",
                dtype="bfloat16",
                max_model_len=128,
                max_num_seqs=4,
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV1Scheduler,
            )

            engine = LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(exception_info.value) == "Exception raised by DummyV1Scheduler"
