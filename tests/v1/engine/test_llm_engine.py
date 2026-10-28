# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest

from irwkv.exceptions import ProfilerAlreadyActiveError
from irwkv.v1.engine.llm_engine import LLMEngine

if TYPE_CHECKING:
    from tests.conftest import IrwkvRunner
else:
    IrwkvRunner = object


@pytest.mark.skip_global_cleanup
def test_step_invalidates_mm_processor_cache_misses():
    engine = object.__new__(LLMEngine)
    engine.should_execute_dummy_batch = False
    engine.log_stats = False
    engine.logger_manager = None

    mm_processor_cache = MagicMock()
    engine.renderer = MagicMock(mm_processor_cache=mm_processor_cache)

    engine_core_output = MagicMock(
        mm_cache_miss_hashes=["missing-hash-1", "missing-hash-2"]
    )
    outputs = MagicMock(
        outputs=[engine_core_output],
        timestamp=0.0,
        scheduler_stats=None,
    )
    engine.engine_core = MagicMock()
    engine.engine_core.get_output.return_value = outputs

    processed_outputs = MagicMock(reqs_to_abort=[], request_outputs=[])
    engine.output_processor = MagicMock()
    engine.output_processor.process_outputs.return_value = processed_outputs

    assert engine.step() == []
    assert mm_processor_cache.invalidate.call_args_list == [
        call("missing-hash-1"),
        call("missing-hash-2"),
    ]


def test_duplicate_profile_start_rejected_before_engine_core_dispatch():
    engine = object.__new__(LLMEngine)
    engine.engine_core = MagicMock()
    engine._profile_session_active = False

    engine.start_profile("first")

    with pytest.raises(ProfilerAlreadyActiveError):
        engine.start_profile("duplicate")

    engine.engine_core.profile.assert_called_once_with(True, "first", None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile_prefix": "../trace"},
        {"delay_iterations": -1},
        {"max_iterations": -1},
    ],
)
def test_invalid_profile_controls_rejected_before_engine_core_dispatch(kwargs):
    engine = object.__new__(LLMEngine)
    engine.engine_core = MagicMock()
    engine._profile_session_active = False

    with pytest.raises(ValueError):
        engine.start_profile(**kwargs)

    engine.engine_core.profile.assert_not_called()
