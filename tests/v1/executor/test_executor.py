# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from irwkv.v1.executor import multiproc_executor as multiproc_executor_module
from irwkv.v1.executor.abstract import Executor
from irwkv.v1.executor.multiproc_executor import MultiprocExecutor
from irwkv.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("IRWKV_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])  # type: ignore[list-item]
    assert proc.terminate_called is expected_terminate
