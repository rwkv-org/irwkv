# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for MultiprocExecutor at the executor level.
This test directly tests the executor without going through the LLM interface,
focusing on executor initialization, RPC calls, and distributed execution.
"""

import pytest

from irwkv.v1.executor import multiproc_executor
from irwkv.v1.executor.multiproc_executor import MultiprocExecutor


@pytest.mark.parametrize(
    ("local_world_size", "data_parallel_size_local", "expected_num_local_procs"),
    [(1, 4, 4), (4, 1, 4), (2, 0, 2)],
)
def test_multiproc_executor_counts_all_local_dp_workers(
    monkeypatch: pytest.MonkeyPatch,
    local_world_size: int,
    data_parallel_size_local: int,
    expected_num_local_procs: int,
):
    """All colocated DP workers share the node's startup CPU budget."""
    executor = object.__new__(MultiprocExecutor)
    executor.world_size = local_world_size
    executor.local_world_size = local_world_size
    executor.parallel_config = type(
        "ParallelConfig",
        (),
        {"data_parallel_size_local": data_parallel_size_local},
    )()

    monkeypatch.setattr(
        executor,
        "_get_parallel_sizes",
        lambda: (local_world_size, 1, 1),
    )

    class StopExecutorInit(Exception):
        pass

    def capture_num_local_procs(num_local_procs: int):
        assert num_local_procs == expected_num_local_procs
        raise StopExecutorInit

    monkeypatch.setattr(
        multiproc_executor,
        "set_multiprocessing_worker_envs",
        capture_num_local_procs,
    )

    with pytest.raises(StopExecutorInit):
        executor._init_executor()
    executor._finalizer.detach()
