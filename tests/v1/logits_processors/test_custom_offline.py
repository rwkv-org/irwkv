# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata

import tests.v1.logits_processors.utils as logitproc_test_utils
from tests.v1.logits_processors.utils import (
    setup_fake_entrypoint,
)

# Create a mixture of requests which do and don't utilize the dummy logitproc


def test_fake_entrypoint_preserves_other_groups(monkeypatch):
    other_group_entrypoints = object()
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: other_group_entrypoints,
    )
    monkeypatch.setattr(
        logitproc_test_utils, "requires_spawn_multiprocessing", lambda: False
    )

    setup_fake_entrypoint(monkeypatch)

    logitproc_entrypoints = importlib.metadata.entry_points(
        group=logitproc_test_utils.LOGITSPROCS_GROUP
    )
    assert logitproc_entrypoints.names == [
        logitproc_test_utils.DUMMY_LOGITPROC_ENTRYPOINT
    ]
    assert (
        importlib.metadata.entry_points(group="another.entrypoint.group")
        is other_group_entrypoints
    )
    assert importlib.metadata.entry_points() is other_group_entrypoints
