# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

from irwkv.compilation.passes.inductor_pass import (
    CallableInductorPass,
    InductorPass,
    pass_context,
)
from irwkv.compilation.passes.pass_manager import PostGradPassManager
from irwkv.config import IrwkvConfig
from irwkv.config.utils import Range


# dummy custom pass that doesn't inherit
def simple_callable(graph: torch.fx.Graph):
    pass


# Should fail to add directly to the pass manager
def test_bad_callable():
    config = IrwkvConfig()

    pass_manager = PostGradPassManager()
    pass_manager.configure(config)

    with pytest.raises(AssertionError):
        pass_manager.add(simple_callable)


def test_default_passes_run_without_optional_kernels():
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    pass_manager = PostGradPassManager()
    pass_manager.configure(IrwkvConfig())

    with pass_context(Range(start=1, end=8)):
        pass_manager(graph.graph)

    graph.recompile()
    assert graph(torch.tensor(2)).item() == 3


# Pass that inherits from InductorPass
class ProperPass(InductorPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        pass


@pytest.mark.parametrize(
    "callable",
    [
        ProperPass(),
        # Can also wrap callables in CallableInductorPass for compliance
        CallableInductorPass(simple_callable),
        CallableInductorPass(simple_callable, InductorPass.hash_source(__file__)),
    ],
)
def test_pass_manager_uuid(callable):
    # Set the pass context as PassManager uuid uses it
    with pass_context(Range(start=1, end=8)):
        config = IrwkvConfig()

        pass_manager = PostGradPassManager()
        pass_manager.configure(config)

        # Check that UUID is different if the same pass is added 2x
        pass_manager.add(callable)
        uuid1 = pass_manager.uuid()
        pass_manager.add(callable)
        uuid2 = pass_manager.uuid()
        assert uuid1 != uuid2

        # UUID should be the same as the original one,
        # as we constructed in the same way.
        pass_manager2 = PostGradPassManager()
        pass_manager2.configure(config)
        pass_manager2.add(callable)
        assert uuid1 == pass_manager2.uuid()

        # UUID should be different due to config change
        config2 = copy.deepcopy(config)
        config2.compilation_config.pass_config.eliminate_noops = (
            not config2.compilation_config.pass_config.eliminate_noops
        )
        pass_manager3 = PostGradPassManager()
        pass_manager3.configure(config2)
        pass_manager3.add(callable)
        assert uuid1 != pass_manager3.uuid()
