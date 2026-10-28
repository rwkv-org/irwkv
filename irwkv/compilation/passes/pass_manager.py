# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from torch import fx as fx

from irwkv import envs
from irwkv.compilation.passes.utility.post_cleanup import PostCleanupPass
from irwkv.config import IrwkvConfig, set_current_irwkv_config
from irwkv.logger import init_logger
from irwkv.utils.system_utils import set_env_var

from .inductor_pass import (
    CustomGraphPass,
    InductorPass,
    get_pass_context,
)
from .ir.clone_elimination import UnsafeCloneEliminationPass
from .ir.lowering_pass import IrwkvIRLoweringPass
from .irwkv_inductor_pass import IrwkvInductorPass, IrwkvPatternMatcherPass
from .utility.noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def with_pattern_match_debug(fn: Callable[P, R]) -> Callable[P, R]:
    """Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if (debug_val := envs.IRWKV_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


class PostGradPassManager(CustomGraphPass):  # type: ignore[misc]
    """The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self) -> None:
        self.passes: list[InductorPass] = []

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph) -> None:
        IrwkvInductorPass.dump_prefix = 0  # reset dump index

        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
                IrwkvInductorPass.dump_prefix += 1
            else:
                logger.debug("Skipping %s with compile range %s", pass_, compile_range)

        # perform the first post-cleanup before IR lowering to clean up fusion artifacts
        # and make sure no dead IR ops are lowered.
        self.post_cleanup(graph)
        IrwkvInductorPass.dump_prefix += 1

        # lowering before cleanup so DCE can clean up lowered ops.
        # DCE handles mutating ops correctly as well.
        self.ir_lowering(graph)
        IrwkvInductorPass.dump_prefix += 1
        self.clone_elimination(graph)
        IrwkvInductorPass.dump_prefix += 1

        # clean up after lowering again
        self.post_cleanup(graph)
        IrwkvInductorPass.dump_prefix += 1

        IrwkvInductorPass.dump_prefix = None  # Cleanup index

        IrwkvPatternMatcherPass.log_match_summary()

    def configure(self, config: IrwkvConfig) -> None:
        self.pass_config = config.compilation_config.pass_config

        # Set the current irwkv config to allow tracing CustomOp instances
        with set_current_irwkv_config(config, check_compile=False):
            if self.pass_config.eliminate_noops:
                self.passes += [NoOpEliminationPass(config)]

            self.ir_lowering = IrwkvIRLoweringPass(config)
            self.clone_elimination = UnsafeCloneEliminationPass(config)
            self.post_cleanup = PostCleanupPass(config)

    def add(self, pass_: InductorPass) -> None:
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self) -> str:
        """The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        passes = []

        state: dict[str, Any] = {"pass_config": self.pass_config.compute_hash()}
        for pass_ in self.passes:
            passes.append(pass_.uuid())

        passes.append(self.post_cleanup.uuid())
        passes.append(self.ir_lowering.uuid())
        passes.append(self.clone_elimination.uuid())
        passes.append(self.post_cleanup.uuid())

        # Include the compile range in the uuid to ensure that inductor
        # recompiles the graph for the new dynamic compile range.
        state["compile_range"] = str(get_pass_context().compile_range)
        state["passes"] = passes
        return InductorPass.hash_dict(state)
