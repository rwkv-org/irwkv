# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import irwkv.config
from irwkv.compilation.passes.irwkv_inductor_pass import (
    IrwkvFusionPatternMatcherPass,
    IrwkvPatternMatcherPass,
    IrwkvPatternReplacement,
)
from irwkv.config import CompilationConfig, CompilationMode, IrwkvConfig
from irwkv.platforms import current_platform
from tests.compile.backend import TestBackend


class ReluToAbsPattern(IrwkvPatternReplacement):
    """Replaces relu(x) with abs(x) — a minimal test fixture."""

    @property
    def pattern(self):
        def _pattern(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.relu.default(x)

        return _pattern

    @property
    def replacement(self):
        def _replacement(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.abs.default(x)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_fp32(4)]


class ExpToSqrtPattern(IrwkvPatternReplacement):
    """A second distinct pattern type — used to test uuid differentiation."""

    @property
    def pattern(self):
        def _pattern(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.exp.default(x)

        return _pattern

    @property
    def replacement(self):
        def _replacement(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.sqrt.default(x)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_fp32(4)]


class ReluFusionPass(IrwkvFusionPatternMatcherPass):
    def __init__(self, config: IrwkvConfig) -> None:
        super().__init__(config, "test_relu_fusion")
        self.register(ReluToAbsPattern())


class TwoPatternFusionPass(IrwkvFusionPatternMatcherPass):
    def __init__(self, config: IrwkvConfig) -> None:
        super().__init__(config, "test_two_pattern_fusion")
        self.register(ReluToAbsPattern())
        self.register(ExpToSqrtPattern())


@pytest.fixture
def irwkv_config():
    return IrwkvConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.IRWKV_COMPILE),
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
def test_register_tracks_patterns(irwkv_config):
    """register() appends each IrwkvPatternReplacement to _pattern_replacements."""
    with irwkv.config.set_current_irwkv_config(irwkv_config):
        single = ReluFusionPass(irwkv_config)
        two = TwoPatternFusionPass(irwkv_config)

    assert len(single._pattern_replacements) == 1
    assert len(two._pattern_replacements) == 2


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
def test_uuid_stable(irwkv_config):
    """Two instances of the same pass class produce identical uuids."""
    with irwkv.config.set_current_irwkv_config(irwkv_config):
        p1 = ReluFusionPass(irwkv_config)
        p2 = ReluFusionPass(irwkv_config)
        p3 = TwoPatternFusionPass(irwkv_config)

    assert p1.uuid() == p2.uuid()
    assert p1.uuid() != p3.uuid()
    assert p2.uuid() != p3.uuid()


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
@pytest.mark.parametrize("N", [1, 2, 4])
def test_matched_count_and_match_table(irwkv_config, N):
    """matched_count and match_table reflect the number of matched patterns."""

    class Model(torch.nn.Module):
        def forward(self, *inputs):
            # N independent relus
            return sum(torch.relu(x) for x in inputs)

    with irwkv.config.set_current_irwkv_config(irwkv_config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)

        fusion_pass = ReluFusionPass(irwkv_config)
        backend = TestBackend(fusion_pass)
        model = torch.compile(Model(), backend=backend)

        inputs = [torch.rand(8) for _ in range(N)]
        model(*inputs)

    assert fusion_pass.matched_count == N
    assert IrwkvPatternMatcherPass.match_table["test_relu_fusion"] >= N
