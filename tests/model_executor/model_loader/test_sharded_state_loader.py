# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from irwkv.model_executor.model_loader import ShardedStateLoader


def test_filter_subtensors():
    state_dict = {
        "a": torch.empty(2),
        "b": torch.empty((2, 4)),
        "c": torch.empty((2, 4, 8)),
    }
    state_dict.update(
        {
            "x": state_dict["b"],
            "y": state_dict["c"][1, 2, :],
            "z": state_dict["c"][1, :, 4],
        }
    )
    filtered_state_dict = ShardedStateLoader._filter_subtensors(state_dict)
    assert tuple(filtered_state_dict.keys()) == ("a", "b", "c")
    for key, tensor in filtered_state_dict.items():
        # NOTE: don't use `equal` here, as the tensor might contain NaNs
        assert tensor is state_dict[key]


def _mock_pp_group(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    *,
    pp_rank: int = 0,
    tp_rank: int = 0,
    tp_world_size: int = 1,
) -> None:
    """Stub the parallel-state accessors used by the sharded state loader."""
    monkeypatch.setattr(
        "irwkv.distributed.get_pp_group",
        lambda: SimpleNamespace(world_size=world_size, rank_in_group=pp_rank),
    )
    monkeypatch.setattr(
        "irwkv.distributed.get_tp_group",
        lambda: SimpleNamespace(world_size=tp_world_size, rank_in_group=tp_rank),
    )
    monkeypatch.setattr(
        "irwkv.distributed.get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(
        "irwkv.distributed.get_pipeline_model_parallel_rank",
        lambda: pp_rank,
    )


@pytest.mark.parametrize(
    "world_size,expected",
    [
        (1, ShardedStateLoader.TP_ONLY_PATTERN),
        (2, ShardedStateLoader.PP_AND_TP_PATTERN),
    ],
)
def test_get_default_pattern(monkeypatch, world_size, expected):
    """The default pattern is chosen from the pipeline parallel group size."""
    _mock_pp_group(monkeypatch, world_size)
    assert ShardedStateLoader._get_default_pattern() == expected


def test_render_default_pattern_default_tp(monkeypatch):
    """Without a custom pattern and PP=1, legacy TP-only names are rendered."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_default_pp(monkeypatch):
    """Without a custom pattern and PP>1, pp/tp-aware names are rendered."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(part=0)
        == "model-pp-0-tp-0-part-0.safetensors"
    )


def test_render_default_pattern_custom_tp_only(monkeypatch):
    """A custom TP-only pattern is honored when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )
        == "model-rank-0-part-0.safetensors"
    )


def test_render_default_pattern_tp_only_with_pp_fails(monkeypatch):
    """TP-only filenames would collide across pipeline stages; fail fast."""
    _mock_pp_group(monkeypatch, 2)
    with pytest.raises(ValueError, match="does not embed the pipeline parallel"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.TP_ONLY_PATTERN, part=0
        )


def test_render_default_pattern_pp_pattern_without_pp_fails(monkeypatch):
    """A pp/tp-aware pattern cannot be rendered when pipeline parallelism is off."""
    _mock_pp_group(monkeypatch, 1)
    with pytest.raises(ValueError, match="requires pipeline parallelism"):
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )


def test_render_default_pattern_custom_pp(monkeypatch):
    """A custom pp/tp-aware pattern is honored when pipeline parallelism is on."""
    _mock_pp_group(monkeypatch, 2)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == "model-pp-0-tp-0-part-0.safetensors"
    )


@pytest.mark.parametrize(
    "pattern,match",
    [
        # tp field present but pp field missing: pipeline stages with the
        # same tp rank would write the same file.
        (
            "model-tp-{tp_rank}-part-{part}.safetensors",
            "does not embed the pipeline parallel",
        ),
        # pp field present but tp field missing: TP workers would overwrite
        # each other's shards.
        (
            "model-pp-{pp_rank}-part-{part}.safetensors",
            "does not embed the tensor parallel",
        ),
    ],
)
def test_render_default_pattern_incomplete_custom_patterns_fail(
    monkeypatch, pattern, match
):
    """A custom pattern must distinguish every active rank (pp and tp)."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)
    with pytest.raises(ValueError, match=match):
        ShardedStateLoader._render_default_pattern(pattern, part=0)


def test_render_default_pattern_no_rank_field_with_tp_fails(monkeypatch):
    """Without any rank field, TP workers would collide even when PP is off."""
    _mock_pp_group(monkeypatch, 1, tp_world_size=2)
    with pytest.raises(ValueError, match="does not embed the tensor parallel"):
        ShardedStateLoader._render_default_pattern(
            "model-part-{part}.safetensors", part=0
        )


def test_render_default_pattern_custom_pp_tp1_allows_pp_only(monkeypatch):
    """With TP=1 a pp-only custom pattern is safe and must be accepted."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_world_size=1)
    assert (
        ShardedStateLoader._render_default_pattern(
            "model-pp-{pp_rank}-part-{part}.safetensors", part=0
        )
        == "model-pp-1-part-0.safetensors"
    )


@pytest.mark.parametrize(
    "pattern,match",
    [
        # {{pp_rank}} renders as literal text: both stages would write the
        # same file, so it must not count as a pp field.
        (
            "model-{{pp_rank}}-tp-{tp_rank}-part-{part}.safetensors",
            "does not embed the pipeline parallel",
        ),
        # Same for an escaped tp field: TP workers would overwrite each
        # other's shards.
        (
            "model-pp-{pp_rank}-tp-{{tp_rank}}-part-{part}.safetensors",
            "does not embed the tensor parallel",
        ),
    ],
)
def test_render_default_pattern_escaped_field_not_rank_field(
    monkeypatch, pattern, match
):
    """Escaped braces are literal text and cannot pose as rank fields."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)
    with pytest.raises(ValueError, match=match):
        ShardedStateLoader._render_default_pattern(pattern, part=0)


@pytest.mark.parametrize(
    "pattern,world_size,tp_world_size,pp_rank,tp_rank,expected",
    [
        # Mixed tp alias works when pipeline parallelism is on.
        (
            "model-pp-{pp_rank}-rank-{rank}-part-{part}.safetensors",
            2,
            2,
            1,
            0,
            "model-pp-1-rank-0-part-0.safetensors",
        ),
        # tp_rank alias works for TP-only checkpoints too.
        (
            "model-tp-{tp_rank}-part-{part}.safetensors",
            1,
            2,
            0,
            3,
            "model-tp-3-part-0.safetensors",
        ),
    ],
)
def test_render_default_pattern_tp_aliases_accepted(
    monkeypatch, pattern, world_size, tp_world_size, pp_rank, tp_rank, expected
):
    """Both tp aliases are supplied at render time, so either works."""
    _mock_pp_group(
        monkeypatch,
        world_size,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
    )
    assert ShardedStateLoader._render_default_pattern(pattern, part=0) == expected


def test_render_default_pattern_unsupported_field_fails(monkeypatch):
    """Unknown placeholders are rejected instead of KeyError-ing later."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_world_size=2)
    with pytest.raises(ValueError, match="unsupported placeholder"):
        ShardedStateLoader._render_default_pattern(
            "model-{pp_rank}-{tp_rank}-{epoch}-part-{part}.safetensors", part=0
        )


@pytest.mark.parametrize(
    "pp_rank,tp_rank,expected",
    [
        (0, 0, "model-pp-0-tp-0-part-0.safetensors"),
        (1, 2, "model-pp-1-tp-2-part-0.safetensors"),
        (3, 5, "model-pp-3-tp-5-part-0.safetensors"),
    ],
)
def test_render_default_pattern_non_zero_ranks(monkeypatch, pp_rank, tp_rank, expected):
    """The rendered filename must embed the actual pp/tp ranks, not zeros."""
    _mock_pp_group(monkeypatch, 2, pp_rank=pp_rank, tp_rank=tp_rank)
    assert (
        ShardedStateLoader._render_default_pattern(
            ShardedStateLoader.PP_AND_TP_PATTERN, part=0
        )
        == expected
    )


def test_render_default_pattern_part_wildcard(monkeypatch):
    """The glob wildcard '*' passes through as the part token for file lookup."""
    _mock_pp_group(monkeypatch, 1, tp_rank=4)
    assert (
        ShardedStateLoader._render_default_pattern(part="*")
        == "model-rank-4-part-*.safetensors"
    )


def test_save_model_pp_aware_filenames(monkeypatch, tmp_path):
    """save_model writes pp/tp-aware filenames for pipeline-parallel models."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=2)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.zeros(4, dtype=torch.float32)}
    )
    ShardedStateLoader.save_model(model, str(tmp_path))
    assert saved == ["model-pp-1-tp-2-part-0.safetensors"]


def test_save_model_part_split(monkeypatch, tmp_path):
    """save_model rolls over to a new part when max_size is exceeded."""
    _mock_pp_group(monkeypatch, 1, tp_rank=0)

    saved: list[str] = []

    def _capture(tensors, path, **kwargs):
        saved.append(os.path.basename(path))

    monkeypatch.setattr("safetensors.torch.save_file", _capture)

    # Two float32[4] tensors (16 B each); max_size=20 fits one but not two.
    model = SimpleNamespace(
        state_dict=lambda: {
            "a": torch.zeros(4, dtype=torch.float32),
            "b": torch.zeros(4, dtype=torch.float32),
        }
    )
    ShardedStateLoader.save_model(model, str(tmp_path), max_size=20)
    assert saved == [
        "model-rank-0-part-0.safetensors",
        "model-rank-0-part-1.safetensors",
    ]


def test_save_model_rejects_bad_pattern_before_overwrite(monkeypatch, tmp_path):
    """An unsafe pattern is rejected before any shard file is written."""
    _mock_pp_group(monkeypatch, 2, pp_rank=1, tp_rank=0, tp_world_size=2)

    # With the escaped pp field this pattern renders both stages to the
    # same literal name; the validation must fire before any write.
    colliding = tmp_path / "model-{pp_rank}-tp-0-part-0.safetensors"
    colliding.write_bytes(b"existing shard")
    before = sorted(p.name for p in tmp_path.iterdir())

    model = SimpleNamespace(
        state_dict=lambda: {"w": torch.zeros(4, dtype=torch.float32)}
    )
    with pytest.raises(ValueError, match="does not embed the pipeline parallel"):
        ShardedStateLoader.save_model(
            model,
            str(tmp_path),
            pattern="model-{{pp_rank}}-tp-{tp_rank}-part-{part}.safetensors",
        )

    # The pre-existing shard is untouched and nothing new was written.
    assert colliding.read_bytes() == b"existing shard"
    assert sorted(p.name for p in tmp_path.iterdir()) == before
