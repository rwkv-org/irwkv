# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from irwkv.v1.attention.backend import AttentionCGSupport
from irwkv.v1.core.kv_cache_utils import KVCacheBlockCopy
from irwkv.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    compute_layout_strides,
)
from irwkv.v1.worker.gpu import attn_utils
from irwkv.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from irwkv.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
    customize_attention_spec,
    map_kv_caches_to_kernel_blocks,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport, varlen_bound: int | None = None):
        self.support = support
        self.varlen_bound = varlen_bound

    def get_cudagraph_support(self, *_args):
        return self.support

    def get_varlen_cudagraph_max_query_len(self, *_args):
        return self.varlen_bound


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_varlen_cudagraph_unsupported_backend_checks_scoped_bounds():
    """ALWAYS passes without a bound, other builders need one at least as wide as
    the requested length, and NEVER fails whatever bound a builder reports."""
    config: Any = SimpleNamespace()
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    def group(
        backend: Any,
        layer_name: str,
        support: AttentionCGSupport,
        bound: int | None = None,
    ):
        builder: Any = _FakeMetadataBuilder(support, bound)
        attn_group = AttentionGroup(backend, [layer_name], spec, 0)
        attn_group.metadata_builders = [builder]
        return attn_group

    target = group(_TargetBackend, "target", AttentionCGSupport.ALWAYS)
    draft = group(_DraftBackend, "draft", AttentionCGSupport.UNIFORM_BATCH, 8)
    never = group(_DraftBackend, "never", AttentionCGSupport.NEVER, 8)

    unsupported = attn_utils.get_varlen_cudagraph_unsupported_backend
    assert unsupported([[target, draft]], config, 8) is None
    assert unsupported([[target, draft]], config, 9) == ("_DraftBackend", 8)
    assert (
        unsupported([[target, draft]], config, 9, checked_layer_names={"target"})
        is None
    )
    assert unsupported([[target, never]], config, 1) == ("_DraftBackend", None)


def test_get_kv_sharing_fast_prefill_eligible_layers(monkeypatch: pytest.MonkeyPatch):
    """Fast prefill applies to the contiguous suffix of KV-sharing layers."""

    def check(
        layer_names: list[str],
        shared: dict[str, str],
    ) -> set[str]:
        monkeypatch.setattr(
            attn_utils,
            "get_layers_from_irwkv_config",
            lambda *a, **k: {
                name: SimpleNamespace(kv_sharing_target_layer_name=shared.get(name))
                for name in layer_names
            },
        )
        monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda *a: shared)
        irwkv_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=True)
        )
        return attn_utils.get_kv_sharing_fast_prefill_eligible_layers(irwkv_config)

    # No KV sharing: nothing is eligible.
    assert check(["t0", "t1"], {}) == set()

    # Trailing run of sharing layers (YOCO-style second half).
    assert check(["t0", "t1", "t2", "t3"], {"t2": "t1", "t3": "t1"}) == {"t2", "t3"}

    # A non-sharing layer after a sharing one breaks the suffix.
    assert check(["t0", "t1", "t2", "t3"], {"t1": "t0", "t3": "t0"}) == {"t3"}

    # Feature flag off: nothing is eligible even with sharing layers.
    monkeypatch.setattr(
        attn_utils, "get_layers_from_irwkv_config", lambda *a, **k: {"t0": None}
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda *a: {"t0": "t0"}
    )
    irwkv_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(irwkv_config) == set()


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, 4, 64),
        ([256], 4, 64),
        ([64], 16, 16),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )


class _TableBuilder:
    requires_block_table_width = False
    supports_update_block_table = False

    def __init__(self, kv_cache_spec, layer_names, irwkv_config, device):
        self.kv_cache_spec = kv_cache_spec

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata.block_table_tensor


_IRWKV_CONFIG = SimpleNamespace(
    model_config=SimpleNamespace(max_model_len=8192),
    scheduler_config=SimpleNamespace(max_num_seqs=4),
    parallel_config=SimpleNamespace(decode_context_parallel_size=1),
)


def _packed_group(tokens_per_state, row_bytes, supported_sizes):
    spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        state_content_bytes=row_bytes,
        tokens_per_state=tokens_per_state,
    )
    backend = SimpleNamespace(
        get_builder_cls=lambda: _TableBuilder,
        get_supported_kernel_block_sizes=lambda _=None: supported_sizes,
        get_name=lambda: "FAKE",
        customize_spec=lambda spec: spec,
    )
    return AttentionGroup(backend, ["layer.0"], spec, 0)


def test_packed_compressed_group_gets_kernel_block_table():
    """A packed indexer cache whose kernel needs 64-state pages gets a table and
    a bound view in kernel blocks that address the same packed-block rows."""
    row, stride_rows = 132, 5 * 64
    group = _packed_group(4, row, [128, 256])
    group.create_metadata_builders(_IRWKV_CONFIG, "cpu", kernel_block_size=1024)
    assert group.get_metadata_builder().kv_cache_spec.block_size == 256

    backing = torch.arange(3 * stride_rows).repeat_interleave(row)
    manager = backing.view(3, 1, stride_rows, row)[:, :, :256]
    pages = map_kv_caches_to_kernel_blocks({"layer.0": manager}, [group])["layer.0"]
    metadata = create_common_attn_metadata(BatchSpec([2048], [1]), 1024, "cpu")
    metadata.block_table_tensor = torch.tensor([[0, 2]], dtype=torch.int32)
    table = group.build_metadata(metadata)
    assert table.tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]
    assert torch.equal(pages[table[0, 4:], 0].flatten(0, 1), manager[2, 0])


def test_split_group_maps_manager_block_table():
    """Groups split in a dense layout keep manager block tables; builders get
    b * blocks_per_kv_block + j against the view allocated in kernel blocks."""
    group = _packed_group(4, 132, [128, 256])
    group.create_metadata_builders(_IRWKV_CONFIG, "cpu", kernel_block_size=256)
    kernel_view = torch.zeros(12, 1, 64, 132)
    bound = map_kv_caches_to_kernel_blocks({"layer.0": kernel_view}, [group])
    assert bound["layer.0"] is kernel_view

    metadata = create_common_attn_metadata(BatchSpec([2048], [1]), 1024, "cpu")
    metadata.block_table_tensor = torch.tensor([[0, 2]], dtype=torch.int32)
    table = group.build_metadata(metadata)
    assert table.tolist() == [[0, 1, 2, 3, 8, 9, 10, 11]]


@pytest.mark.parametrize(
    ("block_size", "page_states"), [(1024, 64), (384, 32), (256, 0)]
)
def test_split_blocks_align_to_kernel_blocks(block_size, page_states):
    """Backends taking only fixed kernel blocks align packed blocks to the kernel
    block the final block size is split into, and not at all when unsplit."""
    group = _packed_group(4, 132, [128, 256])
    spec = customize_attention_spec(group.backend, group.kv_cache_spec)
    spec = spec.copy_with_new_block_size(block_size)
    assert spec.get_block_stride_alignment() == max(page_states * 132, 1)


def test_packed_token_cache_cannot_be_split():
    group = _packed_group(1, 1152, [64])
    group.create_metadata_builders(_IRWKV_CONFIG, "cpu", kernel_block_size=1024)
    packed = torch.zeros(3, 1, 2048, 1152, dtype=torch.uint8)[:, :, :1024]
    with pytest.raises(ValueError, match="cannot be split"):
        map_kv_caches_to_kernel_blocks({"layer.0": packed}, [group])
