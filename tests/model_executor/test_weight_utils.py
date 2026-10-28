# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from unittest.mock import Mock

import huggingface_hub.constants
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from irwkv.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    drop_checkpoint_cache,
)


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


@pytest.mark.parametrize("fails", [False, True])
def test_drop_checkpoint_cache_releases_local_weight_files(
    tmp_path, monkeypatch, fails
):
    """Evict only weights; cache eviction errors must not prevent loading."""
    for name in ("weights.safetensors", "config.json"):
        (tmp_path / name).touch()
    fadvise = Mock(side_effect=OSError("cache eviction failed") if fails else None)
    monkeypatch.setattr("os.posix_fadvise", fadvise)
    drop_checkpoint_cache(str(tmp_path))
    fadvise.assert_called_once()
