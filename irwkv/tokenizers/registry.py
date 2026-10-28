# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import huggingface_hub
from typing_extensions import TypeVar

from irwkv import envs
from irwkv.logger import init_logger
from irwkv.transformers_utils.config import get_config
from irwkv.utils.import_utils import resolve_obj_by_qualname

from .hf import CachedHfTokenizer
from .protocol import TokenizerLike

if TYPE_CHECKING:
    from irwkv.config.model import ModelConfig, RunnerType

logger = init_logger(__name__)


_IRWKV_TOKENIZERS = {"hf": ("hf", "CachedHfTokenizer")}


@dataclass
class _TokenizerRegistry:
    # Tokenizer mode ->  (tokenizer module, tokenizer class)
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, tokenizer_mode: str, module: str, class_name: str) -> None:
        if tokenizer_mode in self.tokenizers:
            logger.warning(
                "%s.%s is already registered for tokenizer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                tokenizer_mode,
            )

        self.tokenizers[tokenizer_mode] = (module, class_name)

    def load_tokenizer_cls(self, tokenizer_mode: str) -> type[TokenizerLike]:
        if tokenizer_mode not in self.tokenizers:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        module, class_name = self.tokenizers[tokenizer_mode]
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_tokenizer(self, tokenizer_mode: str, *args, **kwargs) -> TokenizerLike:
        tokenizer_cls = self.load_tokenizer_cls(tokenizer_mode)
        return tokenizer_cls.from_pretrained(*args, **kwargs)


TokenizerRegistry = _TokenizerRegistry(
    {
        mode: (f"irwkv.tokenizers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _IRWKV_TOKENIZERS.items()
    }
)


def resolve_tokenizer_args(
    tokenizer_name: str | Path,
    *args,
    runner_type: "RunnerType" = "generate",
    tokenizer_mode: str = "auto",
    **kwargs,
):
    revision: str | None = kwargs.get("revision")
    download_dir: str | None = kwargs.get("download_dir")

    if envs.IRWKV_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circular import
        from irwkv.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not Path(tokenizer_name).exists():
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=str(tokenizer_name),
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
                )
                tokenizer_name = tokenizer_path

    if "truncation_side" not in kwargs:
        if runner_type == "generate":
            kwargs["truncation_side"] = "left"
        elif runner_type == "pooling":
            kwargs["truncation_side"] = "right"
        else:
            assert_never(runner_type)

    # Fallback to HF tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"

    return tokenizer_mode, tokenizer_name, args, kwargs


cached_resolve_tokenizer_args = lru_cache(resolve_tokenizer_args)


def tokenizer_args_from_config(config: "ModelConfig", **kwargs):
    return cached_resolve_tokenizer_args(
        config.tokenizer,
        runner_type="generate",
        tokenizer_mode=config.tokenizer_mode,
        revision=config.tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
        **kwargs,
    )


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_cls: type[_T] = TokenizerLike,  # type: ignore[assignment]
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> _T:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.IRWKV_USE_FASTOKENS:
        # Process-global, idempotent patch that swaps the Rust BPE backend
        # of any HF fast tokenizer loaded afterwards. No-op for non-HF modes.
        from .fastokens import apply_fastokens_patch

        apply_fastokens_patch()

    tokenizer_mode, tokenizer_name, args, kwargs = cached_resolve_tokenizer_args(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    if tokenizer_cls == TokenizerLike:
        tokenizer_cls_ = TokenizerRegistry.load_tokenizer_cls(tokenizer_mode)
    else:
        tokenizer_cls_ = tokenizer_cls

    # Register the model config before AutoTokenizer loads it.
    config_format = "hf" if tokenizer_cls_ is CachedHfTokenizer else "auto"
    config = None
    with contextlib.suppress(ValueError, OSError):
        config = get_config(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            config_format=config_format,
            token=kwargs.get("token"),
        )

    if config is not None and tokenizer_cls_ is CachedHfTokenizer:
        # AutoTokenizer otherwise reloads config.json internally. Reuse the
        # config that get_config just loaded successfully so a concurrent Hub
        # cache refresh cannot invalidate the file between the two reads.
        kwargs.setdefault("config", config)

    tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer  # type: ignore


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None

    return cached_get_tokenizer(
        model_config.tokenizer,
        runner_type="generate",
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )
