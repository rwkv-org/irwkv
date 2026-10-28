# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from irwkv.config.model import ModelDType, TokenizerMode


@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str
    """The default model to use for testing this architecture."""

    extras: Mapping[str, str] = field(default_factory=dict)
    """Extra models to use for testing this architecture."""

    tokenizer: str | None = None
    """Set the tokenizer to load for this architecture."""

    tokenizer_mode: TokenizerMode | str = "auto"
    """Set the tokenizer type for this architecture."""

    speculative_model: str | None = None
    """
    The default model to use for testing this architecture, which is only used
    for speculative decoding.
    """

    speculative_method: str | None = None
    """
    The method to use for speculative decoding.
    """

    min_transformers_version: str | None = None
    """
    The minimum version of HF Transformers that is required to run this model.
    """

    max_transformers_version: str | None = None
    """
    The maximum version of HF Transformers that this model runs on.
    """

    transformers_version_reason: dict[Literal["irwkv", "hf"], str] | None = None
    """
    The type and reason to skip test for the minimum/maximum version requirement.
    irwkv: skip all irwkv tests if the version requirement is not met.
    hf: only skip tests that uses HF runner if the version requirement is not met.
    """

    require_embed_inputs: bool = False
    """
    If `True`, enables prompt and multi-modal embedding inputs while
    disabling tokenization.
    """

    dtype: ModelDType = "auto"
    """
    The data type for the model weights and activations.
    """

    enforce_eager: bool = False
    """
    Whether to enforce eager execution. If True, we will
    disable CUDA graph and always execute the model in eager mode.
    If False, we will use CUDA graph and eager execution in hybrid.
    """

    enable_prefix_caching: bool = True
    """
    Whether to enable prefix caching for the model. If True, we will test the model with
    prefix caching enabled. If False, we will test the model without prefix caching.
    """

    is_available_online: bool = True
    """
    Set this to `False` if the name of this architecture no longer exists on
    the HF repo. To maintain backwards compatibility, we have not removed them
    from the main model registry, so without this flag the registry tests will
    fail.
    """

    trust_remote_code: bool = False
    """The `trust_remote_code` level required to load the model."""

    hf_overrides: dict[str, Any] = field(default_factory=dict)
    """The `hf_overrides` required to load the model."""

    max_model_len: int | None = None
    """
    The maximum model length to use for this model. Some models default to a
    length that is too large to fit into memory in CI.
    """

    max_num_batched_tokens: int | None = None
    """
    The maximum number of tokens to be processed in a single batch.
    """

    revision: str | None = None
    """
    The specific revision (commit hash, tag, or branch) to use for the model.
    If not specified, the default revision will be used.
    """

    max_num_seqs: int | None = None
    """Maximum number of sequences to be processed in a single iteration."""

    use_original_num_layers: bool = False
    """
    If True, use the original number of layers from the model config
    instead of minimal layers for testing.
    """

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"],
        check_version_reason: Literal["irwkv", "hf"] = "hf",
        check_min_version: bool = True,
        check_max_version: bool = True,
    ) -> str | None:
        """If the installed transformers version does not meet the requirements,
        perform the given action.
        """
        if (
            self.min_transformers_version is None
            and self.max_transformers_version is None
        ):
            return None

        current_version = TRANSFORMERS_VERSION
        cur_base_version = Version(current_version).base_version
        min_version = self.min_transformers_version
        max_version = self.max_transformers_version
        msg = f"`transformers=={current_version}` installed, but `transformers"
        # Only check the base version for the min/max version, otherwise preview
        # models cannot be run because `x.yy.0.dev0`<`x.yy.0`
        if min_version and Version(cur_base_version) < Version(min_version):
            is_version_valid = False
            should_check_version = check_min_version
            msg += f">={min_version}` is required to run this model."
        elif max_version and Version(cur_base_version) > Version(max_version):
            is_version_valid = False
            should_check_version = check_max_version
            msg += f"<={max_version}` is required to run this model."
        else:
            is_version_valid = True
            should_check_version = False

        # Reasons explain a known incompatibility with a violated version
        # bound. They should not skip models when the installed version is
        # already within the requested range.
        is_reason_applicable = (
            not is_version_valid
            and self.transformers_version_reason is not None
            and check_version_reason in self.transformers_version_reason
        )
        is_transformers_valid = is_version_valid or (
            not should_check_version and not is_reason_applicable
        )
        if is_transformers_valid:
            return None
        elif self.transformers_version_reason:
            for reason_type, reason in self.transformers_version_reason.items():
                msg += f" Reason({reason_type}): {reason}"

        if on_fail == "error":
            raise RuntimeError(msg)
        elif on_fail == "skip":
            pytest.skip(msg)

        return msg

    def check_available_online(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        """If the model is not available online, perform the given action."""
        if not self.is_available_online:
            msg = "Model is not available online"

            if on_fail == "error":
                raise RuntimeError(msg)
            else:
                pytest.skip(msg)


_EXAMPLE_MODELS = {
    "RwkvForCausalLM": _HfExamplesInfo(
        "rwkv-rs/rwkv7-g1-st",
        dtype="bfloat16",
    ),
}
_AUTOMATIC_CONVERTED_MODELS: dict[str, _HfExamplesInfo] = {}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> Set[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        try:
            return self.hf_models[model_arch]
        except KeyError:
            raise ValueError(
                f"No example model defined for {model_arch}; please update this file."
            ) from None

    def find_hf_info(self, model_id: str) -> _HfExamplesInfo:
        for info in self.hf_models.values():
            if info.default == model_id:
                return info

        # Fallback to extras
        for info in self.hf_models.values():
            if any(extra == model_id for extra in info.extras.values()):
                return info

        raise ValueError(
            f"No example model defined for {model_id}; please update this file."
        )


HF_EXAMPLE_MODELS = HfExampleModels(_EXAMPLE_MODELS)
AUTO_EXAMPLE_MODELS = HfExampleModels(_AUTOMATIC_CONVERTED_MODELS)
