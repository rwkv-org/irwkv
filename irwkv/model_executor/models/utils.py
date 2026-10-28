# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import regex as re
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_module_registration_hook

from irwkv.logger import init_logger
from irwkv.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from irwkv.model_executor.model_loader.weight_utils import default_weight_loader
from irwkv.multimodal import NestedTensors
from irwkv.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers.conversion_mapping import WeightRenaming


logger = init_logger(__name__)

ShardId: TypeAlias = str | int | tuple[int, ...]
"""One shard of a stacked parameter. A tuple is a single contiguous span."""
ShardIds: TypeAlias = ShardId | list[ShardId]
"""One shard, or a list of shards the same weight is loaded into in turn."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns.

    If a key maps to a value of `None`, the corresponding weight is ignored."""

    orig_to_new_renaming: list["WeightRenaming"] = field(default_factory=list)
    orig_to_new_regex: Mapping[re.Pattern, str | None] = field(default_factory=dict)
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_stacked: Mapping[str, tuple[str, ShardIds]] = field(
        default_factory=dict
    )
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_renaming=[
                *self.orig_to_new_renaming,
                *other.orig_to_new_renaming,
            ],
            orig_to_new_regex={**self.orig_to_new_regex, **other.orig_to_new_regex},
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_stacked={
                **self.orig_to_new_stacked,
                **other.orig_to_new_stacked,
            },
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def map_name(self, key: str) -> str | None:
        """Map a weight name; returns ``None`` if the weight should be ignored."""
        result = self._map_name_with_shard(key)
        return result[0] if result is not None else None

    def _map_name(self, key: str) -> str | None:
        """Backward-compatible alias for :meth:`map_name`."""
        return self.map_name(key)

    def _map_name_with_shard(self, key: str) -> tuple[str, ShardIds | None] | None:
        """Map a weight name and extract any shard_id metadata.

        Returns:
            (mapped_name, shard_id) if the name should be kept. A list of shard
            ids means the weight is loaded into each of those shards in turn.
            None if the name should be dropped.

        """
        # Deprecation warnings
        if key.endswith(".kv_scale"):
            logger.warning_once(
                "DEPRECATED. Found kv_scale in the checkpoint. "
                "This format is deprecated in favor of separate k_scale and "
                "v_scale tensors and will be removed in a future release. "
                "Functionally, we will remap kv_scale to k_scale and duplicate "
                "k_scale to v_scale"
            )

        for renaming in self.orig_to_new_renaming:
            key, _ = renaming.rename_source_key(key)

        for pattern, new_key in self.orig_to_new_regex.items():
            if pattern.search(key):
                if new_key is None:
                    return None

                key = pattern.sub(new_key, key)

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        shard_id: ShardIds | None = None
        for substr, (new_key, new_shard_id) in self.orig_to_new_stacked.items():
            if substr in key:
                key = key.replace(substr, new_key, 1)
                shard_id = new_shard_id

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key, shard_id

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, data in weights:
            result = self._map_name_with_shard(name)
            if result is None:
                continue
            out_name, shard_id = result
            if shard_id is None:
                yield out_name, data
                continue
            shard_ids = shard_id if isinstance(shard_id, list) else [shard_id]
            for i, one_shard_id in enumerate(shard_ids):
                # Each shard carries its own id, so extra shards need their own
                # tensor object; `detach` aliases the storage instead of copying.
                shard_data = data if i == 0 else data.detach()
                shard_data.shard_id = one_shard_id
                yield out_name, shard_data

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }

    def get_rename_mapper(self) -> "WeightsMapper":
        """Mapper variant keeping only the renames.

        This is what consumers that *name* modules rather than load them need:
        LoRA name parsing and the quantization config's layer lists.

        Stacked maps are dropped so that constituent names (e.g. `q_proj`) survive
        rather than being rewritten to the stacked irwkv name (`qkv_proj`). Mappings to
        `None` are dropped because "do not load this weight" is meaningless to such a
        consumer, and applying it would silently shrink a quantization config's ignore
        list or make LoRA name parsing fail."""
        remove_none = lambda d: {k: v for k, v in d.items() if v is not None}
        return replace(
            self,
            orig_to_new_regex=remove_none(self.orig_to_new_regex),
            orig_to_new_substr=remove_none(self.orig_to_new_substr),
            orig_to_new_stacked={},
            orig_to_new_prefix=remove_none(self.orig_to_new_prefix),
            orig_to_new_suffix=remove_none(self.orig_to_new_suffix),
        )


def _get_tied_embedding_params(module: nn.Module) -> dict[str, str]:
    """Map each tied word embedding qualname to the first name it aliases."""
    canonical = dict[int, str]()
    aliased = dict[str, str]()
    for prefix, submodule in module.named_modules(remove_duplicate=False):
        if not isinstance(submodule, VocabParallelEmbedding):
            continue
        for name, param in submodule.named_parameters(remove_duplicate=False):
            qualname = f"{prefix}.{name}" if prefix else name
            if (first_name := canonical.setdefault(id(param), qualname)) != qualname:
                aliased[qualname] = first_name
    return aliased


class AutoWeightsLoader:
    """Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting the
    environment variable `IRWKV_LOGGING_LEVEL=DEBUG`.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []

        # Weight tying makes two qualnames point at the same `nn.Parameter`
        # (e.g. `lm_head.weight` and `model.embed_tokens.weight`). Loading both
        # would write the same storage twice, so only the first name reached by
        # module traversal is loaded and the rest are skipped.
        self.aliased_params = _get_tied_embedding_params(module)
        self._skipped_aliases = dict[str, str]()
        self._loaded_params_are_complete = True

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return qualname in self.aliased_params

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            if param.dtype != weight_data.dtype:
                logger.warning(
                    "Attempted to load weight %s with dtype %s into "
                    "parameter with dtype %s",
                    weight_qualname,
                    weight_data.dtype,
                    param.dtype,
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(
        self, module: nn.Module, child_params: dict[str, torch.Tensor]
    ):
        """Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats and registered buffers.
        """
        # Add persistent registered buffers.
        # Non-persistent buffers are excluded, matching PyTorch state_dict().
        non_persistent: set[str] = getattr(module, "_non_persistent_buffers_set", set())
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf_name not in child_params and buf_name not in non_persistent:
                child_params[buf_name] = buf

        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                    self._loaded_params_are_complete = False
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                if self._can_skip(prefix):
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                named_parameters = module.named_parameters(recurse=True)
                desc_param_keys = {
                    maybe_prefix(base_prefix, k) for k, _ in named_parameters
                }
                msg = (
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module._get_name()}. "
                    f"The available parameters belonging to {base_prefix} "
                    f"({module._get_name()}) are: {desc_param_keys}"
                )
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        mapper = mapper or WeightsMapper()

        weights = mapper.apply(weights)
        weights = self._filter_skipped(weights)

        autoloaded_weights = set(self._load_module("", self.module, weights))
        self._check_skipped_aliases(autoloaded_weights)
        return autoloaded_weights

    def _filter_skipped(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            if (canonical := self.aliased_params.get(name)) is not None:
                self._skipped_aliases[name] = canonical
            if self._can_skip(name):
                continue

            yield name, weight

    def _check_skipped_aliases(self, autoloaded_weights: set[str]) -> None:
        """Guard against skipping an alias whose canonical name never loads."""
        if not self._loaded_params_are_complete:
            return
        for alias, canonical in self._skipped_aliases.items():
            if canonical not in autoloaded_weights:
                raise ValueError(
                    f"{alias!r} was skipped because it is tied to {canonical!r} "
                    f"in {self.module._get_name()}, but {canonical!r} was not "
                    "found in the checkpoint, so the tied weight is "
                    "uninitialized."
                )


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """
    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """
    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.

    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # If is_multimodal is on CPU this avoids a D2H sync
        inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during index put operation") from e

    return inputs_embeds


class StageMissingLayer(nn.Module):
    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None:
        super().__init__()

        self.stage_name = stage_name

        # Don't register this as a child module in order to
        # avoid missing keys when loading weights
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        return getattr(self.__dict__["module"], name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self} should not be called")

    def extra_repr(self) -> str:
        return f"stage_name={self.stage_name!r}"


@contextmanager
def collect_children(
    module: nn.Module,
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """Within this context, collect all direct child assignments to `module`,
    returning a list of children names that is internally updated until the
    context is exited.

    If `targets` is set, instead collect descendents of `module`
    that are an instance of `targets`, even if they aren't direct children.
    """
    children_names = list[str]()

    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                children_names.append(name)

        with register_module_module_registration_hook(hook):
            yield children_names
    else:
        yield children_names

        for name, module_ in module.named_modules():
            if isinstance(module_, targets):
                children_names.append(name)


@contextmanager
def no_init_weights(
    module: nn.Module,
    placeholder: Callable[[nn.Module], nn.Module],
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """Within this context, prevent weight initialization from using device memory and
    replace direct child assignments to `module` with the result of `placeholder()`.

    If `targets` is set, instead prevent weight initialization and
    replace assignments where the child is an instance of `targets`,
    even if they aren't direct children of `module`.
    """
    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                return placeholder(submodule)

            return submodule

        with register_module_module_registration_hook(hook), torch.device("meta"):
            yield
    else:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if isinstance(module_, targets):
                submodule.to("meta")  # Free memory
            if isinstance(submodule, targets):
                submodule.to("meta")  # Free memory
                return placeholder(submodule)

            return submodule

        # Not all descendents are targeted, so we can't use a blanket
        # `torch.device("meta")` context
        with register_module_module_registration_hook(hook):
            yield


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """A placeholder layer for missing layers in a pipeline parallel model."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.

    Args:
        num_hidden_layers: Total number of hidden layers in the model.
        layer_fn: Function to create a layer given its index.
        prefix: Prefix for layer names.

    Returns:
        Tuple of (start_layer, end_layer, modules).

    """
    from irwkv.distributed.parallel_state import get_pp_group
    from irwkv.distributed.utils import get_pp_indices
    from irwkv.model_executor.offloader import get_offloader

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )

    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
        )
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )

    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".

    """
    return name if not prefix else f"{prefix}.{name}"


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """Extract the layer index from the module name.

    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1

    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    else:
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.


def scatter_output_slices(
    output: torch.Tensor,
    indices: list[int],
    per_item_out_tokens: list[int],
    dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
    clone: bool = False,
) -> None:
    """Slice a concatenated output tensor and scatter into dest by index."""
    offset = 0
    for idx in indices:
        n_tok = per_item_out_tokens[idx]
        sliced = output[offset : offset + n_tok]
        dest[idx] = sliced.clone() if clone else sliced
        offset += n_tok
