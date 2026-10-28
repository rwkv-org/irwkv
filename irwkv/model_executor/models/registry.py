# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of supported RWKV model architectures."""

import importlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import torch.nn as nn

from irwkv import envs
from irwkv.config import ModelConfig
from irwkv.logger import init_logger
from irwkv.logging_utils import logtime
from irwkv.utils.hashing import safe_hash

if TYPE_CHECKING:
    from irwkv.config.model import AttnTypeStr
else:
    AttnTypeStr = Any


from .interfaces import (
    has_inner_state,
    has_noops,
    is_attention_free,
    is_hybrid,
    requires_raw_input_tokens,
    supports_mamba_prefix_caching,
    supports_multimodal,
    supports_multimodal_encoder_tp_data,
    supports_multimodal_raw_input_only,
    supports_pp,
    supports_transcription,
)
from .interfaces_base import (
    get_attn_type,
    is_text_generation_model,
)

logger = init_logger(__name__)


_IRWKV_MODELS = {"RwkvForCausalLM": ("rwkv", "RwkvForCausalLM")}

# This variable is used as the args for subprocess.run(). We
# can modify  this variable to alter the args if needed. e.g.
# when we use par format to pack things together, sys.executable
# might not be the target we want to run.
_SUBPROCESS_COMMAND = [sys.executable, "-m", "irwkv.model_executor.models.registry"]


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    attn_type: AttnTypeStr
    supports_multimodal: bool
    supports_multimodal_raw_input_only: bool
    requires_raw_input_tokens: bool
    supports_multimodal_encoder_tp_data: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    has_noops: bool
    supports_mamba_prefix_caching: bool
    supports_transcription: bool
    supports_transcription_only: bool
    supported_video_pruning_methods: tuple[str, ...]
    supports_mm_device_do_normalize: bool

    @staticmethod
    def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            architecture=model.__name__,
            is_text_generation_model=is_text_generation_model(model),
            attn_type=get_attn_type(model),
            supports_multimodal=supports_multimodal(model),
            supports_multimodal_raw_input_only=supports_multimodal_raw_input_only(
                model
            ),
            requires_raw_input_tokens=requires_raw_input_tokens(model),
            supports_multimodal_encoder_tp_data=supports_multimodal_encoder_tp_data(
                model
            ),
            supports_pp=supports_pp(model),
            has_inner_state=has_inner_state(model),
            is_attention_free=is_attention_free(model),
            is_hybrid=is_hybrid(model),
            supports_mamba_prefix_caching=supports_mamba_prefix_caching(model),
            supports_transcription=supports_transcription(model),
            supports_transcription_only=(
                supports_transcription(model) and model.supports_transcription_only
            ),
            has_noops=has_noops(model),
            supported_video_pruning_methods=getattr(
                model, "supported_video_pruning_methods", ()
            ),
            supports_mm_device_do_normalize=getattr(
                model, "supports_mm_device_do_normalize", False
            ),
        )


class _BaseRegisteredModel(ABC):
    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """Represents a model that has already been imported in the main process."""

    interfaces: _ModelInfo
    model_cls: type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """Represents a model that has not been imported in the main process."""

    module_name: str
    class_name: str

    @staticmethod
    def _get_cache_dir() -> Path:
        return Path(envs.IRWKV_CACHE_ROOT) / "modelinfos"

    def _get_cache_filename(self) -> str:
        cls_name = f"{self.module_name}-{self.class_name}".replace(".", "-")
        return f"{cls_name}.json"

    @staticmethod
    def _get_modelinfo_module_hash(model_path: Path) -> str:
        if model_path.name == "__init__.py":
            # Package entry points often re-export classes implemented in
            # submodules, so include the package contents in the cache key.
            module_paths = sorted(model_path.parent.rglob("*.py"))
            root_path = model_path.parent
        else:
            module_paths = [model_path]
            root_path = model_path.parent

        hasher = safe_hash(b"", usedforsecurity=False)
        for path in module_paths:
            hasher.update(path.relative_to(root_path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
        return hasher.hexdigest()

    def _load_modelinfo_from_cache(self, module_hash: str) -> _ModelInfo | None:
        try:
            try:
                modelinfo_path = self._get_cache_dir() / self._get_cache_filename()
                with open(modelinfo_path, encoding="utf-8") as file:
                    mi_dict = json.load(file)
            except FileNotFoundError:
                logger.debug(
                    "Cached model info file for class %s.%s not found",
                    self.module_name,
                    self.class_name,
                )
                return None

            if mi_dict["hash"] != module_hash:
                logger.debug(
                    "Cached model info file for class %s.%s is stale",
                    self.module_name,
                    self.class_name,
                )
                return None

            # file not changed, use cached _ModelInfo properties
            return _ModelInfo(**mi_dict["modelinfo"])
        except Exception:
            logger.debug(
                "Cached model info for class %s.%s error. ",
                self.module_name,
                self.class_name,
            )
            return None

    def _save_modelinfo_to_cache(self, mi: _ModelInfo, module_hash: str) -> None:
        """Save dictionary json file to cache."""
        from irwkv.model_executor.model_loader.weight_utils import atomic_writer

        try:
            modelinfo_dict = {
                "hash": module_hash,
                "modelinfo": asdict(mi),
            }
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            modelinfo_path = cache_dir / self._get_cache_filename()
            with atomic_writer(modelinfo_path, encoding="utf-8") as f:
                json.dump(modelinfo_dict, f, indent=2)
        except Exception:
            logger.exception("Error saving model info cache.")

    @logtime(logger=logger, msg="Registry inspect model class")
    def inspect_model_cls(self) -> _ModelInfo:
        # Modules registered with a non-default location (e.g. the
        # hardware-isolated ``irwkv.models.<name>`` layout) live outside
        # ``irwkv/model_executor/models``. Resolve the module spec directly
        # so the file-hash cache stays warm for them.
        model_path: Path | None = None
        if self.module_name.startswith("irwkv.model_executor.models."):
            model_path = Path(__file__).parent / f"{self.module_name.split('.')[-1]}.py"
        if model_path is None or not model_path.exists():
            try:
                spec = importlib.util.find_spec(self.module_name)
            except (ImportError, ValueError):
                spec = None
            model_path = Path(spec.origin) if spec is not None and spec.origin else None
        module_hash = None

        if model_path is not None and model_path.exists():
            module_hash = self._get_modelinfo_module_hash(model_path)

            mi = self._load_modelinfo_from_cache(module_hash)
            if mi is not None:
                logger.debug(
                    "Loaded model info for class %s.%s from cache",
                    self.module_name,
                    self.class_name,
                )
                return mi
            else:
                logger.debug(
                    "Cache model info for class %s.%s miss. Loading model instead.",
                    self.module_name,
                    self.class_name,
                )

        # Performed in another process to avoid initializing CUDA
        mi = _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls())
        )
        logger.debug(
            "Loaded model info for class %s.%s", self.module_name, self.class_name
        )

        # save cache file
        if module_hash is not None:
            self._save_modelinfo_to_cache(mi, module_hash)

        return mi

    def load_model_cls(self) -> type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    from irwkv.platforms import current_platform

    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> _ModelInfo | None:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'", model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        """Register an external model to be used in irwkv.

        `model_cls` can be either:

        - A [`torch.nn.Module`][] class directly referencing the model.
        - A string in the format `<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if not isinstance(model_arch, str):
            msg = f"`model_arch` should be a string, not a {type(model_arch)}"
            raise TypeError(msg)

        if model_arch != "RwkvForCausalLM":
            raise ValueError(f"Unsupported model architecture: {model_arch}")

        if model_arch in self.models:
            logger.debug(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls,
            )

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
            model = _RegisteredModel.from_model_cls(model_cls)
        else:
            msg = (
                "`model_cls` should be a string or PyTorch model class, "
                f"not a {type(model_cls)}"
            )
            raise TypeError(msg)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: list[str]):
        if any(arch in self.models for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed to load. "
                "Please check the logs for more details."
            )
        raise ValueError(
            f"Model architectures {architectures} are not supported. "
            f"Supported architectures: {self.get_supported_archs()}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> _ModelInfo | None:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def inspect_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[_ModelInfo, str]:
        del model_config
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        for arch in architectures:
            result = self._try_inspect_model_cls(arch)
            if result is not None:
                return result, arch
        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> tuple[type[nn.Module], str]:
        del model_config
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("No model architectures are specified")
        for arch in architectures:
            result = self._try_load_model_cls(arch)
            if result is not None:
                return result, arch
        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_text_generation_model

    def is_multimodal_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_multimodal

    def is_multimodal_raw_input_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_multimodal_raw_input_only

    def is_pp_supported_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_pp

    def model_has_inner_state(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.has_inner_state

    def is_attention_free_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_attention_free

    def is_hybrid_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.is_hybrid

    def is_noops_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.has_noops

    def is_transcription_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_transcription

    def is_transcription_only_model(
        self,
        architectures: str | list[str],
        model_config: ModelConfig,
    ) -> bool:
        model_info, _ = self.inspect_model_cls(architectures, model_config)
        return model_info.supports_transcription_only


def _resolve_module_name(mod_relname: str) -> str:
    if mod_relname.startswith("irwkv."):
        return mod_relname
    return f"irwkv.model_executor.models.{mod_relname}"


ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=_resolve_module_name(mod_relname),
            class_name=cls_name,
        )
        for model_arch, (mod_relname, cls_name) in _IRWKV_MODELS.items()
    }
)

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        import cloudpickle

        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(
            _SUBPROCESS_COMMAND, input=input_bytes, capture_output=True
        )

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(
                f"Error raised in subprocess:\n{returned.stderr.decode()}"
            ) from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


def _run() -> None:
    # Setup plugins
    from irwkv.plugins import load_general_plugins

    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
