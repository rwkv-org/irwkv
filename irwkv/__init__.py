# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""irwkv: a high-throughput and memory-efficient inference engine for LLMs."""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import irwkv.env_override  # noqa: F401

MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "CompletionOutput": ".outputs:CompletionOutput",
    "RequestOutput": ".outputs:RequestOutput",
}

if typing.TYPE_CHECKING:
    from irwkv.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from irwkv.engine.async_llm_engine import AsyncLLMEngine
    from irwkv.engine.llm_engine import LLMEngine
    from irwkv.entrypoints.llm import LLM
    from irwkv.inputs import PromptType, TextPrompt, TokensPrompt
    from irwkv.model_executor.models import ModelRegistry
    from irwkv.outputs import CompletionOutput, RequestOutput
    from irwkv.sampling_params import SamplingParams
    from irwkv.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
]
