# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING

from irwkv import envs
from irwkv.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group
from irwkv.utils.import_utils import resolve_obj_by_qualname

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def irwkv_version_matches_substr(substr: str) -> bool:
    """Check to see if the irwkv version matches a substring."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        irwkv_version = version("irwkv")
    except PackageNotFoundError as e:
        logger.warning(
            "The irwkv package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail."
        )
        raise e
    return substr in irwkv_version


def cuda_platform_plugin() -> str | None:
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        from irwkv.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: irwkv cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if irwkv is built with cpu too.
            # Otherwise, irwkv will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = (
                pynvml.nvmlDeviceGetCount() > 0
                and not irwkv_version_matches_substr("cpu")
            )
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug("CUDA platform is not available because no GPU is found.")
            if irwkv_version_matches_substr("cpu"):
                logger.debug(
                    "CUDA platform is not available because irwkv is built with CPU."
                )
            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: %s", str(e))
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "irwkv.platforms.cuda.CudaPlatform" if is_cuda else None


def cpu_platform_plugin() -> str | None:
    logger.debug("Checking if CPU platform is available.")
    is_cpu = envs.IRWKV_TARGET_DEVICE == "cpu"
    if is_cpu:
        logger.debug(
            "Confirmed CPU platform is available because "
            "IRWKV_TARGET_DEVICE is set to CPU."
        )
    else:
        try:
            is_cpu = irwkv_version_matches_substr("cpu")
            if is_cpu:
                logger.debug(
                    "Confirmed CPU platform is available because irwkv is built "
                    "with CPU."
                )
            if not is_cpu:
                import sys

                is_cpu = sys.platform.startswith("darwin")
                if is_cpu:
                    logger.debug(
                        "Confirmed CPU platform is available because the machine "
                        "is MacOS."
                    )
        except Exception as e:
            logger.debug("CPU platform is not available because: %s", str(e))

    if not is_cpu:
        return None

    return "irwkv.platforms.cpu.CpuPlatform"


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
    "cpu": cpu_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    # An explicit CPU target is authoritative. Native CPU-only CI jobs reuse
    # an accelerator wheel and can run on accelerator hosts, so probing every
    # plugin would otherwise activate both CPU and the host accelerator.
    if envs.IRWKV_TARGET_DEVICE == "cpu":
        cpu_platform_cls_qualname = cpu_platform_plugin()
        assert cpu_platform_cls_qualname is not None
        logger.debug("Explicitly selected CPU platform.")
        return cpu_platform_cls_qualname

    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated_plugins = []

    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            logger.debug(
                "Platform plugin %s failed during detection.",
                name,
                exc_info=True,
            )

    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys())
    )
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}"
        )
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("Platform plugin %s is activated", activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}"
        )
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()
        logger.debug(
            "Automatically detected platform %s.", activated_builtin_plugins[0]
        )
    else:
        platform_cls_qualname = "irwkv.platforms.interface.UnspecifiedPlatform"
        logger.debug("No platform detected, irwkv is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from irwkv.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `irwkv.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import irwkv`, some irwkv internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "Platform",
    "PlatformEnum",
    "current_platform",
    "CpuArchEnum",
    "_init_trace",
]
