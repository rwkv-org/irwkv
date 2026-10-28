# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING

import torch

from irwkv import envs
from irwkv.logger import init_logger
from irwkv.utils.cpu_resource_utils import (
    DEVICE_CONTROL_ENV_VAR,
    get_memory_node_info,
    get_visible_memory_node,
)
from irwkv.utils.mem_constants import GiB_bytes
from irwkv.v1.attention.backends.registry import AttentionBackendEnum

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = init_logger(__name__)

if TYPE_CHECKING:
    from irwkv.config import IrwkvConfig
    from irwkv.v1.attention.selector import AttentionSelectorConfig
else:
    IrwkvConfig = None


def get_max_threads(pid=0):
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(pid))
    elif platform.system() == "Darwin":
        return os.cpu_count()
    else:
        raise NotImplementedError("Unsupported OS")


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"
    device_control_env_var = DEVICE_CONTROL_ENV_VAR

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.get_cpu_architecture() == CpuArchEnum.POWERPC:
            return [torch.bfloat16, torch.float32, torch.float16]
        elif self.get_cpu_architecture() == CpuArchEnum.ARM and sys.platform.startswith(
            "darwin"
        ):
            # sysctl exits non-zero when the OID is absent, so check_output would
            # raise instead of letting the fp16/fp32 fallback below run.
            bf16 = (
                subprocess.run(
                    ["sysctl", "-n", "hw.optional.arm.FEAT_BF16"], capture_output=True
                ).stdout.strip()
                == b"1"
            )
            if bf16:
                return [torch.bfloat16, torch.float16, torch.float32]
            return [torch.float16, torch.float32]
        elif self.get_cpu_architecture() == CpuArchEnum.RISCV:
            return [torch.bfloat16, torch.float16, torch.float32]
        # x86/aarch64 CPU has supported both bf16 and fp16 natively.
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def check_runner_kv_caches_multi_layer(cls) -> None:
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum | None",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend is None:
            selected_backend = next(
                (
                    backend
                    for backend in AttentionBackendEnum
                    if backend.value or backend.is_overridden()
                ),
                None,
            )
        if selected_backend is None:
            raise ValueError("No attention backend is registered")
        return selected_backend.get_path()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        meminfo = get_memory_node_info(device_id)

        return meminfo.total_memory

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device for the current platform."""
        torch.cpu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, irwkv_config: IrwkvConfig) -> None:
        model_config = irwkv_config.model_config

        if model_config is not None:
            model_config.disable_cascade_attn = True

        # Import lazily: irwkv.triton_utils imports irwkv.platforms.current_platform,
        # which is still being resolved while this platform class is loading.
        from irwkv.triton_utils import HAS_TRITON

        if cls.get_cpu_architecture() == CpuArchEnum.X86 and not HAS_TRITON:
            logger.warning_once(
                "Triton is not installed. triton-cpu is expected on x86 CPUs."
            )

        cache_config = irwkv_config.cache_config

        if not cache_config.user_specified_block_size:
            cache_config.block_size = 128

        # Legacy setting
        env_key = "IRWKV_CPU_KVCACHE_SPACE"
        if env_key in os.environ and os.environ[env_key] != "":
            kv_cache_space = int(os.environ[env_key])
            cache_config.kv_cache_memory_bytes = kv_cache_space * GiB_bytes

        scheduler_config = irwkv_config.scheduler_config
        # async scheduling is not required on CPU
        scheduler_config.async_scheduling = False

        parallel_config = irwkv_config.parallel_config
        if (
            os.environ.get("IRWKV_ENABLE_V1_MULTIPROCESSING", "1") == "1"
            and parallel_config.distributed_executor_backend == "uni"
        ):
            # OMP requires the MP executor to function correctly, UniProc
            # is not supported as it is not possible to set the OMP
            # environment correctly
            parallel_config.distributed_executor_backend = "mp"

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "irwkv.v1.worker.cpu_worker.CPUWorker"
        # Disable DBO
        if parallel_config.enable_dbo:
            logger.warning_once("Dual-Batch Overlap is not supported on CPU, disabled.")
            parallel_config.enable_dbo = False

        from irwkv.config import CompilationMode

        irwkv_config.compilation_config.cudagraph_capture_sizes = []

        compilation_config = irwkv_config.compilation_config
        if irwkv_config.compilation_config.mode == CompilationMode.IRWKV_COMPILE:
            # Note: irwkv V1 is using PIECEWISE level compilation, which will
            # take time to compile kernels just-in-time with the inductor
            # backend. For CPU CI tests, most of them are executed fast and
            # compilations consume too much time, even with torch compile
            # cache. So use IRWKV_CPU_CI_ENV to indicate the CI environment,
            # and just execute model with dynamo + eager mode to save time.
            # IRWKV_CPU_CI_ENV is only used as an internal variable.
            backend = "eager" if envs.IRWKV_CPU_CI_ENV else "inductor"

            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
            compilation_config.backend = backend
            compilation_config.inductor_compile_config.update(
                {
                    "dce": True,
                    "size_asserts": False,
                    "nan_asserts": False,
                    "epilogue_fusion": True,
                    "cpp.dynamic_threads": True,
                }
            )
            compilation_config.ir_enable_torch_wrap = False

        if (
            cls.get_cpu_architecture() == CpuArchEnum.ARM
            and "+gelu" not in compilation_config.custom_ops
            and "-gelu" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+gelu")
        if (
            cls.get_cpu_architecture() == CpuArchEnum.ARM
            and "+gelu_tanh" not in compilation_config.custom_ops
            and "-gelu_tanh" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+gelu_tanh")
        if (
            cls.get_cpu_architecture() == CpuArchEnum.ARM
            and "+gelu_and_mul" not in compilation_config.custom_ops
            and "-gelu_and_mul" not in compilation_config.custom_ops
        ):
            compilation_config.custom_ops.append("+gelu_and_mul")

        irwkv_config.profiler_config.torch_profiler_dump_cuda_time_total = False

        assert irwkv_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        os.environ["IRWKV_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Note: to avoid the error 'nthreads cannot be larger than environment
        # variable "NUMEXPR_MAX_THREADS" (64)'.
        os.environ["NUMEXPR_MAX_THREADS"] = str(get_max_threads())

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Disable multi-stream for shared experts as no Stream on CPU
        os.environ["IRWKV_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

        # Avoid inductor generates num_thread() and breaks the thread binding
        os.environ["TORCHINDUCTOR_CPP_DYNAMIC_THREADS"] = "1"

        ld_preload_str = os.getenv("LD_PRELOAD", "")
        cpu_architecture = Platform.get_cpu_architecture()

        if (
            platform.system() == "Linux"
            and cpu_architecture
            in (
                CpuArchEnum.ARM,
                CpuArchEnum.POWERPC,
                CpuArchEnum.X86,
                CpuArchEnum.S390X,
            )
            and not (
                "libomp" in ld_preload_str
                or "libgomp" in ld_preload_str
                or "libiomp" in ld_preload_str
            )
        ):
            # We need to LD_PRELOAD PyTorch's libgomp, otherwise only
            # one core will be properly utilized when we thread-bind
            # See: https://github.com/vllm-project/vllm/issues/27369
            # TODO: Remove once:
            # https://github.com/pytorch/pytorch/issues/166087 is fixed

            # We need to find the location of PyTorch's libgomp
            torch_pkg = os.path.dirname(torch.__file__)
            site_root = os.path.dirname(torch_pkg)
            # Search both torch.libs and torch/lib - See:
            # https://github.com/vllm-project/vllm/issues/30470
            torch_libs_paths = [
                os.path.join(site_root, "torch.libs"),
                os.path.join(torch_pkg, "lib"),
            ]
            pytorch_libgomp_so_candidates = []
            for torch_libs in torch_libs_paths:
                pytorch_libgomp_so_candidates.extend(
                    glob.glob(os.path.join(torch_libs, "libgomp*.so*"))
                )
            if pytorch_libgomp_so_candidates:
                pytorch_libgomp_so = pytorch_libgomp_so_candidates[0]
                if ld_preload_str:
                    ld_preload_str += ":"
                ld_preload_str += pytorch_libgomp_so
                os.environ["LD_PRELOAD"] = ld_preload_str

        # LD_PRELOAD libtcmalloc, bundled under irwkv/libs to reduce
        # memory allocation overhead
        if (
            platform.system() == "Linux"
            and cpu_architecture
            in (CpuArchEnum.ARM, CpuArchEnum.X86, CpuArchEnum.S390X)
            and "libtcmalloc" not in ld_preload_str
        ):
            irwkv_pkg = os.path.dirname(os.path.dirname(__file__))
            tcmalloc_so = None
            for pattern in ("libtcmalloc_minimal*.so*", "libtcmalloc.so*"):
                tcmalloc_so_candidates = glob.glob(
                    os.path.join(irwkv_pkg, "libs", pattern)
                )
                if tcmalloc_so_candidates:
                    tcmalloc_so = tcmalloc_so_candidates[0]
                    break

            if tcmalloc_so is not None:
                if ld_preload_str:
                    ld_preload_str = f"{tcmalloc_so}:{ld_preload_str}"
                else:
                    ld_preload_str = tcmalloc_so
                os.environ["LD_PRELOAD"] = ld_preload_str

        os.environ["LOCAL_WORLD_SIZE"] = str(
            irwkv_config.parallel_config.tensor_parallel_size
        )

    @classmethod
    def update_block_size_for_backend(cls, irwkv_config: "IrwkvConfig") -> None:
        model_config = irwkv_config.model_config
        if model_config is None or not model_config.is_hybrid:
            return

        # reconcile attention and mamba page sizes
        backend_classes = cls._find_non_ssm_backends(irwkv_config)
        if not backend_classes:
            return

        pre_block_size = irwkv_config.cache_config.block_size
        cls._align_hybrid_block_size(irwkv_config, backend_classes[0])
        cls._check_aligned_block_size(irwkv_config, backend_classes, pre_block_size)

    @classmethod
    def discover_numa_topology(cls) -> list[list[int]]:
        """Discover NUMA topology and keep the last physical core of each numa
        into one core group list for nixl start_kv_load()
        """
        SYS_NODE = "/sys/devices/system/node"
        SYS_CPU = "/sys/devices/system/cpu"

        if not (os.path.exists(SYS_NODE) and os.path.exists(SYS_CPU)):
            return []

        use_highest_sibling = cls.get_cpu_architecture() == CpuArchEnum.X86
        core_rsv_for_kv = []
        for node in os.listdir(SYS_NODE):
            if not node.startswith("node") or not node[4:].isdigit():
                continue
            node_path = f"{SYS_NODE}/{node}"

            seen_phys = set()
            for cpu in os.listdir(node_path):
                if not cpu.startswith("cpu") or not cpu[3:].isdigit():
                    continue

                cpu_id = int(cpu[3:])
                # thread_siblings based on cpu_id
                path = f"{SYS_CPU}/cpu{cpu_id}/topology/thread_siblings_list"

                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            s = f.read()
                        cpus: list[int] = []
                        for part in s.strip().split(","):
                            if "-" in part:
                                a, b = map(int, part.split("-"))
                                cpus.extend(range(a, b + 1))
                            else:
                                cpus.append(int(part))
                        siblings = cpus if cpus else [cpu_id]
                    except (OSError, ValueError):
                        siblings = [cpu_id]
                else:
                    siblings = [cpu_id]

                phys = max(siblings) if use_highest_sibling else min(siblings)

                if phys not in seen_phys:
                    seen_phys.add(phys)

            if len(seen_phys) > 0:
                core_rsv_for_kv.append(list(seen_phys))

        return core_rsv_for_kv

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device specific communicator class for distributed communication."""
        return "irwkv.distributed.device_communicators.cpu_communicator.CpuCommunicator"  # noqa

    @classmethod
    def supports_structured_output(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.get_num_threads()

    @classmethod
    def register_triton_kernel_overrides(cls) -> None:
        """Fallback C++ implementations for core Triton kernels.

        Used when the Triton-CPU backend is unavailable. When Triton-CPU is
        present, the Triton kernels can run natively and the overrides
        would only shadow them, so registration is skipped.
        """
        from irwkv.triton_utils import HAS_TRITON

        if HAS_TRITON:
            return

        from irwkv.triton_utils.dispatcher import register_kernels
        from irwkv.utils import cpu_triton_utils as cpu_tl

        register_kernels(
            {
                "irwkv.v1.worker.block_table.ComputeSlotMappingKernel.kernel": (
                    cpu_tl._compute_slot_mapping_kernel_impl
                ),
            }
        )

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        allowed_mem_node_list = get_visible_memory_node()
        mem_status_list = [get_memory_node_info(i) for i in allowed_mem_node_list]
        memory_usage = 0
        for s in mem_status_list:
            memory_usage += s.total_memory - s.available_memory

        return memory_usage
