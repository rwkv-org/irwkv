# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import logging
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME

# Select irwkv release tags, excluding crate tags such as "proto-v0.3.0".
IRWKV_GIT_DESCRIBE_COMMAND = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--abbrev=40",
    "--match",
    "v[0-9]*",
]


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on irwkv,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "irwkv", "envs.py"))
rust_build = load_module_from_path(
    "rust_build", os.path.join(ROOT_DIR, "tools", "build_rust.py")
)

IRWKV_TARGET_DEVICE = envs.IRWKV_TARGET_DEVICE


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(
        int(os.getenv("IRWKV_DISABLE_SCCACHE", "0"))
    )


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=not is_freethreaded(), **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if _is_cuda() and CUDA_HOME is not None:
            try:
                nvcc_version = get_nvcc_cuda_version()
                if nvcc_version >= Version("11.2"):
                    # `nvcc_threads` is either the value of the NVCC_THREADS
                    # environment variable (if defined) or 1.
                    # when it is set, we reduce `num_jobs` to avoid
                    # overloading the system.
                    nvcc_threads = envs.NVCC_THREADS
                    if nvcc_threads is not None:
                        nvcc_threads = int(nvcc_threads)
                        logger.info(
                            "Using NVCC_THREADS=%d as the number of nvcc threads.",
                            nvcc_threads,
                        )
                    else:
                        nvcc_threads = 1
                    num_jobs = max(1, num_jobs // nvcc_threads)
            except Exception as e:
                logger.warning("Failed to get NVCC version: %s", e)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DIRWKV_TARGET_DEVICE={IRWKV_TARGET_DEVICE}",
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += [f"-DIRWKV_PYTHON_EXECUTABLE={sys.executable}"]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DIRWKV_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += [f"-DFETCHCONTENT_BASE_DIR={fc_base_dir}"]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += [f"-DNVCC_THREADS={nvcc_threads}"]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []
        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda() and CUDA_HOME is not None:
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("irwkv.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return IRWKV_TARGET_DEVICE == "cuda" and has_cuda


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_irwkv_version() -> str:
    # Allow overriding the version. This is useful to build platform-specific
    # wheels (e.g. CPU, TPU) without modifying the source.
    if env_version := os.getenv("IRWKV_VERSION_OVERRIDE"):
        print(
            f"Overriding IRWKV version with {env_version} from IRWKV_VERSION_OVERRIDE"
        )
        sep = "+" if "+" not in env_version else "."
        env_version += f"{sep}rwkv"
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(
            write_to="irwkv/_version.py",
            git_describe_command=IRWKV_GIT_DESCRIBE_COMMAND,
        )

    version = get_version(
        write_to="irwkv/_version.py",
        git_describe_command=IRWKV_GIT_DESCRIBE_COMMAND,
    )
    if not envs.IRWKV_SKIP_VERSION_SUFFIX:
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != envs.IRWKV_MAIN_CUDA_VERSION and "sdist" not in sys.argv:
            sep = "+" if "+" not in version else "."
            version += f"{sep}cu{cuda_version.replace('.', '')[:3]}"

    sep = "+" if "+" not in version else "."
    version += f"{sep}rwkv"

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    return _read_requirements("rwkv.txt")


if not _is_cuda():
    raise ValueError("irwkv requires CUDA")
ext_modules = [
    CMakeExtension(name="irwkv.cumem_allocator"),
    CMakeExtension(name="irwkv.rwkv7_ops"),
]


package_data = {
    "irwkv": [
        "py.typed",
        "libs/*.so*",
        "entrypoints/serve/instrumentator/static/*.js",
        "entrypoints/serve/instrumentator/static/*.css",
        "transformers_utils/chat_templates/*.jinja",
    ]
}


cmdclass = {"build_ext": cmake_build_ext}

# Resolve the Python version first because get_irwkv_version() may set
# SETUPTOOLS_SCM_PRETEND_VERSION, which the Rust version should inherit.
irwkv_version = get_irwkv_version()
rust_build.prepare_build_environment()
rust_extensions = rust_build.rust_extensions(optional=False)

setup(
    # static metadata should rather go in pyproject.toml
    version=irwkv_version,
    ext_modules=ext_modules,
    rust_extensions=rust_extensions,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets", "scipy", "plotly"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.3.3"],
        "instanttensor": ["instanttensor >= 0.1.9"],
        "runai": ["runai-model-streamer[s3,gcs,azure] >= 0.15.7"],
        # Optional deps for OpenTelemetry tracing
        "otel": [
            "opentelemetry-sdk>=1.26.0",
            "opentelemetry-api>=1.26.0",
            "opentelemetry-exporter-otlp>=1.26.0",
            "opentelemetry-semantic-conventions-ai>=0.4.1",
        ],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
