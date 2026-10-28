cmake_minimum_required(VERSION 3.26)

# When building directly using CMake, make sure you run the install step
# (it places the .so files in the correct location).
#
# Example, from the repository root:
# cmake -S . -B build/irwkv -G Ninja -DIRWKV_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" -DCMAKE_INSTALL_PREFIX="$PWD"
# cmake --build build/irwkv --target install
#
# If you want to only build one target, make sure to install it manually:
# cmake --build build/irwkv --target rwkv7_ops
# cmake --install build/irwkv --component rwkv7_ops
project(irwkv_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# PyTorch headers require C++20; GCC < 11.3 has incomplete C++20 support.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.3")
  message(FATAL_ERROR
    "GCC >= 11.3 is required to build irwkv (found ${CMAKE_CXX_COMPILER_VERSION}). "
    "PyTorch's C++20 headers require a compiler with full C++20 support. "
    "See: https://github.com/pytorch/pytorch/pull/167929")
endif()


# CUDA by default, can be overridden by using -DIRWKV_TARGET_DEVICE=... (used by setup.py)
set(IRWKV_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for irwkv")
if(NOT IRWKV_TARGET_DEVICE STREQUAL "cuda")
  message(FATAL_ERROR "irwkv requires IRWKV_TARGET_DEVICE='cuda'")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${IRWKV_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${IRWKV_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.11" "3.12" "3.13" "3.14")

set(TORCH_SUPPORTED_VERSION_CUDA "2.13.0")
# TORCH_NIGHTLY=1 builds run against unpinned nightly wheels, so the supported-
# version check would always warn. Only treat it as a nightly build when the
# value is exactly "1" (the bootstrap exports TORCH_NIGHTLY=0 by default, which
# must NOT suppress the warning for normal builds).
if (DEFINED ENV{TORCH_NIGHTLY} AND "$ENV{TORCH_NIGHTLY}" STREQUAL "1")
  set(TORCH_NIGHTLY_BUILD TRUE)
else()
  set(TORCH_NIGHTLY_BUILD FALSE)
endif()

#
# Try to find python package with an executable that exactly matches
# `IRWKV_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (IRWKV_PYTHON_EXECUTABLE)
  find_python_from_executable(${IRWKV_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set IRWKV_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure the 'nvcc' command is in the PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)

# Supported NVIDIA architectures.
# This check must happen after find_package(Torch) because that's when CMAKE_CUDA_COMPILER_VERSION gets defined
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4)
  # Rubin (10.7) can run SM100 family code, but CUDA 13.4 also supports
  # targeting it directly.
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.7;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
   # starting from CUDA 12.9 and Blackwell (10.0), we use family-specific targets (10.0f, 12.0f, etc)
   # to support the whole generation without specifying all sub-architectures
   # see: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

#
# Set up GPU language and check the torch version and warn if it isn't
# what is expected.
#
if(NOT CUDA_FOUND OR HIP_FOUND OR PYTORCH_FOUND_HIP)
  message(FATAL_ERROR "CUDA PyTorch is required.")
endif()
set(IRWKV_GPU_LANG "CUDA")
if(NOT TORCH_NIGHTLY_BUILD AND NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
  message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} expected for CUDA build, saw ${Torch_VERSION} instead.")
endif()

if(IRWKV_GPU_LANG STREQUAL "CUDA")
  #
  # For cuda we want to be able to control which architectures we compile for on
  # a per-file basis in order to cut down on compile time. So here we extract
  # the set of architectures we want to compile for and remove the from the
  # CMAKE_CUDA_FLAGS so that they are not applied globally.
  #
  # `+PTX` in TORCH_CUDA_ARCH_LIST is not preserved here. If a kernel really
  # needs PTX, add `+PTX` to that kernel's component-specific arch list below.
  #
  clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
  warn_if_ptx_arch_requested("${CUDA_ARCH_FLAGS}")
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # Filter the target architectures by the supported supported archs
  # since for some files we will build for all CUDA_ARCHS.
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
  if(NOT CUDA_ARCHS)
    message(FATAL_ERROR
      "No supported CUDA architectures; the build would produce a binary "
      "with no usable kernels. Detected gencode flags: ${CUDA_ARCH_FLAGS}; "
      "supported: ${CUDA_SUPPORTED_ARCHS}. "
      "Set TORCH_CUDA_ARCH_LIST for your GPU (e.g. 12.0).")
  endif()

endif()

#
# Query torch for additional GPU compilation flags for the given
# `IRWKV_GPU_LANG`.
# The final set of arches is stored in `IRWKV_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(IRWKV_GPU_FLAGS ${IRWKV_GPU_LANG})

#
# Set nvcc parallelism.
#
if(NVCC_THREADS AND IRWKV_GPU_LANG STREQUAL "CUDA")
  list(APPEND IRWKV_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

#
# Set compression mode for CUDA >=13.x.
#
if(IRWKV_GPU_LANG STREQUAL "CUDA" AND
   DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND IRWKV_GPU_FLAGS "--compress-mode=size")
endif()

#
# Set CUDA include flags for CXX compiler.
#
if(IRWKV_GPU_LANG STREQUAL "CUDA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
  if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
  endif()
endif()

#
# Define other extension targets
#

#
# cumem_allocator extension
#

set(IRWKV_CUMEM_EXT_SRC
  "csrc/cumem_allocator.cpp")

set_gencode_flags_for_srcs(
  SRCS "${IRWKV_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

define_extension_target(
  cumem_allocator
  DESTINATION irwkv
  LANGUAGE CXX
  SOURCES ${IRWKV_CUMEM_EXT_SRC}
  LIBRARIES CUDA::cuda_driver
  USE_SABI 3.8
  WITH_SOABI)

set(RWKV7_RUNTIME_CUDA_SRC "csrc/libtorch_stable/cuda_view.cu")
set(RWKV7_RUNTIME_SRC
  ${RWKV7_RUNTIME_CUDA_SRC}
  "csrc/libtorch_stable/rwkv7/rwkv7_registration.cpp")
set_gencode_flags_for_srcs(
  SRCS "${RWKV7_RUNTIME_CUDA_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")
define_extension_target(
  rwkv7_ops
  DESTINATION irwkv
  LANGUAGE CUDA
  SOURCES ${RWKV7_RUNTIME_SRC}
  COMPILE_FLAGS ${IRWKV_GPU_FLAGS}
  ARCHITECTURES ${IRWKV_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
target_compile_definitions(rwkv7_ops PRIVATE
  TORCH_TARGET_VERSION=0x020B000000000000ULL
  USE_CUDA
  IRWKV_RWKV_RUNTIME_OPS)
