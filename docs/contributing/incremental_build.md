# Incremental Compilation Workflow

The RWKV build contains the `rwkv7_ops` and `cumem_allocator` CUDA extensions.
Use CMake to rebuild them after changing files in `csrc/` without reinstalling
the whole package.

## Prerequisites

Install the project from source in its local environment and make sure the CUDA
toolkit, CMake, and Ninja are available:

```console
uv venv --python 3.12
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -e . --torch-backend=auto
```

## Configure and build

From the repository root, configure a build directory using the project's
Python environment. Set `TORCH_CUDA_ARCH_LIST` for the GPU architecture being
built when it cannot be detected automatically.

```console
cmake -S . -B build/irwkv -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIRWKV_TARGET_DEVICE=cuda \
  -DIRWKV_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build build/irwkv --target rwkv7_ops cumem_allocator
cmake --install build/irwkv --component rwkv7_ops
cmake --install build/irwkv --component cumem_allocator
```

Run the relevant RWKV tests after rebuilding. FlashRWKV2 is an external
operator dependency and is not built by this workflow.
