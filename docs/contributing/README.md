# Contributing to irwkv

Follow the [vLLM contribution guide](https://docs.vllm.ai/en/latest/contributing/) and the repository's [agent instructions](../../AGENTS.md). The project supports RWKV inference and uses FlashRWKV2 as an external operator dependency.

## Development setup

Create and use the repository's Python environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -e . --no-build-isolation
uv pip install -r requirements/lint.txt
pre-commit install
```

Rebuild the Rust frontend after Rust changes:

```bash
.venv/bin/python tools/build_rust.py --release
```

For CUDA source changes, see [incremental builds](incremental_build.md). Run the relevant tests and formatting checks for each change. The [RWKV guide](../models/rwkv.md) describes model setup and inference.
