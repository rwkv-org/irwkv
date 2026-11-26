# Contributing to irwkv

Follow the [vLLM contribution guide](https://docs.vllm.ai/en/latest/contributing/) and the repository's [agent instructions](../../AGENTS.md). `irwkv` focuses on RWKV-7 inference and serving while consuming [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2) strictly as an external operator dependency.

---

## Development Setup

Always use `uv` to manage the project-local `./.venv` (never system `python3` or bare `pip`):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -r requirements/rwkv.txt --torch-backend=auto
uv pip install -e . --no-build-isolation --torch-backend=auto
uv pip install -r requirements/lint.txt
pre-commit install
```

---

## Rebuilding Native Components

- **Rust frontend (`irwkv-rs`)**: rebuild after editing files under `rust/`:

```bash
.venv/bin/python tools/build_rust.py --release
```

- **CUDA extensions (`rwkv7_ops`, `cumem_allocator`)**: follow the [incremental compilation workflow](incremental_build.md) after editing files under `csrc/`.

---

## Testing & Linters

Install test dependencies from `requirements/test/cuda.in`, then run targeted RWKV tests and pre-commit checks:

```bash
# Install test dependencies
uv pip install -r requirements/test/cuda.in

# Run RWKV model & parser tests
.venv/bin/python -m pytest tests/models/rwkv/ tests/tool_parsers/test_rwkv_tool_parser.py -v

# Run linters and formatting checks (Ruff, mypy, typos, markdownlint, SPDX, etc.)
pre-commit run --all-files
```

For model-affecting changes, also run accuracy evaluations and `irwkv bench` measurements as described in [Testing RWKV model changes](model/tests.md).

---

## Additional Contributor Guides

- [RWKV-7 Architecture & Reference](../models/rwkv.md)
- [Model Runner V2 Design](../design/model_runner_v2.md)
- [Pull Request Review Guide](pr-review.md)
- [Profiling Guide](profiling.md)
- [Vulnerability Management](vulnerability_management.md)
