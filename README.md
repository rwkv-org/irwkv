# irwkv

<div align="center">

**RWKV-7 inference and serving, powered by FlashRWKV2.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11–3.14-blue)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/backend-CUDA-76B900)](docs/models/rwkv.md)

[Quick start](#quick-start) · [Offline inference](#offline-inference) · [Serving](#serving) · [Documentation](#documentation)

</div>

irwkv is a CUDA inference and serving framework built for RWKV-7. It brings
continuous batching, recurrent-state management, and OpenAI-compatible APIs to
the fused operators in [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2).

## Highlights

- **Native RWKV-7 execution** — recurrent inference with per-request state
  management and chunked prefill.
- **GPU execution** — FlashRWKV2 fused operators, CUDA Graph capture, and
  asynchronous scheduling and detokenization.
- **Familiar APIs** — Python inference, streaming Completions, Chat Completions,
  and Responses, with structured outputs and tool calling.
- **Deployment options** — single-GPU inference, two-stage pipeline parallelism,
  and an optional Rust frontend.

## Quick start

### 1. Install from source

Use a CUDA-capable NVIDIA GPU with a CUDA toolkit and Rust toolchain installed.
The commands below create a dedicated Python 3.12 environment and install the
pinned build and runtime dependencies.

```bash
git clone https://github.com/rwkv-org/irwkv.git
cd irwkv

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -r requirements/rwkv.txt --torch-backend=auto
uv pip install -e . --no-build-isolation --torch-backend=auto
```

### 2. Prepare a checkpoint

Use a local RWKV-7 Transformers-format checkpoint produced by
[transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv), including its
safetensors weights, tokenizer, chat template, and generation configs. See the
[model guide](docs/models/rwkv.md) for the artifact format and
[checkpoint references](docs/models/rwkv.md#references-and-weights).

The examples below use `/path/to/rwkv-checkpoint` as the checkpoint directory.
On first startup, FlashRWKV2 prepares missing kernel tuning profiles and reuses
its persistent cache on subsequent runs. See
[kernel preparation](docs/models/rwkv.md#kernel-preparation) for details.

## Offline inference

```python
from irwkv import LLM, SamplingParams

llm = LLM(model="/path/to/rwkv-checkpoint", dtype="bfloat16")
outputs = llm.generate(
    ["Write a short sentence about a quiet river."],
    SamplingParams(max_tokens=64),
)

for output in outputs:
    print(output.outputs[0].text)
```

## Serving

Start an OpenAI-compatible server with BF16 activations and FP32 recurrent
state:

```bash
irwkv serve /path/to/rwkv-checkpoint \
  --served-model-name rwkv7 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

Send a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "messages": [
      {"role": "user", "content": "Write a short sentence about a quiet river."}
    ],
    "max_tokens": 64
  }'
```

Set `IRWKV_USE_RUST_FRONTEND=1` before starting the server to use the Rust
frontend. Explore the serving benchmark with `irwkv bench serve --help`.

## Documentation

| Guide | Topics |
| --- | --- |
| [RWKV-7 model guide](docs/models/rwkv.md) | Artifact format, precision, and serving configuration |
| [Kernel preparation](docs/models/rwkv.md#kernel-preparation) | Tuning profiles, persistent caches, and startup preparation |
| [Decoding defaults](docs/models/rwkv.md#decoding-defaults) | Generation profiles and sampling controls |
| [Pipeline parallelism](docs/models/rwkv.md#pipeline-parallelism) | Two-GPU deployment |
| [Tool calling](docs/models/rwkv.md#tool-calling) | Chat templates and the RWKV tool parser |

## License and acknowledgments

irwkv is distributed under the [Apache 2.0 license](LICENSE). The framework
derives from [vLLM](https://github.com/vllm-project/vllm), with original
copyright notices retained. RWKV operators are provided by
[FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2), and the Transformers artifact
format and tokenizer come from
[transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv).
