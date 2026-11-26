# Quickstart & Usage Recipes

This guide walks through installing **irwkv**, preparing an RWKV-7 checkpoint, running offline Python inference, serving OpenAI-compatible APIs (including reasoning, tool calling, and structured outputs), and running benchmarks.

---

## 1. Environment & Installation

Create a project-local Python 3.12 virtual environment with [`uv`](https://docs.astral.sh/uv/) and install the pinned build and RWKV runtime dependencies:

```bash
git clone https://github.com/rwkv-org/irwkv.git
cd irwkv

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -r requirements/rwkv.txt --torch-backend=auto
uv pip install -e . --no-build-isolation --torch-backend=auto
```

`requirements/rwkv.txt` pins `torch==2.13.0`, `FlashRWKV2==0.2.0a4` from PyPI, and the verified `transformers-rwkv` and `tokenizers-rwkv` Git revisions.

> **CUDA required**: RWKV-7 execution in `irwkv` uses FlashRWKV2 fused CUDA/TileLang operators exclusively; there is no CPU, Triton, or PyTorch fallback.

---

## 2. Checkpoint Requirements

Point `irwkv` to a local RWKV-7 checkpoint directory produced by [transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv). A valid artifact directory contains:

- `config.json` with `model_type: "rwkv"`, `architecture_version: "rwkv7"`, `head_size: 64`, `intermediate_size == 4 * hidden_size`, and `tie_word_embeddings: false`.
- Safetensors weight files in canonical `transformers-rwkv` naming.
- Tokenizer files and the canonical RWKV chat template (`tokenizer_config.json` / `chat_template.jinja`).
- Three standalone Rapid-Sampling generation configs (`generation_config.json`, `fake_think_generation_config.json`, and `tools_generation_config.json`) without `_from_model_config: true`.

See the [RWKV-7 Architecture & Reference](../models/rwkv.md#references-and-weights) for official Hugging Face checkpoint repositories and naming conventions.

---

## 3. Offline Inference (Python)

### Completions (`LLM.generate`)

```python
from irwkv import LLM, SamplingParams

llm = LLM(
    model="/path/to/rwkv-checkpoint",
    dtype="bfloat16",
    enable_chunked_prefill=True,
    enable_prefix_caching=False,
)

outputs = llm.generate(
    [
        "Write a short sentence about a quiet river.",
        "Explain why constant-memory recurrence helps long-context serving.",
    ],
    SamplingParams(max_tokens=128, temperature=0.8, top_p=0.9),
)

for output in outputs:
    print(output.outputs[0].text)
```

### Chat (`LLM.chat` & `LLM.enqueue_chat`)

When `sampling_params` is omitted, `LLM.chat()` and `LLM.enqueue_chat()` automatically select the model artifact's Open Think, Fake Think, or Tools generation profile based on `tools` and `chat_template_kwargs`:

```python
from irwkv import LLM

llm = LLM(model="/path/to/rwkv-checkpoint", dtype="bfloat16")

messages = [{"role": "user", "content": "Summarize how RWKV-7 updates its state."}]

# Open Think (default profile: generation_config.json)
open_think = llm.chat(messages)
print(open_think[0].outputs[0].text)

# Fake Think (fake_think_generation_config.json, appends <think></think> prompt)
fake_think = llm.chat(
    messages,
    chat_template_kwargs={"fake_cot_mode": True},
)
print(fake_think[0].outputs[0].text)
```

If you explicitly pass `sampling_params`, those values are used as provided. Pass `generation_config="irwkv"` (or `--generation-config irwkv` on CLI) to disable artifact generation profiles completely.

---

## 4. OpenAI-Compatible Serving

Start the server with BF16 weights/activations and FP16 recurrent state (the default serving precision). For RL rollouts, evaluation, or sampling-parameter search, pass `--mamba-ssm-cache-dtype float32` to match training recurrence:

```bash
irwkv serve /path/to/rwkv-checkpoint \
  --served-model-name rwkv7 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

`irwkv serve` exposes standard OpenAI-compatible endpoints including `/v1/completions`, `/v1/chat/completions`, `/v1/responses`, and `/v1/models`.

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="rwkv7",
    messages=[{"role": "user", "content": "Write a haiku about autumn rain."}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

See `examples/basic/online_serving/openai_chat_completion_client.py` and `examples/basic/offline_inference/generate.py` in the repository for runnable example scripts.

---

## 5. Thinking Modes & Reasoning Parser

RWKV-7 artifacts use `<think>...</think>` delimiters. Enable `--reasoning-parser rwkv` to split reasoning tokens into the structured `reasoning` / `reasoning_content` field:

```bash
irwkv serve /path/to/rwkv-checkpoint \
  --served-model-name rwkv7 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --reasoning-parser rwkv
```

- **Open Think (default)**: Leave `chat_template_kwargs.fake_cot_mode` unset or `false`. Uses `generation_config.json`.
- **Fake Think**: Set `chat_template_kwargs.fake_cot_mode` to `true` to end the generation prompt with `<think></think` and select `fake_think_generation_config.json`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "chat_template_kwargs": {"fake_cot_mode": true},
    "max_tokens": 128
  }'
```

---

## 6. Tool Calling

The canonical `transformers-rwkv` chat template emits `**Tool Call:**` followed by a fenced JSON block (` ```json ... ``` `). Load the template directly from the model artifact (do not pass a copied template) and enable the native RWKV tool parser:

```bash
irwkv serve /path/to/rwkv-checkpoint \
  --served-model-name rwkv7 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv
```

Supplying `tools` in a chat request automatically selects `tools_generation_config.json` regardless of `fake_cot_mode`. The RWKV tool parser supports both streaming and non-streaming requests with `auto`, `required`, or named tool choices.

---

## 7. Structured Outputs (Grammar Fast Path)

`irwkv` supports JSON Schema and grammar-constrained decoding (`xgrammar`, `llguidance`, `lm-format-enforcer`, `outlines_core`). Unlike standard eager fallbacks, `RwkvSampler` captures **two separate CUDA Graphs** (`logits_graph` and `sample_graph`) and applies the grammar bitmask directly to the compacted logits buffer between replays, keeping full CUDA Graph acceleration for structured outputs.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7",
    "messages": [{"role": "user", "content": "Return a JSON object with city and country for Kyoto."}],
    "chat_template_kwargs": {"fake_cot_mode": true},
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location",
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"}
          },
          "required": ["city", "country"],
          "additionalProperties": false
        }
      }
    }
  }'
```

---

## 8. Benchmarking

Use the built-in `irwkv bench` CLI for standardized throughput and latency measurements:

```bash
# Inspect available benchmark subcommands (serve, throughput, latency, startup, sweep)
irwkv bench --help

# Online serving benchmark against a running `irwkv serve` instance
irwkv bench serve \
  --host 127.0.0.1 \
  --port 8000 \
  --model rwkv7 \
  --dataset-name random \
  --input-len 512 \
  --output-len 256 \
  --max-concurrency 64 \
  --num-prompts 256

# Use the native Rust benchmark client (lower client-side overhead at high concurrency)
IRWKV_USE_RUST_BENCH=1 irwkv bench serve \
  --host 127.0.0.1 \
  --port 8000 \
  --model rwkv7 \
  --dataset-name random \
  --input-len 512 \
  --output-len 256 \
  --max-concurrency 320 \
  --num-prompts 640
```

---

## Next Steps

- [Serving, State & Multi-GPU Guide](serving.md) — FP16 default vs FP32 parity state tradeoffs, DeltaLog budgeting, offline kernel dispatch, prefix caching, PP2/DP deployment, and the Rust frontend.
- [LoRA, NoRA & Online RL Sync](rl_and_lora.md) — unmerged PEFT adapters, versioned multi-tenant LoRA serving, and online weight reload.
- [RWKV-7 Architecture & Reference](../models/rwkv.md) — full decoding defaults, runtime constraints, environment variables, and official checkpoint links.
