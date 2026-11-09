# RWKV-7 MTP speculative decoding

This tree adds an optional **greedy MTP speculative decoding** path for RWKV-7.
It is a local feature, not part of upstream irwkv.  The code lives in the patch
`RWKV-PEFT/patches/irwkv_rwkv_mtp.patch` (base commit `1f0bd68`).

MTP here = a LightMTP draft head + the base model as verifier.  One round emits
the verifier token plus every accepted draft token; a rejected draft rewinds the
recurrent state and the verifier's token is kept.  In exact arithmetic the final
output is **identical to plain greedy decoding**.

## How it works

- the sidecar (`mtp_config.json` + `mtp.safetensors`) is loaded from the model
  directory into `model.mtp_head`;
- `RwkvMTPSpeculator` drafts `depth` tokens with a CUDA-graph captured chain;
- one base-model forward verifies the draft chain;
- accepted prefix is kept, rejected suffix rewinds the per-layer recurrent state
  using the captured rewind buffers;
- the next round continues from the verifier's hidden state.

The rewind buffers are captured in `model_states/rwkv.py`; the `for_capture`
fix (limit 32) is required.  Without it, the verify CUDA graph captures without
checkpoints and the generated text is corrupted.

## Files (feature surface)

New:

- `irwkv/model_executor/models/rwkv_mtp.py` — sidecar config + MTP head
- `irwkv/v1/worker/gpu/rwkv_mtp_spec.py` — drafter, verify, state rewind

Modified:

- `irwkv/model_executor/models/config.py` — reads `IRWKV_RWKV_MTP_DEPTH`,
  validates PP=1 / chunked prefill / sidecar, disables async scheduling
- `irwkv/model_executor/models/rwkv.py` — loads the MTP head, spec buffers
- `irwkv/v1/attention/backends/rwkv_attn.py` — spec checkpoint/input buffers
- `irwkv/v1/core/sched/scheduler.py` — speculative verify step scheduling
- `irwkv/v1/outputs.py`, `irwkv/v1/request.py`, `irwkv/v1/worker/gpu/input_batch.py`
  — draft token / verify step plumbing
- `irwkv/v1/worker/gpu/model_runner.py` — `_mtp_greedy` gate, `_mtp_verify`
- `irwkv/v1/worker/gpu/model_states/rwkv.py` — rewind buffers + `for_capture`

## Requirements

- irwkv with the `mtp` branch (feature commit `a3b3a01`, parent `1f0bd68`) or
  upstream `1f0bd68` plus `RWKV-PEFT/patches/irwkv_rwkv_mtp.patch`;
- **FlashRWKV2 (tilerwkv) 0.1.0a18** — irwkv imports `flashrwkv2` for the
  RWKV-7 kernels:
  ```bash
  git clone https://github.com/Joluck/tilerwkv.git
  cd tilerwkv
  uv pip install -e .
  ```
- pipeline parallel size **1**;
- chunked prefill enabled (default);
- an RWKV-7 base with a matching MTP sidecar;
- greedy requests; single-request verify steps.

## Model directory

The sidecar must live in the model directory; there is no separate `--mtp-path`.

```
model-dir/
  config.json
  tokenizer_config.json / tokenizer.json
  model-0000*-of-0000*.safetensors (+ index)
  mtp_config.json
  mtp.safetensors
```

Example `mtp_config.json` (the 7.2B tulu d4 head):

```json
{
  "mtp_type": "lightmlp",
  "mtp_depth": 4,
  "mtp_loop": true,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "norm_eps": 1e-05,
  "num_blocks": 1,
  "shared_lm_head": true,
  "num_parameters": 209747968,
  "source_checkpoint": ".../rwkv-0-step1000-mtp.pth",
  "weight_file": "mtp.safetensors"
}
```

`mtp_depth` must equal `IRWKV_RWKV_MTP_DEPTH` and the training depth.  If the
runtime env is larger, `model_config.hf_config.rwkv_mtp_depth` is clamped to the
sidecar's `mtp_depth`.

## Quick start: serve

```bash
CUDA_VISIBLE_DEVICES=0 IRWKV_RWKV_MTP_DEPTH=4 \
  /path/to/irwkv/.venv/bin/irwkv serve \
  /home/BOPENG/jl/models/rwkv7-g1k-7.2b-20260930-ctx25600-d4tulu1k-hf \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.55 \
  --max-num-seqs 1 --port 8000
```

Wait for `Application startup complete`; the engine log must contain:

```
RWKV MTP speculative decoding enabled: depth=4
RWKV MTP draft chain captured in a CUDA graph
```

Greedy request:

```bash
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "/home/BOPENG/jl/models/rwkv7-g1k-7.2b-20260930-ctx25600-d4tulu1k-hf",
  "prompt": "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
  "temperature": 0, "max_tokens": 32
}'
```

## Python

```python
import os
os.environ["IRWKV_RWKV_MTP_DEPTH"] = "4"   # before importing irwkv

from irwkv import LLM, SamplingParams

llm = LLM(
    model="/home/BOPENG/jl/models/rwkv7-g1k-7.2b-20260930-ctx25600-d4tulu1k-hf",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.55,
    max_num_seqs=1,
)
out = llm.generate(
    ["### Instruction:\nWhat is the capital of France?\n\n### Response:\n"],
    SamplingParams(temperature=0.0, max_tokens=256),
)
print(out[0].outputs[0].text)
```

## Environment variables

| variable | meaning |
|---|---|
| `IRWKV_RWKV_MTP_DEPTH` | draft depth; `0`/unset = disabled; must match the sidecar |
| `IRWKV_RWKV_MTP_STATS` | `1` prints `[mtp-stats] rounds=... accepted=...` |
| `IRWKV_RWKV_MTP_GRAPH` | `0` disables the draft CUDA graph (debug) |

The startup warning `Unknown irwkv environment variable detected:
IRWKV_RWKV_MTP_DEPTH` is harmless; the model config reads it.

## Constraints

- **Greedy only.** `_mtp_greedy()` requires: no logprobs, no logits processing,
  `temperature <= 0`, and `top_k <= 1` (or `top_k >= vocab_size`).  Otherwise the
  verify step is skipped and the normal sampler runs.
- **Single-request verify.** `_mtp_verify()` requires `num_reqs == 1`.  With
  `--max-num-seqs > 1`, single-request steps can still use MTP, but batched steps
  fall back to plain decode (startup logs an info message).
- **No non-greedy MTP.** There is no speculative-sampling acceptance rule; for
  `temperature > 0`, use plain decoding.
- **Depth match.** Runtime depth, sidecar `mtp_depth`, and training depth must
  agree.
- **Context.** Validated at `--max-model-len 2048`; longer contexts are not
  benchmarked with this sidecar.
- **PP=1**, chunked prefill; async scheduling is forced off.

## Benchmarks

Local 7.2B tulu d4 head, single H100, 512 prompt → 256 output, greedy, baseline
106.5 tok/s:

| prompt | depth | speedup | accept_len |
|---|---|---|---|
| tulu held-out | 4 | **1.510x** | 2.92 |
| tulu held-out | 3 | 1.504x | 2.72 |
| perfectblend | 3 | 1.271x | 2.15 |

Reproduce with `RWKV-PEFT/scripts/bench_irwkv_mtp.py`:

```bash
CUDA_VISIBLE_DEVICES=0 IRWKV_RWKV_MTP_DEPTH=4 IRWKV_RWKV_MTP_STATS=1 \
  /path/to/irwkv/.venv/bin/python ../RWKV-PEFT/scripts/bench_irwkv_mtp.py \
  --model /home/BOPENG/jl/models/rwkv7-g1k-7.2b-20260930-ctx25600-d4tulu1k-hf \
  --trust-remote-code --dtype bfloat16 \
  --gpu-memory-utilization 0.55 --max-num-seqs 1 --max-model-len 2048 \
  --input-len 512 --output-len 256 --num-iters-warmup 1 --num-iters 3 \
  --prompt-bin ../RWKV-PEFT/data/binidx/tulu.bin --prompt-offset 770536185
```

## Convert a new MTP checkpoint to a sidecar

```bash
python /home/BOPENG/jl/RWKV-PEFT/scripts/convert_lightmtp_sidecar.py \
  <rwkv-step-mtp.pth> /path/to/model-dir --depth 4
```

`model-dir` must already contain the base HF files.  The converter writes
`mtp.safetensors` and `mtp_config.json`; then set the env depth to the same value.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ValueError: IRWKV_RWKV_MTP_DEPTH requires mtp_config.json and mtp.safetensors...` | sidecar missing from the model directory |
| `ValueError: RWKV MTP depth is set but the model has no MTP head` | model code didn't load the head; check patch/model |
| no speedup, but server works | `temperature > 0`, batched requests, or depth mismatch; check the startup log for `MTP speculative decoding enabled` |
| corrupted / repeated text | missing `for_capture` fix or incomplete CUDA-graph capture; do not revert `model_states/rwkv.py` |
| `Unknown irwkv environment variable...` | harmless warning |

## Patch / compatibility

```bash
cd /path/to/irwkv
git apply --check /path/to/RWKV-PEFT/patches/irwkv_rwkv_mtp.patch
git apply         /path/to/RWKV-PEFT/patches/irwkv_rwkv_mtp.patch
```

Base commit `1f0bd68`.  If your irwkv base differs, port the 11 files manually;
the patch may conflict.

## Non-goals

- speculative sampling / non-greedy MTP;
- multi-request batching acceleration;
- upstream support (this is a local feature).
