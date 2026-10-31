# RWKV-7

irwkv serves RWKV-7 artifacts with the native `RwkvForCausalLM`
implementation. The artifact must use the canonical `model_type="rwkv"`
configuration, safetensors weight names, tokenizer, and chat template from
[`transformers-rwkv`](https://github.com/rwkv-rs/transformers-rwkv).

RWKV execution requires CUDA and the public fused operators and recurrent-state
provider in FlashRWKV2. The irwkv implementation does not include a CUDA, Triton,
or PyTorch fallback.

## Install dependencies

Create the project `.venv` and install the build and RWKV dependencies before
installing irwkv:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=auto
uv pip install -r requirements/rwkv.txt --torch-backend=auto
uv pip install -e . --no-build-isolation --torch-backend=auto
```

The requirements file fixes the `transformers-rwkv` and `tokenizers-rwkv`
commits and `FlashRWKV2==0.1.0a17`. FlashRWKV2 exposes the public
`prepare_tuning` and `get_tuning_cache_dir` APIs in addition to its
recurrent-state provider.

## Serve a model

RWKV uses BF16 weights, activations, and shift state with FP32 accumulation,
WKV recurrent state, and head logits. Chunked prefill is required. Prefix
caching is disabled by default, so each request updates one recurrent state.
The default FP32 recurrence follows the training precision contract.

```bash
irwkv serve <model> \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

For optional lower-memory inference, explicitly pass
`--mamba-ssm-cache-dtype float16`. This retains BF16 token inputs/outputs and
shift state, FP32 arithmetic and head logits, and rounds the WKV state to FP16
after every token. It follows a different numerical trajectory from FP32
training and can change logits and generated tokens. Prefill and decode use
the same per-token state rounding. This option does not change training
precision.

RWKV defaults to `CompilationMode.NONE` and full CUDA Graph capture so that
TorchDynamo does not decompose the external fused operators. Use
`--enforce-eager` only for diagnostics.

To opt into prefix sharing and copy-on-write, pass `--enable-prefix-caching`.
This uses irwkv's state-aligned cache blocks and copies recurrent state at block
boundaries.

Continuous batching, asynchronous scheduling, structured outputs, and the
OpenAI-compatible Chat and Responses APIs use the standard irwkv paths.

## Kernel preparation

FlashRWKV2 stores verified tuning profiles and native builds under
`~/.cache/flashrwkv2`. Set `FLASHRWKV2_CACHE_DIR` before starting the process to
choose another persistent root. The tuning root is independent of the virtual
environment, release directory, `TORCH_HOME`, and `XDG_CACHE_HOME`. TileLang
0.1.15 keeps its separate JIT cache at `~/.tilelang/cache`.

`IRWKV_RWKV_TUNING_MODE` controls preparation before CUDA Graph capture:

- `inspect` logs actual dispatch coverage and permits untuned defaults.
- `require` loads verified selections and fails on the first missing profile,
  with the operator, full key, and cache root in the diagnostic.
- `tune` (default) calls FlashRWKV2 to verify and tune only missing keys, then
  requires complete coverage before capture. Eager prefill shapes are prepared
  on first use with their recurrent state restored before the request executes.

Preparation uses the loaded model weights, configured state dtype, and each
actual graph shape, including the separate head/sampling graphs. Model tuning
uses fully active metadata and startup state slots, which are reset before
capture. The report records each selected configuration, fingerprint, device
class, profile path, and hit/missing/tuned/revalidated status. Same-class GPUs
and new worker processes reuse profiles through FlashRWKV2's locking and
compatibility checks.

To prepare the graph shapes before serving, run with the same model and serving
arguments as the target service, including its sequence limit, token budget,
and capture sizes:

```bash
FLASHRWKV2_CACHE_DIR=/path/to/persistent/flashrwkv2 \
IRWKV_RWKV_TUNING_MODE=tune \
timeout 7200s .venv/bin/irwkv serve <model> <serving-arguments>
```

After preparation, restart with `IRWKV_RWKV_TUNING_MODE=require` to verify loading
and capture in a fresh process. Eager workloads can use FlashRWKV2's public
`prepare_tuning` with representative forward calls and disposable state.

For a legacy namespace, add
`IRWKV_RWKV_TUNING_LEGACY_DIR=/path/to/tuning/<legacy-namespace>` to the explicit
`tune` run. FlashRWKV2 revalidates matching old configurations against the
current operators before publishing them; it tunes missing shapes normally.
Preserve the original namespace until the new process passes coverage checks.

## Decoding defaults

RWKV samples through FlashRWKV2 Rapid-Sampling. The model artifact must publish
three standalone generation configs:

| Request mode | Config file | temperature | top_p | top_k | presence | frequency | decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Think | `generation_config.json` | 0.96 | 0.76 | 32 | 1.0 | 0.1 | 0.988 |
| Fake Think | `fake_think_generation_config.json` | 1.0 | 0.28 | 32 | 0.0 | 0.0 | 0.996 |
| Tools | `tools_generation_config.json` | 0.96 | 0.76 | 32 | 0.0 | 0.0 | 0.996 |

These must be standalone `GenerationConfig` documents. Omit
`_from_model_config` or set it to `false`; otherwise Transformers rebuilds the
configuration from the model config and drops the RWKV-only penalty fields.

The default template mode selects Open Think. Setting
`chat_template_kwargs.fake_cot_mode` to `true` selects Fake Think
and ends the generation prompt with `<think></think`:

```json
"chat_template_kwargs": {
  "fake_cot_mode": true
}
```

`fake_cot_mode` defaults to `false`, which selects Open Think. The model's
`chat_template.jinja` must read `fake_cot_mode` directly. Refresh existing
model artifacts with the template from `transformers-rwkv/temp/rwkv_chat_template.jinja`.
Supplying tools always selects the Tools profile, regardless of the
thinking mode. Explicit request sampling fields override the selected profile.

All six controls are executed by
`infer_sampling_six_parameter_forward_varlen`; `frequency_penalty` is the
Rapid-Sampling additive increment and `penalty_decay` decays its per-request
token state. `--generation-config irwkv` explicitly disables artifact profiles.

## Pipeline parallelism

Single GPU and two-stage pipeline parallelism are supported.

For PP2, run one stage per GPU:

```bash
irwkv serve <model> \
  --dtype bfloat16 \
  --pipeline-parallel-size 2 \
  --enable-chunked-prefill
```

The first stage owns embedding and LN0. The second stage owns the final norm and
LM head. `hidden_states`, `residual`, and `v_first` are transferred between the
stages.

## Tool calling

The canonical artifact chat template emits `**Tool Call:**` followed by fenced
JSON. Load that template directly from the artifact and select the RWKV parser:

```bash
irwkv serve <model> \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv
```

The parser supports streaming and non-streaming requests with automatic,
required, or named tool choice. Do not pass a copied chat template.

## Unmerged LoRA and NoRA adapters

RL trainers that keep one PEFT LoRA or NoRA adapter unmerged can serve it
directly. Enable it with the `rwkv_lora` config attribute:

```bash
irwkv serve <model> \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --hf-overrides '{"rwkv_lora": {"r": 256, "lora_alpha": 256, "use_nora": true, "target_modules": ["r_proj", "k_proj", "v_proj", "o_proj", "lm_head"]}}'
```

`r`, `lora_alpha`, `use_nora`, and `target_modules` are all required. The scale
is `lora_alpha / r`. `target_modules` is a subset of `r_proj`, `k_proj`,
`v_proj`, `o_proj`, `mlp.key`, `mlp.value`, and `lm_head`, with PEFT suffix
matching. Each targeted projection gets zero-initialized FP32 parameters in
PEFT layout, `lora_A.weight` `[r, in_features]` and `lora_B.weight`
`[out_features, r]`, named after the base weight, for example
`model.layers.3.linear_attn.r_proj.lora_A.weight`,
`model.layers.3.mlp.value.lora_B.weight`, and `lm_head.lora_A.weight`.
`mlp.value` keeps the checkpoint layout, `[r, intermediate_size]` and
`[hidden_size, r]`, although its base weight is transposed after loading.
Pipeline stages create adapters only for the projections they own.

`load_weights` copies adapter tensors into the existing FP32 storage, so
captured CUDA Graphs stay valid across updates. The raw adapters, scale, and
NoRA flag are passed to the FlashRWKV2 projection operators, which apply NoRA
normalization from the raw A matrix. These are the operators and arguments of
the `transformers-rwkv` forward with an active unmerged PEFT adapter. Without
`rwkv_lora`, the model has no adapter parameters and passes no adapter
arguments.

## Uploads and session snapshots

Runtime uploads support state-tuned agents and trainers that publish new
adapter versions without restarting the server. Static `rwkv_lora` loading
still serves a single adapter loaded with the model; it does not accept later
training updates. Session snapshots retain an exact, computed conversation
frontier on the CPU after its GPU slot is released. Prefix caching instead
shares block-aligned recurrent states on the GPU between matching prompts.
Use prefix caching for shared prompts and snapshots for session-specific
continuations that should survive GPU cache eviction.

The Python server provides RAM-resident uploads when explicitly enabled:

```bash
export IRWKV_ALLOW_RUNTIME_RWKV_UPDATING=1
export IRWKV_RWKV_UPLOAD_MAX_BYTES=1073741824
```

| Operation | Endpoint |
| --- | --- |
| Upload initial state | `POST /v1/state/upload` |
| List or delete initial states | `GET /v1/state/list`, `POST /v1/state/delete` |
| Upload, list or delete adapters | `POST`, `GET`, `DELETE /v1/adapters` |
| Inspect session snapshots and cache hits | `GET /v1/state/status` |
| Delete a session snapshot | `POST /v1/state/session/delete` |

Uploads use multipart form data with one checkpoint `file` field. Adapter uploads
also require `adapter_id`. Ordinary PEFT uploads require an `adapter_config`
JSON file (at most 64 KiB), containing `r`, `lora_alpha`, `target_modules`, and
an explicit boolean `use_nora`; add `use_nora: false` for ordinary LoRA exports
that omit this extension. Both PyTorch tensor checkpoints (`weights_only=True`)
and safetensors are accepted, up to 512 MiB per upload. Initial states use
`blocks.N.att.time_state` tensors for every model layer, matching the
RWKV Lightning state export. Shift states start at zero. LoRA/NoRA tensors
use PEFT names and must match the server's configured `rwkv_lora` rank,
targets, alpha, and NoRA mode. NoRA requires `lora_alpha / r == 1` both at
startup and upload. Tensor names, shapes, finite values and float32 conversion
are checked. Unsupported PEFT variants are rejected: rsLoRA, DoRA, transposed
factors, biases, saved modules, per-layer rank/alpha and layer selection or
replication overrides. Dropout is inactive during inference. When the server
declares `rwkv_base_fingerprint`, PEFT metadata must supply the matching
`base_fingerprint`; otherwise base-model provenance is the caller's responsibility.
Uploads do not change the base model weights.

Lightning's native BF16 `adapter-final.pth` is also supported when it embeds
an `archive/nora.json` manifest. Upload validates its tensor names, shapes,
and content digest, then maps `blocks.N.*.weight.A/B` into the existing
adapter storage. Lightning NoRA normalizes A only at initialization and
trains the raw factors afterward. Set `use_nora` to `false` for these exports
to use the raw A/B forward path; `true` would normalize A again at every
forward and change the trained adapter. PEFT NoRA retains its existing
per-forward normalization behavior.

For a native rank-16, alpha-16 NoRA export targeting all six decoder
projections, start the server with:

```bash
irwkv serve <model> --dtype bfloat16 --enable-chunked-prefill \
  --hf-overrides '{
    "rwkv_lora": {
      "r": 16,
      "lora_alpha": 16,
      "use_nora": false,
      "target_modules": [
        "r_proj", "k_proj", "v_proj", "o_proj", "mlp.key", "mlp.value"
      ]
    }
  }'
```

Adapter storage must be configured before model loading and CUDA Graph
capture; a server started without `rwkv_lora` needs a restart. Use the base
checkpoint recorded by the training run: matching tensor shapes alone do
not establish base-model compatibility. If the model configuration declares
`rwkv_base_fingerprint`, native uploads must match it.

```bash
curl -F file=@state.pth http://localhost:8000/v1/state/upload
curl -F adapter_id=my-adapter -F file=@adapter_model.safetensors \
  -F adapter_config=@adapter_config.json \
  http://localhost:8000/v1/adapters
```

Select uploads using the existing `irwkv_xargs` request extension:

```json
{
  "session_id": "conversation-1",
  "irwkv_xargs": {
    "session_snapshot": true,
    "state_id": "<uploaded-state-id>",
    "adapter_id": "my-adapter",
    "adapter_version": "<uploaded-version>",
    "adapter_scale": 1.0
  }
}
```

Snapshots require both `session_id` and `irwkv_xargs.session_snapshot: true`.
A plain `session_id` retains its existing routing and cache-event semantics,
including deployments with DP > 1 or KV transfer, without entering this path.
Send the complete conversation history on each turn. Each finished snapshot
writer replaces its session's single CPU snapshot, including both shift tensors
and WKV state.
The next turn restores it when its computed token history is an exact prefix
of the new prompt. Any remaining tokens, including the last generated token
that has not yet entered the recurrent state, run normally. A changed prefix,
adapter version or scale, initial state, cache salt, or model weight version
causes a miss. Keep `state_id` consistent between turns when using an uploaded
initial state. Prompt-logprob requests recompute the prompt.

The cache keeps at most `IRWKV_RWKV_SESSION_CACHE_SIZE` session snapshots
(default 128), with only the latest CPU snapshot retained for each session.
When the limit is exceeded, the least recently used snapshot is evicted.
Set the environment variable before starting the server; for example, to
retain at most 64 session snapshots:

```bash
IRWKV_RWKV_SESSION_CACHE_SIZE=64 irwkv serve <model> \
  --dtype bfloat16 \
  --enable-chunked-prefill
```

Set it to `0` to disable session snapshots. This setting limits snapshot count;
snapshot tensor size is fixed by the model and token history by maximum context
length. Uploads have a separate per-worker byte budget, default 1 GiB, controlled
by `IRWKV_RWKV_UPLOAD_MAX_BYTES`. It includes uploaded states, adapter factors,
and references retained by admitted requests, including superseded/deleted
versions and scaled factors. Session snapshots are included in `ram_bytes`
separately from `upload_bytes`. Switching adapters also retains one fixed GPU
backup of the startup factors, avoiding a synchronous GPU-to-CPU read.
These counters measure live tensor storage, not process RSS. Allocator caches,
Python objects and temporary checkpoint/conversion buffers are additional.

At most 16 named adapters are stored using the existing `LRUCache` utility.
Upload and selection refresh recency. Uploads evict least recently used adapters
to satisfy the count and byte budgets; states are removed explicitly. In-flight
requests retain their tensors across eviction, replacement or deletion, and
those references continue to count against the byte budget. An upload that
cannot fit is rejected without changing the registry. A scaled request that
would exceed the budget is also rejected. Upload parsing is serialized per API
process, with at most 512 MiB per checkpoint, in addition to retained tensors.

Status reports `session_cache_hits`, `session_cache_misses`, `ram_bytes`, and
the computed token count for each snapshot. Deleting an upload preserves
references held by already admitted requests. Cache reset also invalidates
session snapshots.

Requests with different adapters or scales execute in separate batches.
Adapters are copied into existing GPU storage, preserving captured graph
addresses. A new cohort fills available slots from the waiting queue. Once an
incompatible request reaches the queue head, the running cohort drains before
another cohort is selected; later compatible arrivals cannot prolong it.
Both FCFS and priority queue order are respected at cohort boundaries. This
prevents starvation but can temporarily leave slots idle during draining.
FlashRWKV2's current batch-wide adapter parameters do not support simultaneous
base/adapter execution; increasing the request pool alone cannot remove that
constraint. No operator implementation is changed here.

This fork does not contain vLLM's `LoRARequest`, multi-LoRA manager or dynamic
LoRA endpoints. The [upstream LoRA interface](https://docs.vllm.ai/en/stable/features/lora/)
requires `SupportsLoRA` and operators supporting per-request factors. The RWKV
path keeps its existing `rwkv_lora` storage, configuration validator, request
extensions and common scheduler queues, and reuses the common LRU cache. The
multipart API also supports initial recurrent states and Lightning's native
initialization-only NoRA factors, which upstream LoRA loading does not represent.

Snapshots gather independent GPU tensors before slot reuse, then enqueue
copies to pinned CPU storage on a separate CUDA stream. Admission polls a CUDA
event; an unfinished copy produces a cache miss without waiting. Token history
is decoded on the CPU only after that event completes. Saving avoids GPU
`.item()`, `.tolist()` and synchronous D2H copies on the execution thread;
gathering, pinned allocation and transfer bandwidth still have a cost.
FlashRWKV2 0.1.0a17 also performs a GPU `.item()` bounds check in its state-handle
`_indices` method. `IRWKV_GPU_SYNC_CHECK=error` fails there even for ordinary
generation without snapshots. This external operator issue must be resolved
in FlashRWKV2 before the complete path can pass that check.

This path supports synchronous and asynchronous scheduling with DP=1, token
prompts, and no KV-transfer connector. Concurrent and multi-choice requests
are allowed: each reads an independent immutable snapshot at admission, and
only the most recently admitted branch may publish a new snapshot, regardless
of completion order. If it aborts, the previous completed snapshot remains
available; an already computed partial state may be saved on worker removal.
Applications needing to resume a particular choice should use distinct session
IDs for those branches. Offline callers can use the same selection fields,
including `session_id`, through `SamplingParams.extra_args`.

### Validation on 2026-10-31

Local validation used the existing `rwkv7-g1k-7.2b-20260930-ctx25600` checkpoint
on one RTX PRO 6000 Blackwell (96 GB), Torch 2.13.0+cu130,
transformers-rwkv 5.21.0.dev0 and FlashRWKV2 0.1.0a17. The designated remote
test host closed SSH connections, so these results are from the local GPU.
All runs used BF16 weights, FP32 recurrent state, async scheduling, the standard
async serving/detokenization path, `max_model_len=4096`, `max_num_seqs=512`,
`max_num_batched_tokens=4096`, GPU utilization 0.85, rank/alpha 16 NoRA on
`r_proj`, `k_proj`, `v_proj`, `o_proj`, and capture sizes `[1,4,64,320,512]`.
`IRWKV_RWKV_TUNING_MODE=inspect` permitted fallback kernels; these numbers do
not establish tuned peak performance or parity with Albatross.

The regression command was:

```bash
.venv/bin/python -m pytest \
  tests/models/rwkv/test_rwkv.py \
  tests/entrypoints/openai/test_session_id.py \
  tests/v1/core/test_scheduler.py \
  tests/v1/engine/test_parallel_sampling.py \
  tests/v1/engine/test_engine_core.py \
  -k 'not structured_output_keeps_content and not guidance_uses_rwkv_byte and (not engine_core or rwkv_parallel)' \
  -q --disable-warnings
```

Result: 179 passed, 8 skipped, 14 deselected. The two excluded structured-output
tests also fail when executed from the HEAD version of the test module:
the grammar mock violates `StructuredOutputGrammar`'s type check, and the
tokenizer mock is rejected by llguidance's fast-tokenizer requirement.
The other deselections are unrelated engine integration tests. All applicable
pre-commit hooks, including mypy, passed on the modified files.

Real HTTP validation passed for streaming and non-streaming Chat `n=2`,
Completions with two prompts and
`n=2`, and four concurrent requests sharing a session, with snapshots both off
and on. Snapshot continuations matched all 16 greedy output tokens from full
recomputation. Missing PEFT metadata and wrong alpha were rejected. A zero-B
adapter and zero uploaded initial state matched base output; a nonzero synthetic
adapter changed logprobs, scale zero restored base output, and eight interleaved
base/adapter requests matched their isolated outputs. This validates serving
behavior; it is not an evaluation of a trained adapter's task accuracy.

The official benchmark command was run for `B=1,4,64,320,512`, with snapshots
and prefix caching disabled, 64 input tokens and up to 32 output tokens:

```bash
.venv/bin/irwkv bench serve --backend openai \
  --base-url http://127.0.0.1:8123 --model rwkv --tokenizer "$MODEL" \
  --dataset-name random --random-input-len 64 --random-output-len 32 \
  --num-prompts "$((B * 2))" --max-concurrency "$B" --num-warmups 1 \
  --ignore-eos --extra-body '{"temperature":0,"presence_penalty":0,"frequency_penalty":0,"top_p":1,"top_k":-1,"stop":[]}'
```

| Concurrency | Completed / failed | Output token/s | Mean TTFT, ms | Mean TPOT, ms |
| --- | --- | --- | --- | --- |
| 1 | 2 / 0 | 66.2 | 24.7 | 14.8 |
| 4 | 8 / 0 | 239.7 | 61.0 | 15.1 |
| 64 | 128 / 0 | 878.3 | 1662.9 | 21.4 |
| 320 | 640 / 0 | 1206.2 | 4591.9 | 125.1 |
| 512 | 1024 / 0 | 813.7 | 8635.0 | 367.4 |

A separate end-to-end async HTTP workload compared snapshot saving with ordinary
generation: each pass sent `B` concurrent requests, each with 64 identical input
tokens and exactly 16 output tokens. Snapshot requests used distinct session IDs
and `session_snapshot=true`; ordinary requests omitted both. Each configuration
had one warmup and three measured passes, with prefix caching disabled. Per-request
prefill rate was prompt length / TTFT; decode rate was completion length /
(total latency - TTFT), timed from streamed responses. Standard async server
detokenization was retained. The session cache was already populated; ordinary
passes did not save snapshots. The table reports the median of the three passes'
median rates, with min/max also shown for decode:

| Concurrency | Prefill token/s, ordinary / snapshot | Ordinary decode token/s | Snapshot decode token/s |
| --- | --- | --- | --- |
| 1 | 778.59 / 770.94 | 98.45 (98.24–98.54) | 98.15 (98.00–98.30) |
| 4 | 521.02 / 512.14 | 97.16 (96.81–97.36) | 96.21 (95.85–96.25) |
| 64 | 128.76 / 140.15 | 75.78 (75.19–76.46) | 74.67 (74.21–74.92) |
| 320 | 6.64 / 6.74 | 29.87 (29.78–29.97) | 21.03 (19.08–21.28) |
| 512 | 6.00 / 6.15 | 21.85 (21.72–21.94) | 11.51 (11.50–11.54) |

Snapshot saving still has a material cost at high concurrency. Aggregate output
throughput also varied substantially: at 512, ordinary passes ranged from
421–2361 token/s and snapshot passes from 408–1873 token/s. The asynchronous
copy change does not establish a throughput improvement; operator synchronization,
gathering, allocation and transfer costs remain to be profiled upstream.

Prefix-cache comparison enabled `--enable-prefix-caching` (Mamba `align`, block
size 16). Each request had a distinct cache salt. A first turn used 64 prompt
tokens and generated 16; the second used that prompt, its output and one extra
token, then generated 16 more. Prefix mode omitted snapshot opt-in; snapshot
mode enabled it. Both modes also ran an independent full recomputation and
matched every output token at every concurrency below. These are single-pass
observations, not stable speedup estimates; cache capacity and scheduling differ
at high concurrency.

| Concurrency | Prefix TTFT, ms | Snapshot TTFT, ms | Snapshot hits / requests | CPU snapshot tensors, MiB |
| --- | --- | --- | --- | --- |
| 1 | 43.3 | 41.1 | 1 / 1 | 32.5 |
| 4 | 81.8 | 64.9 | 4 / 4 | 130.0 |
| 64 | 247.2 | 216.5 | 64 / 64 | 2080.0 |
| 320 | 3707.5 | 4419.2 | 128 / 320 | 4160.0 |
| 512 | 9235.6 | 3027.8 | 128 / 512 | 4160.0 |

Prefix mode retained no CPU snapshot tensors. Both configurations preallocated
about 64.5 GiB of GPU recurrent-state cache. A snapshot is 32.5 MiB for this
model; the default 128-entry cache retained 4.0625 GiB. One uploaded rank-16
adapter plus an initial state used 101,187,584 bytes of the 1 GiB upload budget.
An otherwise identical fresh worker used 2,059,044 KiB RSS. After the 512-request
snapshot workload, worker RSS was 10,737,568 KiB; resetting
the cache reduced live snapshot tensors to zero but left RSS unchanged because
allocator caches can retain freed storage. The snapshot count and upload byte
budget bound live retained tensors, rather than forcing process RSS to shrink.

The complete `IRWKV_GPU_SYNC_CHECK=error` run remains blocked by FlashRWKV2's
state-index bounds check, including for the ordinary path. The CUDA regression
test exercises the new snapshot gather/copy path with sync checking enabled and
a test state handle, including immediate slot/token-buffer reuse. Removing the
external synchronization and supporting simultaneous adapter cohorts require
FlashRWKV2 changes; they are not claimed by this change.

## Runtime constraints

RWKV runs with one tensor-parallel rank and supports pipeline parallel sizes
of one or two. Recurrent state uses Mamba cache mode `none` or `align`.
Rapid-Sampling supports raw logprobs and additive frequency/presence penalties;
its configuration validation checks sampling options before generation starts.

## References and weights

Authoritative implementations and references:

- [RWKV-7 NumPy implementation](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py) and [RWKV-7 Qwen 3.5 reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py).
- [Albatross](https://github.com/BlinkDL/Albatross), the CUDA inference reference for PRO 6000 GPUs.
- [RWKV-7 pretraining implementation](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp) for H100.
- [DPLR mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics), describing parallel computation of transition matrices.
- [transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv), the Transformers adaptation and Rust tokenizer.

Official checkpoints are published in [rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main), [temporary training models](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main), and [rwkv7-g1-st](https://huggingface.co/rwkv-rs/rwkv7-g1-st). Official `.pth` names use `{architecture}-{data}-{size}-{date}-{context}.pth`, for example `rwkv7-g1h-7.2b-20260710-ctx10240.pth`. Architecture variants include `rwkv7`, `rwkv7a` (DeepEmbed), and `rwkv7b` (DeepEmbedAttn); data versions advance from `g1a` toward later letters, and model sizes include 0.1B, 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B. The 1.5B model is often used for RL and 7.2B for inference testing.
