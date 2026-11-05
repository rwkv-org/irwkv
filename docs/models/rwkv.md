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
commits and `FlashRWKV2==0.1.0a18`. FlashRWKV2 exposes the public
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

For finer prefix reuse, set `--prefix-match-unit` to a divisor of the state
block size. For example:

```bash
irwkv serve <model> \
  --enable-prefix-caching \
  --block-size 32 --mamba-block-size 32 --prefix-match-unit 8
```

With a 73-token prompt, a repeated request can resume from token 72 instead of
token 64. The shared checkpoint is copied into private state before
continuation, so divergent suffixes do not modify each other. Only checkpoints
actually saved by the scheduler can be reused; the match unit controls hashing
granularity, not how often states are stored.

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
class, profile path, and fixed/hit/missing/tuned status. Same-class GPUs
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

On RTX 4090 and PRO 6000 GPUs, FlashRWKV2 uses deterministic configurations for
supported BF16 projections and head-size-64 WKV with FP32 state. These dispatches
report `fixed` and do not search or time configurations. Preparation still compiles
their reachable kernel variants before capture. FP16 recurrent state and other
unsupported layouts retain verified profiles and bounded tuning. After upgrading,
run preparation again; compatible TileLang binaries remain reusable.

## Decoding defaults

RWKV samples through FlashRWKV2 Rapid-Sampling. The model artifact must publish
three standalone generation configs. These are the G1l and G1l-preview defaults:

| Request mode | Config file | temperature | top_p | top_k | presence | frequency | decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Think | `generation_config.json` | 0.96 | 0.90 | -1 | 1.0 | 0.09 | 0.994 |
| Fake Think | `fake_think_generation_config.json` | 1.0 | 0.28 | 32 | 0.0 | 0.0 | 0.996 |
| Tools | `tools_generation_config.json` | 0.96 | 0.76 | 32 | 0.0 | 0.0 | 0.996 |

Earlier checkpoints may specify different values; irwkv uses the selected
model artifact’s configuration. `top_k=-1` disables top-k filtering.

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

`fake_cot_mode` defaults to `false`, which selects Open Think. Model artifacts
use the [`transformers-rwkv` chat template](https://github.com/rwkv-rs/transformers-rwkv/blob/83447dd9935ccce8cbec2693853e76a1c9c2212f/temp/rwkv_chat_template.jinja).
Supplying tools always selects the Tools profile, regardless of the
thinking mode. Explicit request sampling fields override the selected profile.
The same profile selection applies to `LLM.chat()` and `LLM.enqueue_chat()`
when `sampling_params` is omitted; explicitly supplied `sampling_params` are
used as provided.

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

## Uploaded adapters and initial states

The Python API server accepts Lightning NoRA `.pth` exports, LoRA tensor
checkpoints, and `blocks.N.att.time_state` checkpoints. Uploads are parsed with
`torch.load(weights_only=True)`, checked against the model dimensions, and kept
in worker CPU memory. No uploaded file is needed after registration. Each worker
has separate adapter and initial-state LRU stores, each limited to **16 entries**.
These stores are process-local and are cleared when the server restarts.

Reserve adapter parameters when starting the server with a base model:

```bash
irwkv serve <base-model> --dtype bfloat16 --enable-chunked-prefill \
  --hf-overrides '{"rwkv_dynamic_adapters": true, "rwkv_lora": {"r": 16, "lora_alpha": 16, "use_nora": false, "target_modules": ["r_proj", "k_proj", "v_proj", "o_proj", "mlp.key", "mlp.value"]}}'
```

`rwkv_dynamic_adapters` permits the reserved adapter parameters to remain
zero-initialized when the base checkpoint does not contain them. The configured
rank is the maximum accepted rank; smaller adapters are zero-padded. Enable all
projections that uploaded adapters will target. Initial-state uploads alone do
not require `rwkv_lora` or `rwkv_dynamic_adapters`.

Lightning's exported NoRA A matrix is already initialized/trained with its own
normalization convention. Its inference computes the raw A/B product: enabling
`use_nora=true` would incorrectly normalize A again. Dynamic uploads therefore
require `use_nora=false` and nonzero `lora_alpha` in the server configuration.
LoRA uploads use canonical PEFT names or Lightning `blocks.N.*.weight.A/B`
names and require metadata with `r`, `lora_alpha`, and optionally
`use_nora: false`. Supply this as a multipart `adapter_config.json` file or as
the `metadata` object in a JSON registration request. Lightning NoRA exports
contain `archive/nora.json` and need no separate metadata.

```bash
curl -f http://localhost:8000/v1/adapters \
  -F adapter_id=my-nora -F file=@adapter-final.pth

curl -f http://localhost:8000/v1/state/upload \
  -F state_id=agentic -F file=@rwkv-state-agentic.pth

curl -f http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-model>","messages":[{"role":"user","content":"你好"}],"adapter_id":"my-nora","state_id":"agentic","max_tokens":64}'
```

`adapter_id`, optional `adapter_version`, and `state_id` are engine extensions
on Chat Completions and Completions. They can also be supplied through
`irwkv_xargs`; conflicting values are rejected. Omitting both IDs selects the
startup model with zero initial state. `adapter_version` pins the content version
returned by upload, so a stale version cannot silently select a replacement.

Both upload endpoints also accept JSON with a server-local `path` and the
corresponding ID. If `state_id` is omitted, the state upload returns a
content-derived ID. The upload limit is 512 MiB per checkpoint. Shape mismatches,
missing A/B pairs, nonfinite tensors, and unsupported formats are rejected.

| Operation | Endpoint |
| --- | --- |
| Register/upload adapter | `POST /v1/adapters` |
| List adapters and cache counters | `GET /v1/adapters` |
| Delete adapter | `DELETE /v1/adapters`, JSON `{"adapter_id":"my-nora"}` |
| Register/upload initial state | `POST /v1/state/upload` |
| List initial states and cache counters | `GET /v1/state/list` |
| Delete initial state | `POST /v1/state/delete`, JSON `{"state_id":"agentic"}` |

The list response exposes `capacity`, `eviction_policy`, `ram_hits`, `evictions`,
and resident tensor sizes. Uploading a seventeenth entry evicts the least
recently used idle entry. Requests pin their selected versions through
completion or cancellation. An active entry cannot be replaced or deleted;
when all sixteen are pinned, a new upload is rejected. An evicted or unknown
ID returns an error rather than falling back to the base model.

Requests using different adapters execute in separate batches; requests using
the same adapter can batch together, including requests with different initial
states. Adapter tensors are copied into fixed GPU buffers, preserving CUDA
Graph addresses. Initial states are transposed from Lightning checkpoint layout
into runtime layout and copied into private request slots with zero shift state.
Prefix-cache identities include adapter and initial-state content versions to
prevent reuse across different selections.

When an uploaded LoRA's `lora_alpha / r` differs from the startup scale,
its batches run the same FlashRWKV2 operators eagerly. This preserves raw A/B
weights and the requested scale without extra BF16 rounding or replaying a
graph with a stale scalar. Matching-scale adapters, including Lightning NoRA
with the configuration above, retain CUDA Graph execution.

Use the base checkpoint on which the adapter was trained. Dimension validation
does not establish base-model identity: a `.pth` fingerprint cannot be compared
directly with converted/sharded safetensors. These endpoints use the server's
normal API-key authentication. The alternative Rust frontend does not expose
these upload routes.

### Adapter validation

Run the CPU/model and scheduling regressions with:

```bash
.venv/bin/python -m pytest tests/models/rwkv/test_rwkv.py \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py -q
```

The optional HTTP regression starts a real CUDA Graph server, uploads the
checkpoints, checks cold/hit prefix isolation and concurrent generation, and
fills the LRU past its sixteen-entry limit:

```bash
IRWKV_RWKV_TEST_MODEL=/path/to/base-hf \
IRWKV_RWKV_TEST_NORA=/path/to/adapter-final.pth \
IRWKV_RWKV_TEST_STATE=/path/to/rwkv-state-agentic.pth \
  .venv/bin/python -m pytest tests/models/rwkv/test_rwkv.py \
  -k uploaded_adapters_generate -v
```

Validation on 2026-11-04 used an RTX PRO 6000 Blackwell, the G1L preview
`rwkv-g1l-7b-temp-3797` base (SHA-256 `7bbef36528bb15e992bd3eb3786a9ebd7cb3e174ed566360770a37d1a7be3e60`,
matching the supplied NoRA manifest), rank-16 NoRA, and the 32-layer agentic
state. The real uploads, cache hits, seventeenth-entry eviction, mixed concurrent
requests, streaming cancellation, and return to the base model were checked.
An additional LoRA registration used the same raw A/B tensors at scale 0.5.

Across two prompts and six adapter/state combinations, all 186 returned token
IDs matched independent `transformers-rwkv` plus PEFT generation. One sequence
ended at the configured `✿` stop string. The reference used transformers commit
`83447dd9935ccce8cbec2693853e76a1c9c2212f` with its required FlashRWKV2 a17;
irwkv used a18. This is a loading/inference regression, not a task-quality eval.

An `irwkv bench serve` smoke run on that base with the NoRA adapter, random
128-token inputs, up to 32 output tokens, 8 prompts, and concurrency 4 completed
8/8 requests: 235.68 output tokens/s, mean TTFT 83.22 ms, and mean TPOT 12.68 ms.
These short-run measurements are not a throughput claim for other batch sizes.

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
