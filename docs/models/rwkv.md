# RWKV-7

irwkv serves RWKV-7 artifacts with the native `RwkvForCausalLM`
implementation. The artifact must use the canonical `model_type="rwkv"`
configuration and safetensors weight names from
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

The requirements file fixes the native `tokenizers-rwkv` commit and
`FlashRWKV2==0.1.0a17`. RWKV configuration, tokenization, chat templates, and
the default llguidance structured-output backend run without Transformers.
The converted artifact format remains compatible with `transformers-rwkv`.
Tokenization uses the bundled original `rwkv_vocab_v20230424.txt` directly in
both Python and Rust. Its 65,529 byte entries retain IDs 1 through 65,529;
converted `tokenizer.json` vocabularies and added-token mappings are ignored.
`✿` (ID 10060) is the end-of-turn token. It belongs to the rendered chat history
and terminates generation without appearing in the returned body. ID 0 and
IDs above 65529 have no entries in the original vocabulary and are excluded
from generation. `<|endoftext|>`, `<think>`, and `</think>` are ordinary text,
not single tokens. The host-state fingerprint includes the original vocabulary.

Python and Rust use the same bundled RWKV chat template. It preserves message
whitespace and replays assistant reasoning and complete thinking markers so
that a following turn extends the earlier token history. Return the assistant's
`reasoning` field along with `content` when continuing an Open Think conversation.
Use `--chat-template` for an explicit custom template; converted artifact
templates no longer replace the native default.
FlashRWKV2 exposes the public
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

## Host state cache and binary snapshots

For multi-turn prefix reuse with one mutable GPU state per request, enable
the optional host cache instead of the token-aligned GPU prefix cache:

```bash
irwkv serve <model> --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"RwkvStateConnector","kv_role":"kv_both","kv_connector_extra_config":{"max_bytes":4294967296}}'
```

The CPU LRU holds immutable, uncompressed snapshots under its byte budget.
Automatic snapshots with the same model, `cache_salt`, and complete consumed
token prefix share one entry and a stable ID. Repeated completions refresh its
LRU position without another GPU-to-CPU copy, including when the first copy is
still in progress. Requests creating a new snapshot retain their GPU slot
until the asynchronous copy completes.
Snapshots contain the actual consumed prefix; the last sampled token may
still be absent from that prefix. Cache lookup checks only existing snapshot
lengths and respects `cache_salt`. A missing or evicted snapshot triggers
recomputation. This initial implementation requires TP=PP=DP=1.

The snapshot is a prefill checkpoint. Supply the full conversation or prompt
on the next request; only an exact token prefix is reused. The prompt must
extend beyond the cached prefix, since generation still computes logits from
at least one new token. Identical-prompt replay does not use a snapshot at its
full length. Sampling RNG and penalty history start fresh for the new request.

Both Python and Rust HTTP frontends expose these custom endpoints:

| Method | Endpoint | Result |
| --- | --- | --- |
| GET | `/v1/rwkv/states` | Cached snapshot metadata in `data` |
| GET | `/v1/rwkv/states/{id}` | Compressed binary download |
| POST | `/v1/rwkv/states` | Upload binary bytes; returns new snapshot metadata |
| DELETE | `/v1/rwkv/states/{id}` | Remove a snapshot |

Metadata includes `id`, the originating engine `request_id`, `num_tokens`, and
`context_sha1`. A finished request's snapshot becomes visible after its copy
completes. Imported snapshots have no originating engine request ID.

```bash
curl http://localhost:8000/v1/rwkv/states
curl http://localhost:8000/v1/rwkv/states/STATE_ID -o conversation.state
curl -X POST http://localhost:8000/v1/rwkv/states \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @conversation.state
```

Select an uploaded state with the existing request extension
`"kv_transfer_params":{"rwkv_state_id":"STATE_ID"}`. Uploaded states are
excluded from automatic shared prefix lookup: validating the file cannot prove
that its tensors were actually computed from its declared context. The explicit
ID remains a hint; a context mismatch falls back to normal prefill.
Uploaded states and their descendants remain separate even with identical
token contexts, since their tensor values may differ.

Files use a versioned header, lossless zlib compression, and safetensors.
They include consumed token IDs, their SHA-1 over little-endian uint32 values,
the cache salt, state tensors, and a model fingerprint over the configuration,
tokenizer, and actual safetensors weights. A SHA-256 checksum detects corruption.
Import checks the decompressed size, model identity, token range, tensor names,
shapes, dtypes, and finite values. Uploads are limited to 512 MiB; the model's
state size and maximum context length further bound decompression. Downloads
compress in a background CPU thread. State files contain conversation tokens
and should be handled as conversation data.

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

These are standalone JSON sampling profiles. The native loader keeps the six
sampling controls and uses `✿` as the end-of-turn token in every profile.

The default template mode selects Open Think. Setting
`chat_template_kwargs.rwkv_generation_prompt` to `fake_think` selects Fake
Think. Supplying tools always selects the Tools profile, regardless of the
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

The native chat template emits `**Tool Call:**` followed by fenced JSON.
Select the RWKV parser:

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
