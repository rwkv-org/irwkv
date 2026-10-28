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
