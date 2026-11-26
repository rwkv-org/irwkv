# RWKV-7 Architecture, Decoding Contract & Reference

`irwkv` serves RWKV-7 checkpoints natively via `RwkvForCausalLM`, backed exclusively by the fused CUDA/TileLang operators and recurrent-state provider in [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2). There is no CUDA, Triton, or PyTorch operator fallback inside `irwkv`.

For step-by-step tutorials and operational recipes, see:

- [Quickstart & Usage Recipes](../usage/quickstart.md) — installation, offline `LLM`, OpenAI APIs, reasoning, structured outputs, and benchmarks.
- [Serving, State & Multi-GPU Guide](../usage/serving.md) — FP32 vs FP16 state, CUDA Graph capture, tuning caches, prefix caching & continuation, PP2/DP, and the Rust frontend.
- [LoRA, NoRA & Online RL Sync](../usage/rl_and_lora.md) — unmerged FP32 adapters, versioned runtime LoRA serving, and online weight updates.

---

## Model Artifact & Operator Contract

Checkpoints must use the canonical `model_type="rwkv"` configuration, safetensors weight names, tokenizer, and chat template produced by [`transformers-rwkv`](https://github.com/rwkv-rs/transformers-rwkv).

### Startup Config Validation (`RwkvForCausalLM`)

| Config Field | Required Value |
| --- | --- |
| `model_type` | `"rwkv"` |
| `architecture_version` | `"rwkv7"` |
| `head_size` | `64` |
| `hidden_size` | Divisible by `head_size` (`64`) |
| `num_attention_heads` | `hidden_size // head_size` |
| `intermediate_size` | `4 * hidden_size` |
| `tie_word_embeddings` | `false` (untied embedding and LM head) |
| `head_dtype` | `None` or `"float32"` (enforced as `"float32"`) |

### FlashRWKV2 Dependency Contract

`requirements/rwkv.txt` pins Git revisions for `transformers-rwkv` and `tokenizers-rwkv` and installs `FlashRWKV2==0.2.0a4` from PyPI. At model load time, `RwkvForCausalLM` verifies the exact FlashRWKV2 version and public API surface:

- **State & metadata**: `prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor`, `prepare_tmix_wkv7_recurrent_fp16_state_from_tensor`, `prepare_tmix_wkv7_recurrent_metadata`
- **Fused layer islands**: `infer_embedding_ln0_forward_varlen`, `infer_tmix_postnorm_tokenshift_forward_varlen`, `infer_tmix_wkv_prepare_forward_varlen`, `infer_tmix_wkv7_recurrent_fp32io16_forward_varlen`, `infer_tmix_wkv7_recurrent_fp16_forward_varlen`, `infer_tmix_readout_forward_varlen`, `infer_cmix_forward_varlen`, `infer_post_norm_output_forward_varlen`, `infer_head_linear_all_forward_varlen`
- **Rapid-Sampling**: `setup_sampling_states`, `infer_sampling_six_parameter_forward_varlen`

---

## Serve a model

RWKV serving defaults to BF16 weights, token activations, and shift states, **FP16 WKV state**, and FP32 internal arithmetic and LM-head logits. Chunked prefill is required.

| Tensor / Stage | Precision | Notes |
| --- | --- | --- |
| Weights, token activations, TMix/CMix shift state | **BF16** | Required (`--dtype bfloat16`, `mamba_cache_dtype="bfloat16"`) |
| Internal recurrence math & LM head logits | **FP32** | Always FP32 regardless of WKV state dtype |
| WKV recurrent state (`--mamba-ssm-cache-dtype float16`, default) | **FP16** | Rounds WKV state to FP16 after every token across prefill and decode; budgets provider `private_bytes_per_slot` (DeltaLog) for `head_size=64` with 40 or 64 heads |
| WKV recurrent state (`--mamba-ssm-cache-dtype float32`) | **FP32** | Matches the FP32 training recurrence precision; use for RL rollouts, evaluation, and sampling-parameter search (`mamba_ssm_cache_dtype="float32"` in `LLM`) |

See [Serving, State & Multi-GPU Guide](../usage/serving.md#2-choosing-recurrent-state-precision-float16-vs-float32) for full serving commands, DeltaLog materialization (`RwkvStateLayer.materialize_state_slots`), bounded state transfers, and prefix-caching details.

---

## Kernel preparation

FlashRWKV2 selects kernels directly from the offline configurations shipped with the provider. `irwkv` warms up the model and head/sampling operations before CUDA Graph capture and passes active token and sequence counts to the recurrent operators so padding leaves request state intact.

- Remove `IRWKV_RWKV_TUNING_MODE` and standalone online tuning preflight steps from existing deployments (`--fail-on-environ-validation` rejects the retired variable).
- `FLASHRWKV2_CACHE_DIR` continues to select the native build cache (defaults to `~/.cache/flashrwkv2`).

See [CUDA Graph Capture and Kernel Preparation](../usage/serving.md#3-cuda-graph-capture-and-kernel-preparation) for details.

---

## Decoding defaults

RWKV samples through FlashRWKV2 Rapid-Sampling (`infer_sampling_six_parameter_forward_varlen`), where `frequency_penalty` is the additive increment and `penalty_decay` decays each request's per-token penalty state.

Unless `--generation-config irwkv` is passed to disable artifact profiles, every RWKV-7 artifact must ship **three standalone `GenerationConfig` files** containing all six Rapid-Sampling fields (`temperature`, `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `penalty_decay`). Omit `_from_model_config` or set it to `false` so Transformers does not rebuild from `config.json` and drop RWKV-specific fields.

Reference defaults for **G1l** and **G1l-preview** artifacts (`top_k=-1` disables top-k filtering; earlier checkpoints may publish different values, and `irwkv` always reads the active artifact's files):

| Request Mode | Config File | `temperature` | `top_p` | `top_k` | `presence_penalty` | `frequency_penalty` | `penalty_decay` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Open Think** (default) | `generation_config.json` | `0.96` | `0.90` | `-1` | `1.0` | `0.09` | `0.994` |
| **Fake Think** | `fake_think_generation_config.json` | `1.0` | `0.28` | `32` | `0.0` | `0.0` | `0.996` |
| **Tools** | `tools_generation_config.json` | `0.96` | `0.76` | `32` | `0.0` | `0.0` | `0.996` |

### Profile Selection Precedence

1. **Tools active**: If `tools` are supplied in the request, `tools_generation_config.json` is always selected regardless of `fake_cot_mode`.
2. **Fake Think**: If `chat_template_kwargs.fake_cot_mode` is `true` (default `false`), `fake_think_generation_config.json` is selected and the [canonical `transformers-rwkv` chat template](https://github.com/rwkv-rs/transformers-rwkv/blob/83447dd9935ccce8cbec2693853e76a1c9c2212f/temp/rwkv_chat_template.jinja) ends the generation prompt with `<think></think`.
3. **Open Think**: Default when neither condition above applies (`generation_config.json`).

Explicit request sampling fields override the selected profile. Offline `LLM.chat()` and `LLM.enqueue_chat()` apply the exact same profile selection when `sampling_params` is omitted. `min_p` defaults to `0` (disabled); a positive value filters tokens with probability below `min_p` times the highest token probability after applying current penalties and temperature and before Rapid-Sampling applies `top_k` and `top_p`.

---

## Pipeline parallelism

Single GPU (`--pipeline-parallel-size 1`) and two-stage pipeline parallelism (`--pipeline-parallel-size 2`) are supported, and both can be combined with `--data-parallel-size`. Stage 0 owns embedding and LN0; stage 1 owns the final norm and LM head; `hidden_states`, `residual`, and `v_first` pass between stages. See [Multi-GPU Deployment](../usage/serving.md#5-multi-gpu-deployment-pp2-and-dp) for launch commands.

---

## Tool calling

The canonical artifact chat template emits `**Tool Call:**` followed by a fenced JSON block (` ```json ... ``` `). Serve with `--enable-auto-tool-choice --tool-call-parser rwkv` (and optionally `--reasoning-parser rwkv` for `<think>...</think>` extraction). Do not pass a copied chat template. See [Quickstart: Tool Calling](../usage/quickstart.md#6-tool-calling).

---

## Runtime Constraints

`RwkvForCausalLMConfig.verify_and_update_config` enforces the following constraints at startup:

| Feature / Setting | Supported Values / Behavior |
| --- | --- |
| `--dtype` | `bfloat16` only |
| `--enable-chunked-prefill` | **Required** (`true`) |
| `--tensor-parallel-size` | `1` only (TP is not supported) |
| `--pipeline-parallel-size` | `1` or `2` (combinable with `--data-parallel-size`) |
| Decode / prefill context parallelism | `1` only (DCP/PCP not supported) |
| `--mamba-cache-dtype` | `bfloat16` (or `auto` → `bfloat16`) |
| `--mamba-ssm-cache-dtype` | `float16` (default / `auto`) or `float32` |
| `--mamba-cache-mode` | `none` (prefix caching off) or `align` (prefix caching on); `all` is unsupported |
| Compilation & CUDA Graph defaults | `CompilationMode.NONE`, `CUDAGraphMode.FULL` (or `NONE` with `--enforce-eager`) |
| KV offloading / KV connectors / stochastic rounding | Not supported |
| Rapid-Sampling restrictions | Raw logprobs only (no `processed_logits` / `processed_logprobs`), no custom logits processors, no sampling masks, no trace replay, no expanded logits |

---

## Environment Variables Reference

| Variable | Default | Description |
| --- | --- | --- |
| `FLASHRWKV2_CACHE_DIR` | `~/.cache/flashrwkv2` | Persistent cache root for FlashRWKV2 native builds |
| `IRWKV_USE_RUST_FRONTEND` | `0` | Set to `1` to run the Rust `irwkv-rs` HTTP/gRPC frontend with `irwkv serve` |
| `IRWKV_USE_RUST_BENCH` | `0` | Set to `1` to delegate `irwkv bench serve` to the native `irwkv-rs` benchmark binary |
| `IRWKV_RUST_FRONTEND_PATH` | Bundled binary | Override path to the `irwkv-rs` executable |
| `IRWKV_CHAT_TEMPLATE_FUEL` | `500000` | MiniJinja template execution fuel limit validated by the Rust frontend |
| `IRWKV_ALLOW_RUNTIME_LORA_UPDATING` | `0` | Set to `1` to expose `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` (requires `--api-server-count 1`) |
| `IRWKV_RELOAD_MODE` | `layerwise` | Online weight reload strategy: `layerwise` (default) or `direct` |
| `IRWKV_SERVER_DEV_MODE` | `0` | Enables internal/RL dev endpoints; **never enable in public production** (see [Security Guide](../usage/security.md)) |

---

## References and weights

Authoritative implementations and mathematical references:

- [RWKV-7 NumPy reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py) and [RWKV-7 Qwen 3.5 reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py)
- [Albatross](https://github.com/BlinkDL/Albatross), the CUDA inference reference for PRO 6000 GPUs
- [RWKV-7 pretraining implementation](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp) for H100
- [DPLR mathematics](https://zhiyuan1i.github.io/posts/dplr-mathematics), describing parallel computation of transition matrices
- [transformers-rwkv](https://github.com/rwkv-rs/transformers-rwkv) and [tokenizers-rwkv](https://github.com/rwkv-rs/tokenizers-rwkv), the Transformers checkpoint format, chat template, and Rust tokenizer

Official checkpoint repositories:

- [BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main)
- [BlinkDL/temp-latest-training-models](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)
- [rwkv-rs/rwkv7-g1-st](https://huggingface.co/rwkv-rs/rwkv7-g1-st) (Transformers safetensors format)

Official `.pth` checkpoint names follow `{architecture}-{data}-{size}-{date}-{context}.pth` (for example, `rwkv7-g1h-7.2b-20260710-ctx10240.pth`):

- **Architecture variants**: `rwkv7`, `rwkv7a` (DeepEmbed), `rwkv7b` (DeepEmbedAttn)
- **Data versions**: progress alphabetically from `g1a` onward (`g1h`, `g1k`, `g1l`, ...)
- **Model sizes**: `0.1B`, `0.4B`, `1.5B`, `2.9B`, `7.2B`, and `13.3B` (`1.5B` is commonly used for RL experiments and `7.2B` for inference benchmarking)
