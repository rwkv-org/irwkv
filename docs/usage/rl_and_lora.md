# LoRA, NoRA & Online RL Weight Synchronization

This guide describes how to serve unmerged PEFT **LoRA** and **NoRA** adapters without invalidating captured CUDA Graphs, manage versioned multi-tenant adapters at runtime, and push live checkpoint weight updates from RL trainers.

---

## 1. Unmerged LoRA & NoRA Adapters

RL trainers and PEFT workflows that keep an adapter unmerged can serve it directly inside `RwkvForCausalLM`. FlashRWKV2 applies the raw adapter weights, scaling factor, and NoRA normalization directly inside its fused projection kernels, matching `transformers-rwkv` forward semantics.

### Enabling `rwkv_lora`

Configure the `rwkv_lora` model attribute via `--hf-overrides`:

```bash
irwkv serve /path/to/rwkv-checkpoint \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --hf-overrides '{"rwkv_lora": {"r": 256, "lora_alpha": 256, "use_nora": true, "target_modules": ["r_proj", "k_proj", "v_proj", "o_proj", "lm_head"]}}'
```

### Configuration Contract

All four keys in `rwkv_lora` are **required** (extra or missing keys fail validation at startup):

| Key | Type | Requirements |
| --- | --- | --- |
| `r` | `int` | Positive integer LoRA rank (`> 0`) |
| `lora_alpha` | `int \| float` | Finite number; effective scale is `lora_alpha / r` |
| `use_nora` | `bool` | `true` for NoRA normalization from raw `A`; `false` for standard LoRA |
| `target_modules` | `list[str]` | Non-empty subset of `r_proj`, `k_proj`, `v_proj`, `o_proj`, `mlp.key`, `mlp.value`, `lm_head` |

### Parameter Storage & CUDA Graph Stability

- Each targeted projection allocates zero-initialized **FP32** parameters in PEFT layout:
    - `lora_A.weight`: `[r, in_features]`
    - `lora_B.weight`: `[out_features, r]`
- Parameter names follow the base weight prefix — for example:
    - `model.layers.3.linear_attn.r_proj.lora_A.weight`
    - `model.layers.3.mlp.value.lora_B.weight`
    - `lm_head.lora_A.weight`
- `mlp.value` retains its checkpoint layout (`[r, intermediate_size]` and `[hidden_size, r]`) even though its base weight is transposed after loading.
- Under PP2, each pipeline stage allocates adapters only for the layers (and `lm_head`) it owns.
- `load_weights` copies updated adapter tensors **in place** into the preallocated FP32 buffers, so captured CUDA Graphs stay valid across weight updates. Without `rwkv_lora`, no adapter parameters are allocated and no adapter arguments are passed to FlashRWKV2.

---

## 2. Versioned Multi-Tenant Runtime LoRA Serving

To load, replace, and unload multiple adapters dynamically at runtime, reserve the shared adapter buffer layout via `rwkv_lora` and enable `IRWKV_ALLOW_RUNTIME_LORA_UPDATING=1`:

```bash
IRWKV_ALLOW_RUNTIME_LORA_UPDATING=1 irwkv serve /path/to/base-model \
  --served-model-name base \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --api-server-count 1 \
  --max-cpu-loras 8 \
  --hf-overrides '{"rwkv_lora": {"r": 32, "lora_alpha": 32, "use_nora": false, "target_modules": ["r_proj", "k_proj", "v_proj", "o_proj", "mlp.key", "mlp.value", "lm_head"]}}'
```

> **Important constraints**:
>
> - Dynamic LoRA endpoints require a single Python API server (`--api-server-count 1`) and are intended for trusted/internal environments (see [Security Guide](security.md)).
> - Every loaded adapter directory must contain `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin` / `.pt`), use `peft_type="LORA"`, match the server's configured `r`, `lora_alpha / r`, and `use_nora`, omit `rank_pattern`/`alpha_pattern`/`modules_to_save`, and target only layers enabled in `rwkv_lora.target_modules`.

### Load, Replace & Unload Endpoints

**Load a new adapter:**

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "tenant-a", "lora_path": "/path/to/adapter-v1"}'
```

**Query the adapter** by passing its `lora_name` as `model`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tenant-a",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64
  }'
```

**Atomically replace an adapter in place** (versioned reload):

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "tenant-a", "lora_path": "/path/to/adapter-v2", "load_inplace": true}'
```

**Unload an adapter:**

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "tenant-a"}'
```

### Versioning, Isolation & Capacity Behavior

- **Distinct cache identities**: Every load receives a unique internal adapter ID. In-flight requests using `adapter-v1` keep running on `v1` until completion while new requests route to `v2`.
- **Batch isolation**: Requests targeting the base model and different adapter versions are scheduled in separate batches; switching the active adapter copies pinned CPU snapshots into the model's preallocated FP32 GPU buffers without breaking CUDA Graphs.
- **CPU slot capacity**: `--max-cpu-loras` sets per-worker pinned CPU adapter capacity. An in-place replacement (`"load_inplace": true`) can use one extra staging slot while the retiring version drains its active requests.

---

## 3. Online Weight Updates (RL Training Loop)

`RwkvForCausalLM` accepts checkpoint-format live weight updates through irwkv's native `nccl` and `ipc` weight-transfer backends so an RL trainer can synchronize updated policy weights without restarting the inference workers.

### Reload Modes (`IRWKV_RELOAD_MODE`)

| Mode | Description |
| --- | --- |
| `layerwise` (default) | Upstream-style staged layerwise reload with lifecycle finalization. On finalization failure, unfinished reload state is restored so a subsequent full update can reuse the original storage. |
| `direct` | In-place direct weight reload. |

### Operational Rules for Safe Weight Sync

1. **Use FP32 recurrent state** (`--mamba-ssm-cache-dtype float32`) so rollout trajectories and logprobs follow the FP32 training precision contract.
2. **Pause generation** before pushing weights and **reset the prefix cache** before resuming if prefix caching is enabled, so stale recurrent states computed under old weights are never reused.
3. **Folded embedding normalization**: When embedding normalization (`model.embedding_norm.*`) is folded into `model.embed_tokens.weight`, any update that modifies `model.embedding_norm.*` **must include the unfolded `model.embed_tokens.weight`** in the same update.
4. **Error recovery**: If a weight update fails mid-transfer after mutating parameters, push a complete checkpoint update before resuming generation.
5. **Rust frontend support**: With `IRWKV_USE_RUST_FRONTEND=1` and `IRWKV_SERVER_DEV_MODE=1`, `irwkv-rs` also exposes the HTTP weight-transfer lifecycle endpoints (`/update_weight_version`, `/weight_info`, and `HTTPIRWKVWeightSyncClient` routes). Never enable `IRWKV_SERVER_DEV_MODE=1` on untrusted or public networks (see [Security Guide](security.md)).
