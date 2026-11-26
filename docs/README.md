# irwkv Documentation

**irwkv** is a CUDA inference and serving engine for **RWKV-7**, combining continuous batching, constant-memory recurrent state management, full CUDA Graph capture, and OpenAI-compatible APIs over [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2).

---

## Documentation Map

### Getting Started

| Guide | Contents |
| --- | --- |
| [Quickstart & Usage Recipes](usage/quickstart.md) | Source installation (`uv`), checkpoint requirements, offline `LLM` (`generate` / `chat`), OpenAI serving, reasoning, tool calling, structured outputs, and `irwkv bench` |

### Production Serving & Scaling

| Guide | Contents |
| --- | --- |
| [Serving, State & Multi-GPU Guide](usage/serving.md) | FP16 default vs FP32 parity WKV state, DeltaLog budgeting, offline kernel dispatch & CUDA Graphs, block-aligned prefix caching & multi-turn continuation, PP2/DP deployment, and the Rust frontend |
| [Security Guide](usage/security.md) | Network isolation, API authentication, media input bounds, and dev-mode endpoint restrictions |

### RL & PEFT Integration

| Guide | Contents |
| --- | --- |
| [LoRA, NoRA & Online RL Sync](usage/rl_and_lora.md) | Unmerged FP32 LoRA/NoRA adapters inside CUDA Graphs, versioned runtime `/v1/load_lora_adapter`, and `nccl`/`ipc` online weight updates |

### Architecture & Reference

| Guide | Contents |
| --- | --- |
| [RWKV-7 Architecture & Reference](models/rwkv.md) | Artifact validation contract, FlashRWKV2 operator surface, Rapid-Sampling decoding defaults, runtime constraints, environment variables, and official checkpoint links |

### Repository Developer Docs

Contributor and internal architecture guides live in the repository under `docs/contributing/` (`README.md`, `incremental_build.md`, `model/tests.md`, `pr-review.md`), `docs/design/model_runner_v2.md`, and `rust/README.md`.
