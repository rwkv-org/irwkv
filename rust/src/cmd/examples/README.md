# `irwkv-rs` CLI Quick Start

Start an RWKV-7 checkpoint with one managed `irwkv-rs serve` command from the
repository root:

```bash
HF_HUB_OFFLINE=1 \
IRWKV_CPU_KVCACHE_SPACE=2 \
IRWKV_HOST_IP=127.0.0.1 \
IRWKV_LOOPBACK_IP=127.0.0.1 \
cargo run --bin irwkv-rs -- serve \
  /path/to/rwkv7-checkpoint \
  --python .venv/bin/python \
  --max-model-len 512 \
  -- \
  --dtype bfloat16
```

This launches:

- a managed headless Python `irwkv` engine
- the Rust OpenAI-compatible frontend on `127.0.0.1:8000`

All Python engine arguments must be placed after `--`. Arguments before `--` are parsed by the Rust
frontend itself.

You can then send OpenAI-style requests to the Rust frontend:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/rwkv7-checkpoint",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

If you already started headless `irwkv` yourself, use `frontend` instead:

```bash
cargo run --bin irwkv-rs -- frontend \
  --handshake-address tcp://127.0.0.1:62100 \
  /path/to/rwkv7-checkpoint
```
