# LLM Example

Start a headless RWKV-7 engine as shown in the
[`irwkv-rs` CLI Quick Start](../../cmd/examples/README.md), then run the
example from the repository root:

```bash
cargo run -p irwkv-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --model /path/to/rwkv7-checkpoint \
  --host 127.0.0.1
```

Run the crate tests with `cargo test -p irwkv-llm`.
