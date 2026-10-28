# Engine-Core Client Examples

Start a headless RWKV-7 engine from the repository root. Include
`--enable-sleep-mode` for the utility-call example:

```bash
cargo run --bin irwkv-rs -- serve \
  /path/to/rwkv7-checkpoint \
  --python .venv/bin/python \
  --max-model-len 512 \
  --headless \
  -- \
  --dtype bfloat16 \
  --enable-sleep-mode
```

Then run the engine-core examples from another shell:

```bash
cargo run -p irwkv-engine-core-client --example external_engine_utility_call -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --model /path/to/rwkv7-checkpoint \
  --host 127.0.0.1

cargo run -p irwkv-engine-core-client --example external_engine_logprobs -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --model /path/to/rwkv7-checkpoint \
  --host 127.0.0.1
```

Run the crate tests with `cargo test -p irwkv-engine-core-client`.
