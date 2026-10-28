# irwkv-bench

`irwkv-bench` is the Rust benchmark client for an OpenAI-compatible server. It is
part of this repository's Rust workspace.

## Build

From the repository root, build the client with:

```bash
cargo build --release --manifest-path rust/Cargo.toml \
  -p irwkv-bench -p irwkv-cmd --features irwkv-cmd/native-tls-vendored
```

The executable is `rust/target/release/irwkv-bench` unless `CARGO_TARGET_DIR` is
set. Run `irwkv-bench --help` for the available datasets and options.

## Benchmark a server

Start an irwkv server with an RWKV model, then run:

```bash
rust/target/release/irwkv-bench \
  --backend irwkv \
  --base-url http://127.0.0.1:8000 \
  --model rwkv-rs/rwkv7-g1-st \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 320 \
  --max-concurrency 320
```

Use `--save-result` to write the benchmark results or `--dry-run` to inspect the
requests without sending them. The client reports latency and throughput
statistics for the requests it executes.
