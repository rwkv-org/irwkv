// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "irwkv-bench",
    about = "Benchmark online serving throughput and offline multimodal preprocessing",
    version = irwkv_build_info::VERSION
)]
struct Cli {
    #[command(flatten)]
    args: irwkv_bench::BenchServeArgs,

    /// Optional subcommand; absent by default for the online serving benchmark.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Offline multimodal preprocessing latency benchmark.
    MmProcessor(irwkv_bench::MmProcessorArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    irwkv_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    match cli.command {
        Some(Command::MmProcessor(args)) => {
            let (timing_layer, timing_stats) = irwkv_chat::mm_timing_layer();
            irwkv_tracing::init_tracing_with("Bench", timing_layer);
            runtime.block_on(irwkv_bench::run_mm_processor(args, timing_stats))
        }
        None => {
            irwkv_tracing::init_tracing("Bench");
            runtime.block_on(irwkv_bench::run(cli.args))
        }
    }
}
