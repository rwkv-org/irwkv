# Profiling irwkv

!!! warning
    Profiling is only intended for irwkv developers and maintainers to understand the proportion of time spent in different parts of the codebase. **irwkv end-users should never turn on profiling** as it will significantly slow down the inference.

!!! tip "Choosing a profiler"
    - Use **Nsight Systems** for low-overhead, performance-critical profiling.
    - Use **PyTorch Profiler** for medium-overhead profiling with richer debugging information (e.g., stack traces, memory, shapes). Note that enabling these features adds overhead and is not recommended for benchmarking.

## Profile with PyTorch Profiler

We support tracing irwkv workers using different profilers. You can enable profiling by setting the `--profiler-config` flag when launching the server.

To use the `torch.profiler` module, set the `profiler` entry to `'torch'` and `torch_profiler_dir` to the directory where you want to save the traces. Additionally, you can control the profiling content by specifying the following additional arguments in the config:

- `torch_profiler_record_shapes` to enable recording Tensor Shapes, off by default
- `torch_profiler_with_memory` to record memory, off by default
- `torch_profiler_with_stack` to enable recording stack information, on by default
- `torch_profiler_with_flops` to enable recording FLOPs, off by default
- `torch_profiler_use_gzip` to control gzip-compressing profiling files, on by default
- `torch_profiler_dump_cuda_time_total` to control dumping and printing the aggregated CUDA self time table, on by default
- `torch_profiler_activities` to select worker activities. Defaults are
  platform-specific: `["CPU"]` on CPU, `["CPU", "CUDA"]` on NVIDIA GPUs, and
  `["CPU", "XPU"]` on Intel GPUs. Selecting `["CUDA"]` omits CPU annotations
  and the AsyncLLM CPU trace to reduce profiling overhead and trace size.

When using `irwkv bench serve`, you can enable profiling by passing the `--profile` flag.

Traces can be visualized using <https://ui.perfetto.dev/>.

Each worker writes its trace and summary table under `torch_profiler_dir`,
named after its rank, for example `rank0.<id>.pt.trace.json.gz` and
`rank0.profiler_out.txt`. Depending on the parallel configuration, the rank
name can also include the parallel ranks, such as
`dp0_pp0_tp0_dcp0_ep0_rank0`. The AsyncLLM CPU trace uses
`<hostname>_<pid>.async_llm` instead of the rank name. When a session sets
`profile_prefix`, it is prepended as `<profile_prefix>_`, so summaries from
different sessions do not overwrite each other.

!!! note
    The summary table was previously written to `profiler_out_<rank>.txt`.

!!! tip
    You can directly call bench module without installing irwkv using `python -m irwkv.entrypoints.cli.main bench`.

!!! tip
    Only send a few requests through irwkv when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.

### Example commands and usage

#### OpenAI Server

```bash
irwkv serve rwkv-rs/rwkv7-g1-st --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./irwkv_profile"}'
```

irwkv bench command:

```bash
irwkv bench serve \
    --backend irwkv \
    --model rwkv-rs/rwkv7-g1-st \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

Or use http request:

```shell
# We need first call /start_profile api to start profile.
$ curl -X POST http://localhost:8000/start_profile

# Call model generate.
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
                "model": "rwkv-rs/rwkv7-g1-st",
                "messages": [
                        {
                                "role": "user",
                                "content": "San Francisco is a"
                        }
                ]
    }'

# After need call /stop_profile api to stop profile.
$ curl -X POST http://localhost:8000/stop_profile
```

`/start_profile` also accepts an optional JSON body with settings for that
profiling session. The Python and Rust API frontends accept the same fields:

```shell
curl -X POST http://localhost:8000/start_profile \
    -H "Content-Type: application/json" \
    -d '{
        "profile_prefix": "sharegpt_run_1",
        "delay_iterations": 5000,
        "max_iterations": 20
    }'
```

`profile_prefix` is prepended to the generated trace and summary file names. It
must be 1-128 ASCII characters, start with a letter or number, and contain only
letters, numbers, `.`, `_`, or `-`. `delay_iterations` skips worker iterations
before collection starts, and `max_iterations` limits the number of collected
worker iterations (`0` means no limit). Both must be non-negative JSON integers;
values of the wrong type, such as `"5"` or `5.0`, are rejected rather than
converted. Unknown fields are ignored. These iteration
bounds do not apply to the AsyncLLM frontend CPU profiler, which records from
`/start_profile` until `/stop_profile`. Always call `/stop_profile`, including
after the worker reaches `max_iterations`; starting another session before
stopping the current one returns HTTP 409.

## Profile with Triton Proton

[Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton)
is Triton's GPU profiler. It can collect a low-overhead aggregate tree or a
Chrome trace and works through the same irwkv profiling controls as the PyTorch
and CUDA profilers. Proton currently supports NVIDIA GPUs through CUPTI and
supports CUDA graph attribution.

Start a server with a local output directory and graph attribution:

```bash
irwkv serve rwkv-rs/rwkv7-g1-st \
    --profiler-config '{
        "profiler": "proton",
        "proton_profiler_dir": "./proton_profile",
        "proton_output_format": "hatchet",
        "proton_hook": "triton",
        "proton_graph_attribution": true
    }'
```

Then use `/start_profile` and `/stop_profile` as shown above, or pass
`--profile` to a irwkv benchmark. Each worker uses a topology- and
rank-qualified output name, such as
`proton_dp0_pp0_tp0_dcp0_ep0_rank0_pid1234_0123456789abcdef0123456789abcdef_run0.hatchet`,
so distributed workers, restarted servers, and repeated profiling runs do not
overwrite one another. A `profile_prefix` is included when supplied. Each
profile is written by `/stop_profile` and is ready to inspect immediately.

The Proton-specific options are:

- `proton_context`: `shadow` (default) or `python`
- `proton_data`: `tree` (default) or `trace`
- `proton_backend`: `cupti` or automatic
- `proton_mode`: an optional backend mode string
- `proton_hook`: `triton` to record Triton launch metadata, or unset
- `proton_output_format`: `hatchet`, `hatchet_msgpack`, `chrome_trace`, or unset
- `proton_graph_attribution`: observe CUDA graph capture for replay attribution;
  disabled by default and requires `proton_data: "tree"`

`hatchet` and `hatchet_msgpack` require `proton_data: "tree"`, while
`chrome_trace` requires `proton_data: "trace"`.

Automatic backend selection is recommended. irwkv currently supports Proton's
`cupti` backend on NVIDIA GPUs. ROCm support is not yet available. irwkv does not
expose Proton's experimental instrumentation backend because current upstream
can produce profiles without timing metrics. When `proton_graph_attribution` is
enabled, Proton observes irwkv's CUDA graph capture with the configured profiling
session active, then deactivates that same session until profiling starts. This
lets later profiles attribute replayed kernels without retaining model-startup
activity. Backend-specific modes can be selected with `proton_mode`;
`pcsampling` synchronizes the CUDA context and therefore requires
`--enforce-eager`. When CUDA graphs are enabled (including encoder graphs),
Proton requires `proton_graph_attribution: true` to collect replayed kernels.
For Chrome traces, disable CUDA graphs with `--enforce-eager`.

CUDA graph-attributed profiles support repeated `start_profile`/`stop_profile`
runs. Each stop flushes and writes one tree-data phase while preserving the
graph-aware session. Without `proton_graph_attribution`, each `stop_profile`
instead finalizes and writes an independent Proton session.

CUDA graph attribution requires Triton 3.7 or newer. It uses the phase data API
to discard graph-capture activity and separate profiling runs. The `hatchet_msgpack`
output format and `periodic_flushing` mode also require Triton 3.7 or newer.
`periodic_flushing` cannot be combined with graph attribution because both
manage the session's data phases. Ordinary Proton profiling remains available
with Triton 3.6.

Graph attribution retains capture metadata for the worker lifetime. With eager
execution or no graphs to capture, profiling uses ordinary independent sessions.

Inspect tree profiles with:

```bash
proton-viewer -m time/ns \
    proton_profile/proton_dp0_pp0_tp0_dcp0_ep0_rank0_pid1234_0123456789abcdef0123456789abcdef_run0.hatchet
```

Chrome traces (`proton_data: "trace"`) can be opened in
<https://ui.perfetto.dev/>. Proton is imported lazily, so selecting another
profiler does not require a Proton-capable Triton installation.

## Profile with NVIDIA Nsight Systems

Nsight systems is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

[Install nsight-systems](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) using your package manager.
The following block is an example for Ubuntu.

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

!!! tip
    When profiling with `nsys`, it is advisable to set the environment variable `IRWKV_WORKER_MULTIPROC_METHOD=spawn`. The default is to use the `fork` method instead of `spawn`. More information on the topic can be found in the [Nsight Systems release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues).

The Nsight Systems profiler can be launched with `nsys profile ...`, with a few recommended flags for irwkv: `--trace-fork-before-exec=true --cuda-graph-trace=node`.

### Example commands and usage

#### Offline Inference

For basic usage, you can just append the profiling command before any existing script you would run for offline inference.

The following is an example using the `irwkv bench latency` script:

```bash
nsys profile  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
irwkv bench latency \
    --model rwkv-rs/rwkv7-g1-st \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI Server

To profile the server, you will want to prepend your `irwkv serve` command with `nsys profile` just like for offline inference, but you will need to specify a few other arguments to enable dynamic capture similarly to the Torch Profiler:

```bash
# server
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    irwkv serve rwkv-rs/rwkv7-g1-st --profiler-config.profiler cuda

# client
irwkv bench serve \
    --backend irwkv \
    --model rwkv-rs/rwkv7-g1-st \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

With `--profile`, irwkv will capture a profile for each run of `irwkv bench serve`. Once the server is killed, the profiles will all be saved.

#### Analysis

You can view these profiles either as summaries in the CLI, using `nsys stats [profile-file]`, or in the GUI by installing Nsight [locally following the directions here](https://developer.nvidia.com/nsight-systems/get-started).

## Profiling irwkv Python Code

The Python standard library includes
[cProfile](https://docs.python.org/3/library/profile.html) for profiling Python
code.

### Example usage - function call

If a filename is specified, the profile will be saved to that file. If no
filename is specified, profile data can be printed to stdout.

```python
import cProfile


def expensive_function():
    # some expensive code
    pass


profiler = cProfile.Profile()
profiler.runcall(expensive_function)
profiler.dump_stats("expensive_function.prof")
```

### Example usage - context manager style

```python
import cProfile


def another_function():
    # more expensive code
    pass


profiler = cProfile.Profile()
profiler.enable()
try:
    another_function()
finally:
    profiler.disable()
    profiler.dump_stats("another_function.prof")
```

### Analyzing Profile Results

There are multiple tools available that can help analyze the profile results.
One example is [snakeviz](https://jiffyclub.github.io/snakeviz/).

```bash
uv pip install snakeviz
snakeviz expensive_function.prof
```

### Analyzing Garbage Collection Costs

Leverage IRWKV_GC_DEBUG environment variable to debug GC costs.

- IRWKV_GC_DEBUG=1: enable GC debugger with gc.collect elapsed times
- IRWKV_GC_DEBUG='{"top_objects":5}': enable GC debugger to log top 5
  collected objects for each gc.collect
