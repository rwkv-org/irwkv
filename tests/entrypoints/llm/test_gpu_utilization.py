# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv import SamplingParams


def test_gpu_memory_utilization(irwkv_runner):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4)

    # makes sure gpu_memory_utilization is per-instance limit,
    # not a global limit
    with (
        irwkv_runner(
            "rwkv-rs/rwkv7-g1-st",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_chunked_prefill=True,
            max_model_len=128,
            max_num_seqs=4,
        ) as runner_0,
        irwkv_runner(
            "rwkv-rs/rwkv7-g1-st",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_chunked_prefill=True,
            max_model_len=128,
            max_num_seqs=4,
        ) as runner_1,
        irwkv_runner(
            "rwkv-rs/rwkv7-g1-st",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_chunked_prefill=True,
            max_model_len=128,
            max_num_seqs=4,
        ) as runner_2,
    ):
        for runner in (runner_0, runner_1, runner_2):
            outputs = runner.llm.generate(prompts, sampling_params)
            assert len(outputs) == len(prompts)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
