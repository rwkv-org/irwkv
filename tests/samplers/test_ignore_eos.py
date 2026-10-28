# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from irwkv import SamplingParams

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"


def test_ignore_eos(irwkv_runner):
    with irwkv_runner(
        MODEL_NAME,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=4,
    ) as runner:
        eos = runner.llm.get_tokenizer().eos_token_id
        for ignore_eos in (False, True):
            params = SamplingParams(
                max_tokens=8,
                ignore_eos=ignore_eos,
                allowed_token_ids=[eos],
                temperature=0,
            )
            output = runner.llm.generate("Hello", sampling_params=params)[0].outputs[0]
            assert output.token_ids == [eos] * (8 if ignore_eos else 1)
