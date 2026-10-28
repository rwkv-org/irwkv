# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import weakref

import jsonschema
import pytest
import regex as re

from irwkv import SamplingParams
from irwkv.sampling_params import StructuredOutputsParams

MODEL_NAME = "rwkv-rs/rwkv7-g1-st"
JSON_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module", params=["xgrammar", "lm-format-enforcer"])
def structured_llm(request, irwkv_runner):
    with irwkv_runner(
        MODEL_NAME,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=512,
        max_num_seqs=4,
        generation_config="irwkv",
        structured_outputs_config={"backend": request.param},
    ) as runner:
        yield weakref.proxy(runner.llm)


@pytest.mark.parametrize("constraint", ["choice", "regex", "json"])
def test_structured_output(structured_llm, constraint):
    constraints = {
        "choice": StructuredOutputsParams(choice=["yes", "no"]),
        "regex": StructuredOutputsParams(regex=r"[0-9]{2}"),
        "json": StructuredOutputsParams(json=JSON_SCHEMA),
    }
    output = (
        structured_llm.generate(
            "Complete the response:",
            SamplingParams(
                temperature=0,
                max_tokens=64,
                structured_outputs=constraints[constraint],
            ),
        )[0]
        .outputs[0]
        .text
    )

    if constraint == "choice":
        assert output in {"yes", "no"}
    elif constraint == "regex":
        assert re.fullmatch(r"[0-9]{2}", output)
    else:
        jsonschema.validate(json.loads(output), JSON_SCHEMA)


def test_structured_and_unconstrained_requests_in_same_batch(structured_llm):
    outputs = structured_llm.generate(
        ["Answer yes or no:", "Complete the sentence: The sky is"],
        [
            SamplingParams(
                temperature=0,
                max_tokens=16,
                structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
            ),
            SamplingParams(temperature=0, max_tokens=16),
        ],
    )

    assert outputs[0].outputs[0].text in {"yes", "no"}
    assert outputs[1].outputs[0].token_ids
