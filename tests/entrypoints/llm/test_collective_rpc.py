# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ...utils import create_new_process_for_each_test


@create_new_process_for_each_test()
def test_collective_rpc(monkeypatch, irwkv_runner):
    # intentionally define the method and class in the test function,
    # to test if they can be serialized and sent to the workers
    def echo_rank(self):
        return self.rank

    monkeypatch.setenv("IRWKV_ALLOW_INSECURE_SERIALIZATION", "1")
    with irwkv_runner(
        "rwkv-rs/rwkv7-g1-st",
        enforce_eager=True,
        enable_chunked_prefill=True,
        load_format="dummy",
        tensor_parallel_size=1,
    ) as runner:
        assert runner.llm.collective_rpc(echo_rank) == [0]
