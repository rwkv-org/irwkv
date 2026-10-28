# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.v1.attention.backends.rwkv_attn import RwkvAttentionBackend


class DummyAttentionBackend(RwkvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
