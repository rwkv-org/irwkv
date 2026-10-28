# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.model_executor.layers.layernorm import RMSNorm


@RMSNorm.register_oot
class DummyRMSNorm(RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addition_config = True
