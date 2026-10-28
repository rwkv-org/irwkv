# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass


@dataclass
class BaseModelPath:
    name: str
    model_path: str
