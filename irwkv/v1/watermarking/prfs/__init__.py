# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.config.watermarking import WatermarkPRFName
from irwkv.v1.watermarking.prfs.base import WatermarkPRF
from irwkv.v1.watermarking.prfs.philox import PhiloxPRF


def create_prf(name: WatermarkPRFName, key: int) -> WatermarkPRF:
    if name == "philox":
        return PhiloxPRF(key)
    raise ValueError(f"Unknown watermark PRF: {name}")


__all__ = [
    "PhiloxPRF",
    "WatermarkPRF",
    "create_prf",
]
