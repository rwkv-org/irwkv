# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import irwkv


def test_embedded_commit_defined():
    assert hasattr(irwkv, "__version__")
    assert hasattr(irwkv, "__version_tuple__")
    assert irwkv.__version__ != "dev"
    assert irwkv.__version_tuple__ != (0, 0, "dev")
