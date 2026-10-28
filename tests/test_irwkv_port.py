# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest

from irwkv.envs import get_irwkv_port


def test_get_irwkv_port_not_set():
    """Test when IRWKV_PORT is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_irwkv_port() is None


def test_get_irwkv_port_valid():
    """Test when IRWKV_PORT is set to a valid integer."""
    with patch.dict(os.environ, {"IRWKV_PORT": "5678"}, clear=True):
        assert get_irwkv_port() == 5678


def test_get_irwkv_port_invalid():
    """Test when IRWKV_PORT is set to a non-integer value."""
    with (
        patch.dict(os.environ, {"IRWKV_PORT": "abc"}, clear=True),
        pytest.raises(ValueError, match="must be a valid integer"),
    ):
        get_irwkv_port()


def test_get_irwkv_port_uri():
    """Test when IRWKV_PORT is set to a URI."""
    with (
        patch.dict(os.environ, {"IRWKV_PORT": "tcp://localhost:5678"}, clear=True),
        pytest.raises(ValueError, match="appears to be a URI"),
    ):
        get_irwkv_port()
