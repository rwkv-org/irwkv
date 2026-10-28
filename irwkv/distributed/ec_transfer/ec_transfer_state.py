# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from irwkv.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from irwkv.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

if TYPE_CHECKING:
    from irwkv.config import IrwkvConfig

_EC_CONNECTOR_AGENT: ECConnectorBase | None = None


def get_ec_transfer() -> ECConnectorBase:
    assert _EC_CONNECTOR_AGENT is not None, "disaggregated EC cache is not initialized"
    return _EC_CONNECTOR_AGENT


def has_ec_transfer() -> bool:
    return _EC_CONNECTOR_AGENT is not None


def ensure_ec_transfer_initialized(irwkv_config: "IrwkvConfig") -> None:
    """Initialize EC cache connector."""
    global _EC_CONNECTOR_AGENT

    if irwkv_config.ec_transfer_config is None:
        return

    if (
        irwkv_config.ec_transfer_config.is_ec_transfer_instance
        and _EC_CONNECTOR_AGENT is None
    ):
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
            config=irwkv_config, role=ECConnectorRole.WORKER
        )


def ensure_ec_transfer_shutdown() -> None:
    global _EC_CONNECTOR_AGENT
    if _EC_CONNECTOR_AGENT is not None:
        _EC_CONNECTOR_AGENT.shutdown()
        _EC_CONNECTOR_AGENT = None
