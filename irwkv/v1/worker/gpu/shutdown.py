# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from irwkv.config import IrwkvConfig
from irwkv.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(irwkv_config: IrwkvConfig) -> None:
    from irwkv.v1.worker.workspace import reset_workspace_manager

    cache_config = irwkv_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = irwkv_config.compilation_config
    compilation_config.static_forward_context.clear()

    reset_workspace_manager()
