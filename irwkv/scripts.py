# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from irwkv.entrypoints.cli.main import main as irwkv_main
from irwkv.logger import init_logger

logger = init_logger(__name__)


# Backwards compatibility for the move from irwkv.scripts to
# irwkv.entrypoints.cli.main
def main():
    logger.warning(
        "irwkv.scripts.main() is deprecated. Please re-install "
        "irwkv or use irwkv.entrypoints.cli.main.main() instead."
    )
    irwkv_main()
