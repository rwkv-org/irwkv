# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace

from fastapi import FastAPI

from irwkv import envs
from irwkv.config import ModelConfig
from irwkv.tasks import SupportedTask


def register_api_routers(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    from irwkv.entrypoints.serve import register_irwkv_serve_api_routers

    register_irwkv_serve_api_routers(app)

    from irwkv.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    from irwkv.entrypoints.serve.sagemaker.api_router import (
        attach_router as register_sagemaker_api_router,
    )

    register_sagemaker_api_router(app, supported_tasks, model_config)

    if envs.IRWKV_SERVER_DEV_MODE:
        from irwkv.entrypoints.serve import register_irwkv_dev_api_routers

        register_irwkv_dev_api_routers(app)

        from irwkv.entrypoints.rl.factories import register_rl_api_routers

        register_rl_api_routers(app)

    if "generate" in supported_tasks:
        from irwkv.entrypoints.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

    if "generate" in supported_tasks or "render" in supported_tasks:
        from irwkv.entrypoints.scale_out.factories import register_scale_out_api_routers

        register_scale_out_api_routers(app, supported_tasks)
