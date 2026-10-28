# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import functools
from typing import Annotated, Literal

import pydantic
from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

import irwkv.envs as envs
from irwkv.collect_env import get_env_info
from irwkv.config import IrwkvConfig
from irwkv.logger import init_logger

logger = init_logger(__name__)


router = APIRouter()
PydanticIrwkvConfig = pydantic.TypeAdapter(IrwkvConfig)


def _get_irwkv_env_vars():
    from irwkv.config.utils import normalize_value

    irwkv_envs = {}
    for key in dir(envs):
        if key.startswith("IRWKV_") and "KEY" not in key:
            value = getattr(envs, key, None)
            if value is not None:
                value = normalize_value(value)
                irwkv_envs[key] = value
    return irwkv_envs


@functools.lru_cache(maxsize=1)
def _get_system_env_info_cached():
    return get_env_info()._asdict()


@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    irwkv_config: IrwkvConfig = raw_request.app.state.irwkv_config
    server_info = {
        "irwkv_config": (
            str(irwkv_config)
            if config_format == "text"
            else PydanticIrwkvConfig.dump_python(
                irwkv_config, mode="json", fallback=str
            )
        ),
        # fallback=str is needed to handle e.g. torch.dtype
        "irwkv_env": _get_irwkv_env_vars(),
        "system_env": await asyncio.to_thread(_get_system_env_info_cached),
    }
    return JSONResponse(content=server_info)


def attach_router(app: FastAPI):
    app.include_router(router)
