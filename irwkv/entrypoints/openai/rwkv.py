# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV state, adapter upload, and session snapshot management routes."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile

from irwkv import envs

router = APIRouter()


async def _rwkv_rpc(
    raw_request: Request, operation: str, data: dict[str, Any] | None = None
):
    engine = raw_request.app.state.engine_client
    if getattr(engine.model_config.hf_config, "model_type", None) != "rwkv":
        raise HTTPException(404, "RWKV cache endpoints require an RWKV model")
    try:
        results = await engine.collective_rpc(
            "rwkv_cache", args=(operation, data or {})
        )
    except (ValueError, RuntimeError, KeyError) as error:
        raise HTTPException(400, str(error)) from error
    errors = [worker["error"] for worker in results if "error" in worker]
    if errors:
        raise HTTPException(400, errors[0])
    result = results[0]
    if operation == "status":
        result["ram_bytes"] = sum(worker["ram_bytes"] for worker in results)
    return result


async def _rwkv_identifier(raw_request: Request, name: str) -> dict[str, Any]:
    try:
        data = await raw_request.json()
    except ValueError as error:
        raise HTTPException(400, "request body must be a JSON object") from error
    if (
        not isinstance(data, dict)
        or not isinstance(data.get(name), str)
        or not data[name]
    ):
        raise HTTPException(400, f"missing {name}")
    return data


async def _rwkv_upload(raw_request: Request, operation: str):
    if not envs.IRWKV_ALLOW_RUNTIME_RWKV_UPDATING:
        raise HTTPException(
            403, "runtime uploads require IRWKV_ALLOW_RUNTIME_RWKV_UPDATING=1"
        )
    lock = getattr(raw_request.app.state, "rwkv_upload_lock", None)
    if lock is None:
        lock = raw_request.app.state.rwkv_upload_lock = asyncio.Lock()
    async with lock:
        return await _rwkv_upload_locked(raw_request, operation)


async def _rwkv_upload_locked(raw_request: Request, operation: str):
    async with raw_request.form(
        max_files=2 if operation == "upload_adapter" else 1,
        max_fields=2,
        max_part_size=64 << 10,
    ) as form:
        files = [
            value
            for key, value in form.multi_items()
            if key != "adapter_config" and isinstance(value, UploadFile)
        ]
        if len(files) != 1:
            raise HTTPException(400, "upload requires exactly one file")
        file: UploadFile = files[0]
        payload = await file.read((512 << 20) + 1)
        if not payload or len(payload) > 512 << 20:
            raise HTTPException(400, "upload must contain 1 to 512 MiB")
        data: dict[str, Any] = {"payload": payload}
        if operation == "upload_adapter":
            adapter_id = form.get("adapter_id")
            if not isinstance(adapter_id, str) or not adapter_id:
                raise HTTPException(400, "missing adapter_id")
            data["adapter_id"] = adapter_id
            metadata = form.get("adapter_config")
            if isinstance(metadata, UploadFile):
                metadata = await metadata.read((64 << 10) + 1)
            if metadata is not None:
                if len(metadata) > 64 << 10:
                    raise HTTPException(400, "adapter_config exceeds 64 KiB")
                try:
                    data["adapter_config"] = json.loads(metadata)
                except (ValueError, UnicodeDecodeError) as error:
                    raise HTTPException(400, "invalid adapter_config JSON") from error
        await _rwkv_rpc(raw_request, "validate_" + operation, data)
        return await _rwkv_rpc(raw_request, operation, data)


@router.post("/v1/state/upload")
async def upload_rwkv_state(raw_request: Request):
    return await _rwkv_upload(raw_request, "upload_state")


@router.get("/v1/state/list")
@router.post("/v1/state/list")
async def list_rwkv_states(raw_request: Request):
    status = await _rwkv_rpc(raw_request, "status")
    return {"object": "list", "data": status["states"]}


@router.post("/v1/state/delete")
@router.delete("/v1/state/delete")
async def delete_rwkv_state(raw_request: Request):
    data = await _rwkv_identifier(raw_request, "state_id")
    return await _rwkv_rpc(raw_request, "delete_state", data)


@router.post("/v1/adapters")
async def upload_rwkv_adapter(raw_request: Request):
    return await _rwkv_upload(raw_request, "upload_adapter")


@router.get("/v1/adapters")
async def list_rwkv_adapters(raw_request: Request):
    status = await _rwkv_rpc(raw_request, "status")
    return {"object": "list", "data": status["adapters"], "uploads": status["uploads"]}


@router.delete("/v1/adapters")
async def delete_rwkv_adapter(raw_request: Request):
    data = await _rwkv_identifier(raw_request, "adapter_id")
    return await _rwkv_rpc(raw_request, "delete_adapter", data)


@router.get("/state/status")
@router.post("/state/status")
@router.get("/v1/state/status")
async def rwkv_cache_status(raw_request: Request):
    return await _rwkv_rpc(raw_request, "status")


@router.post("/state/delete")
@router.post("/v1/state/session/delete")
async def delete_rwkv_session(raw_request: Request):
    data = await _rwkv_identifier(raw_request, "session_id")
    return await _rwkv_rpc(raw_request, "delete_session", data)


def attach_router(app: FastAPI):
    app.include_router(router)
