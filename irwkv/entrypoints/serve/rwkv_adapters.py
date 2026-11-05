# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upload and inspect RWKV adapters retained in worker host memory."""

import asyncio
import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile

from irwkv.model_executor.models.rwkv_adapters import MAX_UPLOAD_BYTES

router = APIRouter()


async def _json_body(request: Request):
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "invalid JSON request") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    return body


async def _rpc(request: Request, operation: str, **kwargs):
    results = await request.app.state.engine_client.collective_rpc(
        "rwkv_adapter_rpc", args=(operation,), kwargs=kwargs
    )
    for result in results:
        if "error" in result:
            raise HTTPException(400, result["error"])
    return results[0]["result"]


async def _upload(request: Request, kind: str):
    metadata = None
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        async with request.form(max_files=2, max_fields=4) as form:
            files = [
                value
                for _, value in form.multi_items()
                if isinstance(value, UploadFile)
            ]
            checkpoints = [f for f in files if f.filename != "adapter_config.json"]
            if len(checkpoints) != 1:
                raise HTTPException(400, "upload requires exactly one PTH file")
            data = await checkpoints[0].read(MAX_UPLOAD_BYTES + 1)
            configs = [f for f in files if f.filename == "adapter_config.json"]
            if configs:
                if len(configs) != 1:
                    raise HTTPException(400, "duplicate adapter metadata")
                raw_metadata = await configs[0].read((8 << 20) + 1)
                if len(raw_metadata) > 8 << 20:
                    raise HTTPException(400, "adapter metadata exceeds 8 MiB")
                try:
                    metadata = json.loads(raw_metadata)
                except ValueError as exc:
                    raise HTTPException(400, "invalid adapter metadata JSON") from exc
            adapter_id = form.get("state_id" if kind == "state" else "adapter_id")
    else:
        body = await _json_body(request)
        if not isinstance(body, dict) or not isinstance(body.get("path"), str):
            raise HTTPException(400, "JSON upload requires a local path")

        def read_checkpoint():
            with Path(body["path"]).open("rb") as source:
                return source.read(MAX_UPLOAD_BYTES + 1)

        try:
            data = await asyncio.to_thread(read_checkpoint)
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc
        adapter_id = body.get("state_id" if kind == "state" else "adapter_id")
        metadata = body.get("metadata")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "checkpoint must contain 1 byte to 512 MiB")
    if kind == "state" and adapter_id is None:
        adapter_id = hashlib.sha256(data).hexdigest()
    if not isinstance(adapter_id, str) or not adapter_id:
        raise HTTPException(400, "missing adapter_id/state_id")
    if metadata is not None and not isinstance(metadata, dict):
        raise HTTPException(400, "metadata must be an object")
    result = await _rpc(
        request, "put", adapter_id=adapter_id, data=data, kind=kind, metadata=metadata
    )
    if kind == "state":
        result["state_id"] = result["adapter_id"]
    return result


@router.post("/v1/adapters")
async def upload_adapter(request: Request):
    return await _upload(request, "adapter")


@router.get("/v1/adapters")
async def list_adapters(request: Request):
    return await _rpc(request, "list")


@router.delete("/v1/adapters")
async def delete_adapter(request: Request):
    body = await _json_body(request)
    if not isinstance(body.get("adapter_id"), str) or not body["adapter_id"]:
        raise HTTPException(400, "missing adapter_id")
    return await _rpc(request, "delete", adapter_id=body.get("adapter_id"))


@router.post("/v1/state/upload")
async def upload_state(request: Request):
    return await _upload(request, "state")


@router.get("/v1/state/list")
async def list_states(request: Request):
    result = await _rpc(request, "list", kind="state")
    for entry in result["data"]:
        entry["state_id"] = entry["adapter_id"]
    return result


@router.post("/v1/state/delete")
async def delete_state(request: Request):
    body = await _json_body(request)
    if not isinstance(body.get("state_id"), str) or not body["state_id"]:
        raise HTTPException(400, "missing state_id")
    return await _rpc(request, "delete", kind="state", adapter_id=body.get("state_id"))


def attach_router(app: FastAPI):
    app.include_router(router)
