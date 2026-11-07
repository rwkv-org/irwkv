# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV-specific binary state endpoints alongside the OpenAI API."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter(prefix="/v1/rwkv/states")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


async def _operate(request: Request, operation: str, value=None):
    try:
        result = await request.app.state.engine_client.rwkv_state(operation, value)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(400, detail=result["error"])
        return result
    except (ValueError, KeyError) as error:
        raise HTTPException(400, detail=str(error)) from error


@router.get("")
async def list_states(request: Request):
    return {"data": await _operate(request, "list")}


@router.post("", status_code=201)
async def upload_state(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="RWKV state upload exceeds 512 MiB")
        data.extend(chunk)
    return await _operate(request, "import", bytes(data))


@router.get("/{state_id}")
async def download_state(request: Request, state_id: str):
    data = await _operate(request, "export", state_id)
    return Response(
        data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="rwkv.state"',
        },
    )


@router.delete("/{state_id}", status_code=204)
async def delete_state(request: Request, state_id: str):
    await _operate(request, "delete", state_id)
    return Response(status_code=204)
