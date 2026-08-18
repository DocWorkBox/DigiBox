from __future__ import annotations

import asyncio
import inspect
import re
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from avaturn_live_streamer.memory.sqlite_store import StaleRevisionError
from avaturn_live_streamer.memory.transfer import (
    InvalidTransferError,
    StaleImportPlanError,
)


MAX_IMPORT_BYTES = 16 * 1024 * 1024
_CLEAR_CONFIRMATION = "清空全部本地记忆"
_DEFAULT_IMPORT_FILENAME = "digibox-memory-import.json"
_EXPORT_FILENAME = "digibox-memory.json"
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_LOOPBACK_AUTHORITY = re.compile(
    r"(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?",
    re.IGNORECASE,
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


class MemoryAdmin(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def available(self) -> bool: ...

    @property
    def degraded_reason(self) -> str | None: ...

    async def stats(self) -> object: ...

    async def list_records(
        self,
        *,
        kind: str | None = None,
        state: str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 30,
    ) -> object: ...

    async def get_record(self, memory_id: str) -> object | None: ...

    async def forget(
        self,
        memory_id: str,
        *,
        expected_revision: int,
        reason: str,
    ) -> object: ...

    async def clear_all(
        self,
        *,
        expected_db_revision: int,
        backup: bool = True,
    ) -> object: ...

    async def export_json(self, destination: Path) -> object: ...

    async def preview_import(self, source: Path) -> object: ...

    async def apply_import(self, plan: Mapping[str, object] | str) -> object: ...


LoopbackGuard = Callable[[Request], object | Awaitable[object]]


class _ClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str
    expected_db_revision: StrictInt = Field(ge=0)


class _ApplyImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_token: str = Field(min_length=1, max_length=512)
    expected_db_revision: StrictInt = Field(ge=0)


def _json_response(value: object) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(value),
        headers=_NO_STORE_HEADERS,
    )


def _detail(error: Exception, fallback: str) -> str:
    return (str(error).strip() or fallback)[:500]


def _require_available(admin: MemoryAdmin) -> None:
    if admin.available:
        return
    raise HTTPException(
        status_code=503,
        detail=admin.degraded_reason or "memory service unavailable",
    )


def _is_loopback_authority(authority: str) -> bool:
    match = _LOOPBACK_AUTHORITY.fullmatch(authority)
    if match is None:
        return False
    port = match.group(1)
    return port is None or 0 < int(port) <= 65535


def _is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        bool(parsed.scheme)
        and _is_loopback_authority(parsed.netloc)
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _require_trusted_http_headers(request: Request) -> None:
    host_values = request.headers.getlist("host")
    if len(host_values) != 1 or not _is_loopback_authority(host_values[0]):
        raise HTTPException(status_code=403, detail="untrusted Host header")
    origin_values = request.headers.getlist("origin")
    if len(origin_values) > 1 or (
        origin_values and not _is_loopback_origin(origin_values[0])
    ):
        raise HTTPException(status_code=403, detail="untrusted Origin header")


async def _read_import_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="invalid Content-Length",
            ) from error
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="invalid Content-Length")
        if declared_length > MAX_IMPORT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="import exceeds the 16 MiB limit",
            )

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_IMPORT_BYTES:
            raise HTTPException(
                status_code=413,
                detail="import exceeds the 16 MiB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_import_filename(header_value: str | None) -> str:
    if header_value is None:
        return _DEFAULT_IMPORT_FILENAME
    value = header_value.strip()
    if not value or len(value) > 255 or any(ord(character) < 32 for character in value):
        raise HTTPException(status_code=400, detail="invalid import filename")
    filename = value.replace("\\", "/").rsplit("/", 1)[-1]
    device_name = filename.split(".", 1)[0].rstrip(" .").upper()
    if (
        filename in {"", ".", ".."}
        or filename.endswith((" ", "."))
        or any(character in '<>:"|?*' for character in filename)
        or device_name in _WINDOWS_RESERVED_NAMES
    ):
        raise HTTPException(status_code=400, detail="invalid import filename")
    return filename


def create_memory_router(
    admin: MemoryAdmin,
    require_loopback: LoopbackGuard,
) -> APIRouter:
    """Create the loopback-only local memory administration router."""

    class _LoopbackRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
            route_handler = super().get_route_handler()

            async def guarded_route_handler(request: Request) -> Response:
                _require_trusted_http_headers(request)
                guard_result = require_loopback(request)
                if inspect.isawaitable(guard_result):
                    await guard_result
                return await route_handler(request)

            return guarded_route_handler

    router = APIRouter(
        prefix="/memory",
        route_class=_LoopbackRoute,
    )

    @router.get("/stats")
    async def memory_stats() -> JSONResponse:
        base: dict[str, object] = {
            "enabled": admin.enabled,
            "available": admin.available,
            "degraded_reason": admin.degraded_reason,
            "counts": {},
            "db_revision": None,
        }
        if not admin.available:
            return _json_response(base)
        try:
            result = await admin.stats()
            if not isinstance(result, Mapping):
                raise TypeError("memory stats must be a mapping")
        except Exception as error:
            base["available"] = False
            base["degraded_reason"] = _detail(error, "memory stats unavailable")
            return _json_response(base)
        base.update(result)
        base["enabled"] = admin.enabled
        base["available"] = admin.available
        base["degraded_reason"] = admin.degraded_reason
        return _json_response(base)

    @router.get("/records")
    async def list_memory_records(
        kind: Literal["person", "relationship", "event"] | None = None,
        state: Literal["candidate", "confirmed"] | None = None,
        q: str | None = Query(default=None, max_length=200),
        cursor: str | None = Query(default=None, max_length=2048),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> JSONResponse:
        _require_available(admin)
        try:
            result = await admin.list_records(
                kind=kind,
                state=state,
                q=q,
                cursor=cursor,
                limit=limit,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=_detail(error, "invalid memory records query"),
            ) from error
        return _json_response(result)

    @router.get("/records/{memory_id}")
    async def get_memory_record(memory_id: str) -> JSONResponse:
        _require_available(admin)
        result = await admin.get_record(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail="memory record not found")
        return _json_response(result)

    @router.delete("/records/{memory_id}")
    async def delete_memory_record(
        memory_id: str,
        expected_revision: int = Query(ge=1),
    ) -> JSONResponse:
        _require_available(admin)
        try:
            result = await admin.forget(
                memory_id,
                expected_revision=expected_revision,
                reason="user_deleted",
            )
        except StaleRevisionError as error:
            raise HTTPException(
                status_code=409,
                detail=_detail(error, "memory revision changed"),
            ) from error
        if result is None:
            raise HTTPException(status_code=404, detail="memory record not found")
        return _json_response(result)

    @router.post("/clear")
    async def clear_memory(request: _ClearRequest) -> JSONResponse:
        _require_available(admin)
        if request.confirmation != _CLEAR_CONFIRMATION:
            raise HTTPException(
                status_code=400,
                detail="confirmation text does not match",
            )
        try:
            result = await admin.clear_all(
                expected_db_revision=request.expected_db_revision,
                backup=True,
            )
        except StaleRevisionError as error:
            raise HTTPException(
                status_code=409,
                detail=_detail(error, "database revision changed"),
            ) from error
        return _json_response(result)

    @router.get("/export")
    async def export_memory() -> Response:
        _require_available(admin)
        try:
            with tempfile.TemporaryDirectory(prefix="digibox-memory-export-") as temp:
                destination = Path(temp) / _EXPORT_FILENAME
                await admin.export_json(destination)
                payload = await asyncio.to_thread(destination.read_bytes)
        except InvalidTransferError as error:
            raise HTTPException(
                status_code=400,
                detail=_detail(error, "memory export failed"),
            ) from error
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{_EXPORT_FILENAME}"',
                **_NO_STORE_HEADERS,
            },
        )

    @router.post("/import/preview")
    async def preview_memory_import(request: Request) -> JSONResponse:
        _require_available(admin)
        content_type = request.headers.get("content-type", "").partition(";")[0]
        if content_type.strip().lower() != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json",
            )
        payload = await _read_import_body(request)
        filename = _safe_import_filename(request.headers.get("x-digibox-filename"))
        try:
            with tempfile.TemporaryDirectory(prefix="digibox-memory-import-") as temp:
                source = Path(temp) / filename
                await asyncio.to_thread(source.write_bytes, payload)
                result = await admin.preview_import(source)
        except InvalidTransferError as error:
            raise HTTPException(
                status_code=400,
                detail=_detail(error, "invalid memory import"),
            ) from error
        return _json_response(result)

    @router.post("/import/apply")
    async def apply_memory_import(request: _ApplyImportRequest) -> JSONResponse:
        _require_available(admin)
        try:
            result = await admin.apply_import(request.model_dump())
        except InvalidTransferError as error:
            raise HTTPException(
                status_code=400,
                detail=_detail(error, "invalid memory import"),
            ) from error
        except StaleImportPlanError as error:
            raise HTTPException(
                status_code=409,
                detail=_detail(error, "import preview is stale"),
            ) from error
        return _json_response(result)

    return router
