"""Isolated loopback worker for the official FeyNoBg model.

Run this module from ``.venv-feynobg``.  Keeping it outside AVTR-1's main
environment avoids upgrading the renderer's pinned Hugging Face stack merely
to process the occasional uploaded portrait.
"""

from __future__ import annotations

import gc
import io
import ipaddress
import os
import threading
from contextlib import nullcontext
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

_MODEL_ID = "feyninc/FeyNobg"
_MODEL_REVISION = "c1fd67fbefe3efeb78fe2a003270fb5350a0bb1c"
_MAX_UPLOAD_BYTES = 80 * 1024 * 1024


class ModelBusyError(RuntimeError):
    """Raised when an explicit unload races with an active cutout request."""


def _is_loopback_request(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    try:
        address = ipaddress.ip_address(client.host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped is not None and mapped.is_loopback)


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
    except (ImportError, OSError):
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        available = bool(cuda.is_available())
    except Exception:
        return
    if not available:
        return
    for cleanup_name in ("empty_cache", "ipc_collect"):
        cleanup = getattr(cuda, cleanup_name, None)
        if not callable(cleanup):
            continue
        try:
            cleanup()
        except Exception:
            continue


class CutoutService(Protocol):
    model_id: str
    device: str

    @property
    def loaded(self) -> bool: ...

    @property
    def released(self) -> bool: ...

    @property
    def active_requests(self) -> int: ...

    def cutout(self, image: Image.Image) -> Image.Image: ...

    def release(self) -> dict[str, str | bool | int]: ...


class FeyNoBgService:
    """Lazy, serialized wrapper around ``nobg``'s official model API."""

    def __init__(self, *, model_id: str, revision: str, device: str) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._active_requests = 0
        self._released = False

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def released(self) -> bool:
        return self._released

    @property
    def active_requests(self) -> int:
        return self._active_requests

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        from nobg import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
        ).eval().to(self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self._released = False

    def cutout(self, image: Image.Image) -> Image.Image:
        import torch

        with self._lock:
            self._active_requests += 1
            try:
                self._ensure_loaded()
                assert self._model is not None
                assert self._processor is not None
                rgb = image.convert("RGB")
                inputs = self._processor(rgb, return_tensors="pt").to(self.device)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.startswith("cuda")
                    else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    outputs = self._model(pixel_values=inputs["pixel_values"])
                alpha = self._processor.post_process_alpha_matting(
                    outputs,
                    target_sizes=[(rgb.height, rgb.width)],
                )[0]
                return self._processor.cutout(rgb, alpha).convert("RGBA")
            finally:
                self._active_requests -= 1

    def release(self) -> dict[str, str | bool | int]:
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("FeyNoBg has an active cutout request")
        try:
            if self._active_requests:
                raise ModelBusyError(
                    f"FeyNoBg has {self._active_requests} active cutout request(s)"
                )
            self._model = None
            self._processor = None
            self._released = True
            _release_runtime_memory()
            return {
                "service": "feynobg",
                "status": "released",
                "released": True,
                "loaded": False,
                "active_requests": 0,
            }
        finally:
            self._lock.release()


def _default_service() -> FeyNoBgService:
    return FeyNoBgService(
        model_id=os.environ.get("AVTR1_FEYNOBG_MODEL", _MODEL_ID),
        revision=os.environ.get("AVTR1_FEYNOBG_REVISION", _MODEL_REVISION),
        device=os.environ.get("AVTR1_FEYNOBG_DEVICE", "cpu"),
    )


def create_app(service: CutoutService | None = None) -> FastAPI:
    worker = _default_service() if service is None else service
    app = FastAPI(title="AVTR-1 FeyNoBg Worker")

    @app.get("/health")
    async def health() -> dict[str, str | bool | int]:
        return {
            "status": "released" if worker.released else "ok",
            "service": "feynobg",
            "model": worker.model_id,
            "device": worker.device,
            "loaded": worker.loaded,
            "released": worker.released,
            "active_requests": worker.active_requests,
        }

    @app.post("/release")
    async def release_runtime(request: Request) -> dict[str, str | bool | int]:
        if not _is_loopback_request(request):
            raise HTTPException(status_code=403, detail="release is restricted to loopback")
        try:
            return await run_in_threadpool(worker.release)
        except ModelBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/cutout")
    async def cutout(file: UploadFile) -> Response:
        payload = await file.read(_MAX_UPLOAD_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="上传图片为空")
        if len(payload) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="上传图片不能超过 80 MiB")
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="无法识别上传图片") from exc

        result = await run_in_threadpool(worker.cutout, image)
        output = io.BytesIO()
        result.convert("RGBA").save(output, format="PNG", optimize=True)
        return Response(
            output.getvalue(),
            media_type="image/png",
            headers={"X-AVTR-Cutout-Backend": "feynobg"},
        )

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "avaturn_live_streamer.integrations.feynobg_server:app",
        host="127.0.0.1",
        port=int(os.environ.get("AVTR1_FEYNOBG_PORT", "8767")),
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = ["FeyNoBgService", "ModelBusyError", "app", "create_app"]
