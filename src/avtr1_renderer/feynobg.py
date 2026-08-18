"""Optional FeyNoBg person cutout support.

The main renderer environment intentionally keeps its older Hugging Face
dependency set.  FeyNoBg currently requires a newer ``nobg`` stack, so the
preferred Windows deployment is a tiny loopback service in its own virtual
environment.  A direct in-process model is still supported when ``nobg`` is
already installed.  If neither backend is available, the upload fails clearly
instead of persisting an opaque image that looks like a valid cutout.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

LOG = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "feyninc/FeyNobg"
_DEFAULT_REVISION = "c1fd67fbefe3efeb78fe2a003270fb5350a0bb1c"
_DEFAULT_SERVICE_URL = "http://127.0.0.1:8767"


@dataclass(slots=True)
class _LocalFeyNoBg:
    model_id: str
    device: str
    model: object
    processor: object

    @classmethod
    def load(cls) -> _LocalFeyNoBg:
        import torch
        from nobg import AutoModel, AutoProcessor

        model_id = os.environ.get("AVTR1_FEYNOBG_MODEL", _DEFAULT_MODEL_ID).strip()
        revision = os.environ.get("AVTR1_FEYNOBG_REVISION", _DEFAULT_REVISION).strip()
        device = os.environ.get("AVTR1_FEYNOBG_DEVICE", "cpu").strip() or "cpu"
        model = AutoModel.from_pretrained(model_id, revision=revision).eval().to(device)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        LOG.info("FeyNoBg loaded", extra={"model_id": model_id, "device": device})
        return cls(model_id=model_id, device=device, model=model, processor=processor)

    def cutout(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb, mode="RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(pixel_values=inputs["pixel_values"])
        alpha = self.processor.post_process_alpha_matting(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0]
        rgba = np.asarray(self.processor.cutout(image, alpha).convert("RGBA"))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


_LOCAL_LOCK = threading.Lock()
_LOCAL_MODEL: _LocalFeyNoBg | None = None


def local_feynobg_cutout(image_bgr: np.ndarray) -> np.ndarray:
    """Run the official NoBg model in-process and return a BGRA cutout."""

    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        with _LOCAL_LOCK:
            if _LOCAL_MODEL is None:
                _LOCAL_MODEL = _LocalFeyNoBg.load()
    return _LOCAL_MODEL.cutout(image_bgr)


def _service_cutout(image_bgr: np.ndarray, service_url: str) -> np.ndarray:
    ok, png = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode image for FeyNoBg")
    timeout = httpx.Timeout(connect=1.0, read=180.0, write=30.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{service_url.rstrip('/')}/v1/cutout",
            files={"file": ("person.png", png.tobytes(), "image/png")},
        )
        response.raise_for_status()
    cutout = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    if cutout is None or cutout.ndim != 3 or cutout.shape[2] != 4:
        raise RuntimeError("FeyNoBg service returned an invalid RGBA image")
    return cutout


def feynobg_cutout(image_bgr: np.ndarray) -> np.ndarray:
    """Try the isolated service, then direct NoBg, or fail the upload."""

    service_url = os.environ.get("AVTR1_FEYNOBG_URL", _DEFAULT_SERVICE_URL).strip()
    if service_url:
        try:
            return _service_cutout(image_bgr, service_url)
        except (httpx.HTTPError, RuntimeError) as exc:
            LOG.warning("FeyNoBg loopback service unavailable: %s", exc)

    try:
        return local_feynobg_cutout(image_bgr)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        LOG.error("FeyNoBg unavailable: %s", exc)
        raise RuntimeError("FeyNoBg unavailable") from exc


__all__ = ["feynobg_cutout", "local_feynobg_cutout"]
