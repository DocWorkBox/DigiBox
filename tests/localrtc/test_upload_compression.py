from __future__ import annotations

from io import BytesIO
from typing import ClassVar

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from avaturn_live_streamer import local_stream_cli

_RENDERER_LIMIT = 12 * 1024 * 1024
_HARD_UPLOAD_LIMIT = 80 * 1024 * 1024


def _compressor():
    compress = getattr(local_stream_cli, "_compress_image_for_upload", None)
    assert callable(compress), "local image upload compression is not implemented"
    return compress


def _encode_image(
    *,
    image_format: str,
    size: tuple[int, int],
    seed: int = 7,
) -> bytes:
    width, height = size
    pixels = np.random.default_rng(seed).integers(
        0,
        256,
        (height, width, 3),
        dtype=np.uint8,
    )
    output = BytesIO()
    save_options = {"quality": 100, "subsampling": 0} if image_format == "JPEG" else {}
    Image.fromarray(pixels, mode="RGB").save(
        output,
        format=image_format,
        **save_options,
    )
    return output.getvalue()


def test_small_image_upload_is_returned_byte_for_byte() -> None:
    output = BytesIO()
    Image.new("RGB", (64, 48), (23, 47, 89)).save(output, format="PNG")
    payload = output.getvalue()

    compressed = _compressor()(payload, content_type="image/png")

    assert compressed == payload


@pytest.mark.parametrize(
    ("image_format", "content_type", "size"),
    [
        ("PNG", "image/png", (2400, 2400)),
        ("JPEG", "image/jpeg", (4000, 3000)),
    ],
)
def test_large_image_is_compressed_below_renderer_limit_and_remains_decodable(
    image_format: str,
    content_type: str,
    size: tuple[int, int],
) -> None:
    payload = _encode_image(image_format=image_format, size=size)
    assert _RENDERER_LIMIT < len(payload) <= _HARD_UPLOAD_LIMIT

    compressed = _compressor()(payload, content_type=content_type)

    assert len(compressed) <= _RENDERER_LIMIT
    with Image.open(BytesIO(compressed)) as decoded:
        decoded.load()
        assert decoded.format == image_format
        assert decoded.width <= size[0]
        assert decoded.height <= size[1]
        assert min(decoded.size) >= 720
        assert decoded.width / decoded.height == pytest.approx(size[0] / size[1], rel=0.01)


def test_image_above_hard_upload_limit_is_rejected_before_decoding() -> None:
    payload = b"not-an-image".ljust(_HARD_UPLOAD_LIMIT + 1, b"\0")

    with pytest.raises(HTTPException) as raised:
        _compressor()(payload, content_type="image/png")

    assert raised.value.status_code == 413


def test_upload_proxy_compresses_large_image_before_renderer_request(monkeypatch) -> None:
    payload = _encode_image(image_format="PNG", size=(2400, 2400))
    assert len(payload) > _RENDERER_LIMIT
    forwarded: list[tuple[str, bytes, str | None]] = []

    class FakeResponse:
        headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}

        def __init__(self, *, accepted: bool) -> None:
            self.status_code = 200 if accepted else 413
            self.content = (
                b'{"id":"user_background_compressed","kind":"background"}'
                if accepted
                else b'{"detail":"image exceeds 12 MiB"}'
            )

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            _ = args

        async def post(self, url, *, files):
            _ = url
            _filename, content, content_type = files["file"]
            forwarded.append((_filename, content, content_type))
            return FakeResponse(accepted=len(content) <= _RENDERER_LIMIT)

    monkeypatch.setattr(local_stream_cli, "_renderer_base_url", lambda: "http://renderer")
    monkeypatch.setattr(local_stream_cli.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        local_stream_cli._make_app(idle_timeout=60, max_duration=300),
        client=("127.0.0.1", 50000),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/assets/background",
        files={"file": ("large.png", payload, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "user_background_compressed"
    assert len(forwarded) == 1
    filename, compressed, content_type = forwarded[0]
    assert filename == "large.png"
    assert content_type == "image/png"
    assert len(compressed) <= _RENDERER_LIMIT
    with Image.open(BytesIO(compressed)) as decoded:
        decoded.verify()
