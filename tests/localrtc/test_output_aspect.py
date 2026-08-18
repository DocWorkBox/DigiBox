from __future__ import annotations

import pytest

from avaturn_live_streamer.localrtc import worklet
from avaturn_live_streamer.types import PixelFormat


@pytest.mark.parametrize(
    ("aspect", "expected"),
    [
        ("16:9", (720, 1280)),
        ("9:16", (720, 404)),
        ("1:1", (720, 720)),
        ("4:3", (720, 960)),
        ("3:4", (720, 540)),
    ],
)
def test_output_aspect_center_crop_dimensions(aspect: str, expected: tuple[int, int]) -> None:
    assert worklet._crop_dimensions(720, 1280, aspect) == expected


def test_output_aspect_changes_the_actual_webrtc_video_frame_size() -> None:
    height, width = 720, 720
    i420 = bytes(height * 3 // 2 * width)

    frame = worklet._build_video_frame(
        i420,
        PixelFormat.YUV_I420,
        output_aspect="1:1",
    )

    assert (frame.height, frame.width) == (720, 720)


def test_rtx_frame_uses_renderer_reported_dimensions_without_cpu_rescale() -> None:
    height, width = 1080, 1920
    i420 = bytes(height * 3 // 2 * width)

    frame = worklet._build_video_frame(
        i420,
        PixelFormat.YUV_I420,
        input_dimensions=(height, width),
        output_dimensions=(height, width),
    )

    assert (frame.height, frame.width) == (1080, 1920)
