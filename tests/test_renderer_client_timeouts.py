from __future__ import annotations

from avaturn_live_streamer.renderer.client import _RENDERER_CALL_TIMEOUTS


def test_renderer_call_tolerates_native_windows_response_latency() -> None:
    """The portable Windows renderer can take seconds between streamed frames."""

    assert _RENDERER_CALL_TIMEOUTS.connect >= 0.5
    assert _RENDERER_CALL_TIMEOUTS.read >= 30.0
