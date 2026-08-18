from __future__ import annotations

import ast
import inspect
import textwrap

from avaturn_live_streamer import local_stream_cli
from avaturn_live_streamer.localrtc.worklet import LocalRTCWorklet


def test_session_uses_localrtc_worklet_public_constructor_keywords() -> None:
    """Keep the production call aligned with attrs' underscore-stripped API."""

    source = textwrap.dedent(inspect.getsource(local_stream_cli._run_session_inner))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LocalRTCWorklet"
    ]

    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    constructor_names = set(inspect.signature(LocalRTCWorklet).parameters)

    assert keyword_names == {
        "peer",
        "pixel_format",
        "output_aspect",
        "input_dimensions",
        "output_dimensions",
    }
    assert keyword_names <= constructor_names
