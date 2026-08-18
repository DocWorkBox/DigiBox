from __future__ import annotations

import numpy as np
import pytest

from avtr1_renderer import feynobg as subject


def test_missing_feynobg_backend_fails_instead_of_returning_an_opaque_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AVTR1_FEYNOBG_URL", "")

    def unavailable(_image: np.ndarray) -> np.ndarray:
        raise ImportError("nobg is not installed")

    monkeypatch.setattr(subject, "local_feynobg_cutout", unavailable)

    with pytest.raises(RuntimeError, match="FeyNoBg unavailable"):
        subject.feynobg_cutout(np.zeros((4, 4, 3), dtype=np.uint8))
