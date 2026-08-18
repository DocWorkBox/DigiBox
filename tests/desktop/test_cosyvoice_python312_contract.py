from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_cosyvoice_dependencies_use_a_python312_pyworld_wheel() -> None:
    requirements = (ROOT / "requirements-windows-cosyvoice.txt").read_text(
        encoding="utf-8"
    )

    assert "pyworld==0.3.5" in requirements
    assert "pyworld==0.3.4" not in requirements


def test_cosyvoice_inference_sources_parse_with_python312_grammar() -> None:
    source_roots = [
        ROOT / "third_party" / "CosyVoice" / "cosyvoice",
        ROOT
        / "third_party"
        / "CosyVoice"
        / "third_party"
        / "Matcha-TTS"
        / "matcha",
        ROOT / "src" / "avaturn_live_streamer" / "integrations",
    ]
    failures: list[str] = []
    parsed = 0

    for source_root in source_roots:
        for source_path in source_root.rglob("*.py"):
            parsed += 1
            try:
                ast.parse(
                    source_path.read_text(encoding="utf-8"),
                    filename=str(source_path),
                    feature_version=(3, 12),
                )
            except (SyntaxError, UnicodeDecodeError) as error:
                failures.append(f"{source_path.relative_to(ROOT)}: {error}")

    assert parsed > 50
    assert failures == []
