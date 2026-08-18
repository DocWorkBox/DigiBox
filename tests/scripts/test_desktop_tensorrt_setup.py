from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "desktop" / "build_tensorrt.ps1"
LAUNCHER = ROOT / "scripts" / "desktop" / "DigiBox-TensorRT-Setup.cmd"


def test_tensorrt_setup_is_powershell_51_safe_and_transactional() -> None:
    raw = SCRIPT.read_bytes()
    text = raw.decode("ascii")

    assert "[System.IO.Path]::GetRelativePath" not in text
    assert text.count("Assert-AvtrServicesStopped") >= 3
    assert "Restore-ActiveArtifacts" in text
    assert "--engine-path" in text
    assert "catch" in text
    assert "rollback" in text.casefold()


def test_tensorrt_setup_validates_staged_engines_before_installing() -> None:
    text = SCRIPT.read_text(encoding="ascii")

    staged_probe = text.index("Validate staged TensorRT engines")
    first_install = text.rindex("Install-ActiveArtifacts")

    assert staged_probe < first_install


def test_double_click_launcher_is_ascii_and_preserves_quoted_runtime_path() -> None:
    text = LAUNCHER.read_bytes().decode("ascii")

    assert 'set "RUNTIME_ROOT=%~1"' in text
    assert '-RuntimeRoot "%RUNTIME_ROOT%"' in text
    assert "Standard" in text
    assert "Full" in text
