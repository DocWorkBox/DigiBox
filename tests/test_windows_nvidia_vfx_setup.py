from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_setup_validates_the_full_nvidia_vfx_runtime() -> None:
    setup = (ROOT / "scripts" / "setup_windows.ps1").read_text(encoding="utf-8")
    requirements = ROOT / "requirements-windows-nvidia-vfx.txt"

    assert "[switch]$EnableNvidiaVfx" in setup
    assert "if ($EnableNvidiaVfx)" in setup
    assert '"import nvvfx"' in setup
    assert "pip install" not in setup.lower()
    assert "scripts/download_nvidia_vfx_runtime.py" not in setup
    assert requirements.is_file()
    assert (ROOT / "scripts" / "download_nvidia_vfx_runtime.py").is_file()

    requirement_text = requirements.read_text(encoding="utf-8")
    assert "nvidia-vfx @ https://pypi.nvidia.com/nvidia-vfx/" in requirement_text
    assert "nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl" in requirement_text
    assert (
        "#sha256=b6cfaff5f435ad18329a1e1c1ac3ceb36f2aa6cfb0774d271c0bcc3aeaf31c53"
        in requirement_text
    )
