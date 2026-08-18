from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUILDER = ROOT / "scripts" / "desktop" / "build_portable_runtime.ps1"
TAURI_BUILDER = ROOT / "scripts" / "build_tauri_windows.ps1"
ELECTRON_BUILDER = ROOT / "scripts" / "build_desktop_windows.ps1"
VFX_REQUIREMENTS = "requirements-windows-nvidia-vfx.txt"
VFX_WHEEL = "nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl"
VFX_WHEEL_SHA256 = (
    "b6cfaff5f435ad18329a1e1c1ac3ceb36f2aa6cfb0774d271c0bcc3aeaf31c53"
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the Runtime build contract"
    return executable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _portable_plan(tmp_path: Path) -> dict[str, object]:
    destination = tmp_path / "avtr-runtime"
    result = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNTIME_BUILDER),
            "-SourceRoot",
            str(ROOT),
            "-Destination",
            str(destination),
            "-Layout",
            "PortableV2",
            "-PlanOnly",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "PlanOnly must emit a machine-readable Runtime plan"
    assert not destination.exists(), "PlanOnly must remain read-only"
    return json.loads(lines[-1])


def test_portable_v2_plan_declares_nvidia_vfx_as_a_main_layer_requirement(
    tmp_path: Path,
) -> None:
    plan = _portable_plan(tmp_path)

    requirements = plan["dependencyRequirements"]
    assert requirements["main"] == [
        "requirements-windows.txt",
        VFX_REQUIREMENTS,
        "requirements-windows-tensorrt.txt",
    ]
    assert plan["nvidiaVfx"]["distribution"] == "nvidia-vfx==0.1.0.1"
    assert plan["nvidiaVfx"]["import"] == "nvvfx"
    assert plan["nvidiaVfx"]["layer"] == "packages/main"
    local_wheel = ROOT / ".downloads" / VFX_WHEEL
    local_is_verified = (
        local_wheel.is_file()
        and local_wheel.stat().st_size == 490_396_952
        and _sha256(local_wheel) == VFX_WHEEL_SHA256
    )
    assert plan["nvidiaVfx"]["source"] == (
        "verified-local-wheel" if local_is_verified else "pinned-requirement-url"
    )
    assert plan["nvidiaVfx"]["wheel"] == (
        str(local_wheel) if local_is_verified else None
    )
    assert plan["nvidiaVfx"]["sha256"] == VFX_WHEEL_SHA256


def test_runtime_builder_installs_and_bundles_the_vfx_requirement() -> None:
    source = RUNTIME_BUILDER.read_text(encoding="utf-8")

    assert (
        '$portableResolutionArguments + @($resolvedNvidiaVfxWheel)'
        in source
    )
    assert '"nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl"' in source
    assert "490396952" in source
    assert VFX_WHEEL_SHA256 in source.casefold()
    assert "Resolve-NvidiaVfxWheel" in source
    assert (
        '"-r", (Join-Path $resolvedSourceRoot '
        '"requirements-windows-nvidia-vfx.txt")'
        in source
    )
    assert "-Arguments $nvidiaVfxArguments" in source
    assert "Installing NVIDIA VFX Runtime into the main package layer" in source
    assert "Assert-NvidiaVfxLayerInventory" in source
    assert 'Where-Object { $_.name -eq "nvidia-vfx" }' in source
    assert '$matches[0].layer -ne "main"' in source
    assert '"requirements-windows-nvidia-vfx.txt"' in source


def test_full_delivery_gates_import_nvvfx_from_the_main_python_route() -> None:
    for builder in (TAURI_BUILDER, ELECTRON_BUILDER):
        source = builder.read_text(encoding="utf-8")

        assert "import torch,tensorrt,nvvfx,fastapi" in source
