from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPTS = (
    ROOT / "scripts" / "build_tauri_windows.ps1",
    ROOT / "scripts" / "build_desktop_windows.ps1",
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the privacy contract"
    return executable


def _run_privacy_scan(
    tmp_path: Path,
    build_script: Path,
    relative_pem_path: str,
) -> subprocess.CompletedProcess[str]:
    runtime = tmp_path / "avtr-runtime"
    pem = runtime.joinpath(*relative_pem_path.split("/"))
    pem.parent.mkdir(parents=True, exist_ok=True)
    pem.write_text("test certificate", encoding="utf-8")

    harness = tmp_path / "invoke-privacy-scan.ps1"
    harness.write_text(
        """
param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot
)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $BuildScript,
    [ref]$tokens,
    [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "Build script failed to parse: $($parseErrors -join ', ')"
}
$functionAst = $ast.Find(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq "Assert-NoForbiddenRuntimePayload"
    },
    $true
)
if ($null -eq $functionAst) {
    throw "Assert-NoForbiddenRuntimePayload was not found"
}
Invoke-Expression $functionAst.Extent.Text
Assert-NoForbiddenRuntimePayload -RuntimeRoot $RuntimeRoot
""".strip(),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-BuildScript",
            str(build_script),
            "-RuntimeRoot",
            str(runtime),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize("build_script", BUILD_SCRIPTS, ids=lambda path: path.stem)
@pytest.mark.parametrize(
    "relative_pem_path",
    (
        "packages/shared/certifi/cacert.pem",
        "python-main/Lib/site-packages/certifi/cacert.pem",
        "python/Lib/site-packages/pip/_vendor/certifi/cacert.pem",
    ),
    ids=("portable-v2-shared-certifi", "portable-v1-certifi", "managed-pip-certifi"),
)
def test_managed_certifi_ca_bundles_are_allowed(
    tmp_path: Path,
    build_script: Path,
    relative_pem_path: str,
) -> None:
    result = _run_privacy_scan(tmp_path, build_script, relative_pem_path)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("build_script", BUILD_SCRIPTS, ids=lambda path: path.stem)
@pytest.mark.parametrize(
    "relative_pem_path",
    (
        "packages/shared/private-client.pem",
        "outside-managed/certifi/cacert.pem",
    ),
    ids=("other-managed-pem", "unmanaged-certifi-lookalike"),
)
def test_other_pem_files_remain_forbidden(
    tmp_path: Path,
    build_script: Path,
    relative_pem_path: str,
) -> None:
    result = _run_privacy_scan(tmp_path, build_script, relative_pem_path)

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "forbidden" in output.lower()
    assert relative_pem_path.replace("/", "\\").lower() in output.lower()
