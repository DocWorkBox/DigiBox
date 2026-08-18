from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TAURI_BUILD = ROOT / "scripts" / "build_tauri_windows.ps1"
ELECTRON_BUILD = ROOT / "scripts" / "build_desktop_windows.ps1"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the Runtime validation contract"
    return executable


def _write_fake_python(runtime: Path) -> Path:
    python = runtime / "python" / "python.cmd"
    python.parent.mkdir(parents=True)
    python.write_text(
        r"""@echo off
echo %PYTHONDONTWRITEBYTECODE%^|%PYTHONNOUSERSITE%^|%*>>"%~dp0..\observed-environment.txt"
if not "%PYTHONDONTWRITEBYTECODE%"=="1" mkdir "%~dp0..\src\__pycache__" >nul 2>nul
echo {"schema_version":1,"platform":"windows","artifacts_ready":true,"models_ready":true,"engine_files":[]}
exit /b 0
""",
        encoding="utf-8",
    )
    return python


def _write_runtime(runtime: Path) -> Path:
    python = _write_fake_python(runtime)
    for relative in (
        "packages/main",
        "packages/cosyvoice",
        "packages/feynobg",
        "packages/shared",
        "src",
        "third_party/CosyVoice",
        "third_party/CosyVoice/third_party/Matcha-TTS",
        "scripts/desktop",
    ):
        runtime.joinpath(*relative.split("/")).mkdir(parents=True, exist_ok=True)
    (runtime / "scripts" / "desktop" / "inspect_runtime.py").write_text(
        "# fake inspector\n", encoding="utf-8"
    )
    manifest = {
        "schemaVersion": 2,
        "layout": "portable-v2",
        "paths": {"python": "python/python.cmd"},
        "python": {
            "packageLayers": {
                "main": ["packages/main", "packages/shared", "src"],
                "cosyvoice": [
                    "packages/cosyvoice",
                    "packages/shared",
                    "third_party/CosyVoice",
                    "third_party/CosyVoice/third_party/Matcha-TTS",
                    "src",
                ],
                "feynobg": ["packages/feynobg", "packages/shared", "src"],
            }
        },
    }
    manifest_path = runtime / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return python


def _run_validation_harness(
    tmp_path: Path,
    build_script: Path,
    function_name: str,
    initial_value: str | None,
) -> subprocess.CompletedProcess[str]:
    runtime = tmp_path / "avtr-runtime"
    _write_runtime(runtime)
    harness = tmp_path / "invoke-runtime-validation.ps1"
    harness.write_text(
        r"""
param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot,
    [Parameter(Mandatory = $true)][string]$FunctionName,
    [Parameter(Mandatory = $true)][string]$InitialValue
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
$requiredFunctions = @(
    "Resolve-RuntimeManifestPath",
    "Resolve-RuntimePythonPath",
    "Resolve-RuntimePackageLayers",
    $FunctionName
) | Select-Object -Unique
foreach ($requiredFunction in $requiredFunctions) {
    $functionAst = $ast.Find(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                $node.Name -eq $requiredFunction
        },
        $true
    )
    if ($null -ne $functionAst) {
        Invoke-Expression $functionAst.Extent.Text
    }
}
$manifest = Get-Content -LiteralPath (Join-Path $RuntimeRoot "runtime-manifest.json") -Raw |
    ConvertFrom-Json
if ($InitialValue -eq "__ABSENT__") {
    [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", $null, "Process")
} else {
    [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", $InitialValue, "Process")
}
if ($FunctionName -eq "Assert-RuntimeInspection") {
    Assert-RuntimeInspection -RuntimeRoot $RuntimeRoot -Manifest $manifest
} else {
    Assert-RuntimeDependencyImports -RuntimeRoot $RuntimeRoot -Manifest $manifest
}
$after = [Environment]::GetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "Process")
[ordered]@{
    after = $after
    cacheExists = Test-Path -LiteralPath (Join-Path $RuntimeRoot "src\__pycache__")
} | ConvertTo-Json -Compress
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
            "-FunctionName",
            function_name,
            "-InitialValue",
            initial_value if initial_value is not None else "__ABSENT__",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize(
    ("build_script", "function_name"),
    (
        (TAURI_BUILD, "Assert-RuntimeInspection"),
        (TAURI_BUILD, "Assert-RuntimeDependencyImports"),
        (ELECTRON_BUILD, "Assert-RuntimeDependencyImports"),
    ),
    ids=("tauri-inspector", "tauri-imports", "electron-imports"),
)
@pytest.mark.parametrize("initial_value", (None, "original-setting"), ids=("absent", "restore"))
def test_runtime_validation_disables_bytecode_writes_and_restores_environment(
    tmp_path: Path,
    build_script: Path,
    function_name: str,
    initial_value: str | None,
) -> None:
    result = _run_validation_harness(
        tmp_path, build_script, function_name, initial_value
    )

    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    state = json.loads(lines[-1])
    expected_after = initial_value
    assert state == {"after": expected_after, "cacheExists": False}
    observed = (tmp_path / "avtr-runtime" / "observed-environment.txt").read_text(
        encoding="utf-8"
    )
    assert observed.splitlines()
    assert all(line.startswith("1|") and "|-B" in line for line in observed.splitlines())


@pytest.mark.parametrize("build_script", (TAURI_BUILD, ELECTRON_BUILD), ids=("tauri", "electron"))
def test_preexisting_pycache_remains_forbidden(
    tmp_path: Path, build_script: Path
) -> None:
    runtime = tmp_path / "avtr-runtime"
    cache = runtime / "src" / "package" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.cpython-312.pyc").write_bytes(b"existing cache")
    harness = tmp_path / "invoke-privacy-scan.ps1"
    harness.write_text(
        r"""
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
$functionAst = $ast.Find(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq "Assert-NoForbiddenRuntimePayload"
    },
    $true
)
Invoke-Expression $functionAst.Extent.Text
Assert-NoForbiddenRuntimePayload -RuntimeRoot $RuntimeRoot
""".strip(),
        encoding="utf-8",
    )
    result = subprocess.run(
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

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "forbidden" in output.lower()
    assert "__pycache__" in output.lower()
