from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_desktop_windows.ps1"
MEMORY_PACKAGE_MODULES = (
    "__init__.py",
    "admin.py",
    "api.py",
    "extractor.py",
    "models.py",
    "paths.py",
    "schema.py",
    "service.py",
    "sqlite_store.py",
    "transfer.py",
    "worklet.py",
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the Electron build contract"
    return executable


def _run_build(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _plan(*arguments: str) -> dict[str, object]:
    result = _run_build(*arguments, "-PlanOnly")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, "PlanOnly must emit a machine-readable build plan"
    return json.loads(lines[-1])


def _write_file(root: Path, relative: str, content: str = "x") -> None:
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _invoke_build_function(
    tmp_path: Path,
    function_name: str,
    invocation: str,
) -> subprocess.CompletedProcess[str]:
    runner = tmp_path / f"invoke-{function_name}.ps1"
    runner.write_text(
        rf'''
param([Parameter(Mandatory = $true)][string]$BuildScript)
$ErrorActionPreference = "Stop"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $BuildScript,
    [ref]$tokens,
    [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {{
    throw "Build script failed to parse: $($parseErrors -join ', ')"
}}
$functionAst = $ast.Find(
    {{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "{function_name}"
    }},
    $true
)
if ($null -eq $functionAst) {{
    throw "{function_name} is missing"
}}
Invoke-Expression $functionAst.Extent.Text
{invocation}
'''.strip(),
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
            str(runner),
            "-BuildScript",
            str(BUILD_SCRIPT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_full_electron_plan_defaults_to_v2_and_documents_both_accepted_layouts() -> None:
    default = _plan("-Edition", "Full", "-SkipRuntimeBuild")
    legacy = _plan(
        "-Edition",
        "Full",
        "-SkipRuntimeBuild",
        "-RuntimeLayout",
        "LegacyV1",
    )

    assert default["runtimeBuildLayout"] == "PortableV2"
    assert default["acceptedRuntimeLayouts"] == ["portable-v1", "portable-v2"]
    assert legacy["runtimeBuildLayout"] == "LegacyV1"


def test_electron_plan_records_nonexistent_torch_wheelhouse_without_validating_it(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "future-wheelhouse"

    plan = _plan(
        "-Edition",
        "Full",
        "-TorchWheelhouse",
        str(wheelhouse),
    )

    assert Path(str(plan["torchWheelhouse"])) == wheelhouse.resolve()
    assert not wheelhouse.exists()


def test_electron_rejects_torch_wheelhouse_for_legacy_runtime_layout(
    tmp_path: Path,
) -> None:
    result = _run_build(
        "-Edition",
        "Full",
        "-RuntimeLayout",
        "LegacyV1",
        "-TorchWheelhouse",
        str(tmp_path / "wheelhouse"),
        "-PlanOnly",
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "TorchWheelhouse" in message
    assert "PortableV2" in message


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_electron_actual_runtime_build_requires_torch_wheelhouse_directory(
    tmp_path: Path,
    path_kind: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    wheelhouse = tmp_path / "torch-wheelhouse"
    if path_kind == "file":
        wheelhouse.write_text("not a directory", encoding="utf-8")

    result = _run_build(
        "-Edition",
        "Full",
        "-SourceRoot",
        str(source),
        "-RuntimeDestination",
        str(tmp_path / "runtime"),
        "-TorchWheelhouse",
        str(wheelhouse),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "Torch wheelhouse" in message
    assert "directory" in message.lower()


def test_electron_runtime_builder_receives_named_paths_layout_and_switches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source root"
    runtime = tmp_path / "runtime destination"
    wheelhouse = tmp_path / "torch wheelhouse"
    wheelhouse.mkdir()
    _write_file(source, "package.json", "{}")
    _write_file(source, "electron-builder-full.yml")

    fake_npm = tmp_path / "fake-npm.cmd"
    fake_npm.write_text("@exit /b 0\n", encoding="utf-8")
    observed_path = source / "scripts" / "desktop" / "runtime-builder-observed.json"
    _write_file(
        source,
        "scripts/desktop/build_portable_runtime.ps1",
        r'''
[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$Destination = "",
    [ValidateSet("PortableV2", "LegacyV1")][string]$Layout = "PortableV2",
    [string]$TorchWheelhouse = "",
    [switch]$SkipDependencies,
    [switch]$SkipModels,
    [switch]$Clean
)
$observed = [ordered]@{
    sourceRoot = $SourceRoot
    destination = $Destination
    layout = $Layout
    torchWheelhouse = $TorchWheelhouse
    clean = [bool]$Clean
    skipDependencies = [bool]$SkipDependencies
    skipModels = [bool]$SkipModels
}
[System.IO.File]::WriteAllText(
    (Join-Path $PSScriptRoot "runtime-builder-observed.json"),
    ($observed | ConvertTo-Json -Compress),
    [System.Text.UTF8Encoding]::new($false)
)
throw "RUNTIME_BUILDER_PROBE_STOP"
''',
    )

    result = _run_build(
        "-Edition",
        "Full",
        "-SourceRoot",
        str(source),
        "-RuntimeDestination",
        str(runtime),
        "-RuntimeLayout",
        "PortableV2",
        "-NpmExecutable",
        str(fake_npm),
        "-TorchWheelhouse",
        str(wheelhouse),
        "-CleanRuntime",
    )

    assert result.returncode != 0
    assert "RUNTIME_BUILDER_PROBE_STOP" in result.stdout + result.stderr
    assert observed_path.is_file()
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    assert Path(observed["sourceRoot"]) == source.resolve()
    assert Path(observed["destination"]) == runtime.resolve()
    assert Path(observed["torchWheelhouse"]) == wheelhouse.resolve()
    assert observed["layout"] == "PortableV2"
    assert observed["clean"] is True
    assert observed["skipDependencies"] is False
    assert observed["skipModels"] is False


def test_full_electron_validation_recognizes_v2_before_privacy_scan(tmp_path: Path) -> None:
    runtime = tmp_path / "avtr-runtime"
    manifest = {
        "schemaVersion": 2,
        "layout": "portable-v2",
        "runtimeId": "digibox-electron-v2-test",
        "paths": {
            "python": "python/python.exe",
            "orchestrator": "scripts/run_local_stream.py",
            "source": "src",
            "artifacts": "artifacts/main",
            "models": "models",
        },
        "python": {
            "version": "3.12.9",
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
            },
        },
        "components": {
            "dependenciesIncluded": True,
            "modelsIncluded": True,
            "frontendVendorIncluded": True,
            "tensorRtBuildInputsIncluded": True,
        },
        "tensorrt": {"engines": []},
    }
    for relative in (
        "python/python.exe",
        "scripts/desktop/build_tensorrt.ps1",
        "scripts/desktop/DigiBox-TensorRT-Setup.cmd",
        "scripts/desktop/inspect_runtime.py",
    ):
        _write_file(runtime, relative)
    for relative in (
        "packages/main",
        "packages/shared",
        "packages/cosyvoice",
        "packages/feynobg",
        "src",
        "third_party/CosyVoice",
        "third_party/CosyVoice/third_party/Matcha-TTS",
        "artifacts/main",
        "models",
    ):
        runtime.joinpath(*relative.split("/")).mkdir(parents=True, exist_ok=True)
    _write_file(runtime, "runtime-manifest.json", json.dumps(manifest))
    _write_file(runtime, "local_voices/private-reference.wav", "private")

    result = _run_build(
        "-Edition",
        "Full",
        "-SkipRuntimeBuild",
        "-RuntimeDestination",
        str(runtime),
        "-NpmExecutable",
        sys.executable,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "forbidden" in output.lower()
    assert "local_voices" in output.lower()
    assert "python-main" not in output.lower()
    assert "unsupported portable runtime" not in output.lower()


def test_v2_probe_and_privacy_contract_use_one_python_and_managed_layers() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert 'Join-Path $RuntimeRoot "python"' in source
    assert 'Join-Path $RuntimeRoot "packages"' in source
    assert "$Manifest.python.packageLayers.main" in source
    assert "$Manifest.python.packageLayers.cosyvoice" in source
    assert "$Manifest.python.packageLayers.feynobg" in source
    assert "Layout = $RuntimeLayout" in source
    for module_name in MEMORY_PACKAGE_MODULES:
        assert f'src\\avaturn_live_streamer\\memory\\{module_name}' in source


@pytest.mark.parametrize("missing_module", MEMORY_PACKAGE_MODULES)
def test_electron_required_payload_gate_requires_complete_memory_package(
    tmp_path: Path,
    missing_module: str,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for module_name in MEMORY_PACKAGE_MODULES:
        if module_name != missing_module:
            _write_file(
                runtime,
                f"src/avaturn_live_streamer/memory/{module_name}",
            )

    result = _invoke_build_function(
        tmp_path,
        "Assert-MemoryPackageEntrypoint",
        f'Assert-MemoryPackageEntrypoint -RuntimeRoot "{runtime}"',
    )

    assert result.returncode != 0
    assert missing_module in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("relative_path", "is_directory"),
    [
        ("memory.sqlite3", False),
        ("memory.sqlite3-wal", False),
        ("memory.sqlite3-shm", False),
        ("memory/backups", True),
        ("memory/pending-imports", True),
        ("exports/digibox-memory-20260817.json", False),
    ],
)
def test_electron_privacy_gate_rejects_memory_persistence_payloads(
    tmp_path: Path,
    relative_path: str,
    is_directory: bool,
) -> None:
    runtime = tmp_path / "runtime"
    target = runtime.joinpath(*relative_path.split("/"))
    if is_directory:
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("private", encoding="utf-8")

    result = _invoke_build_function(
        tmp_path,
        "Assert-NoForbiddenRuntimePayload",
        f'Assert-NoForbiddenRuntimePayload -RuntimeRoot "{runtime}"',
    )

    assert result.returncode != 0
    assert "forbidden" in (result.stdout + result.stderr).lower()


def test_full_dependency_probes_match_runtime_pythonpath_bootstrap() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '[Environment]::SetEnvironmentVariable("PYTHONPATH"' in source
    assert '[Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", "1"' in source
    assert "assert 'sitecustomize' in sys.modules" in source
    assert "sys.path[:0]=sys.argv[1:]" not in source
    assert '& $probe.python -I' not in source
    assert "import avaturn_live_streamer.local_stream_cli" in source
    assert "import avaturn_live_streamer.memory.admin" in source
    assert "import avaturn_live_streamer.memory.api" in source
