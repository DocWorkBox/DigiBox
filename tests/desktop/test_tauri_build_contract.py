from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_tauri_windows.ps1"
ROOT_TENSORRT_LAUNCHER_SOURCE = (
    ROOT / "scripts" / "desktop" / "DigiBox-TensorRT-Setup-Root.cmd"
)
FULL_CONFIG = ROOT / "src-tauri" / "tauri.full.conf.json"
STANDARD_OFFLINE_CONFIG = (
    ROOT / "src-tauri" / "target" / "digibox-nsis-offline" / "tauri.offline.conf.json"
)
WEBVIEW2_INSTALLER_NAME = "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
WEBVIEW2_INSTALLER_SIZE = 209_653_456
WEBVIEW2_INSTALLER_SHA256 = (
    "F8D4AB074C22A0CD136434F37C6B34DFB64EBF8A32CE42E03BD8F2A6B51A3892"
)
REQUIRED_LICENSES = {
    "LICENSE.md",
    "LICENSE-MODEL.md",
    "LICENSE-RENDERER.md",
    "LICENSE-STREAMER.md",
    "PATENTS.md",
    "THIRD-PARTY-NOTICES.md",
}
TENSORRT_HELPERS = {
    "scripts/desktop/build_tensorrt.ps1",
    "scripts/desktop/DigiBox-TensorRT-Setup.cmd",
    "scripts/desktop/inspect_runtime.py",
}
REQUIRED_COSYVOICE_MODEL_FILES = {
    "models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/generation_config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/model.safetensors",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/tokenizer_config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/merges.txt",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/vocab.json",
}
TARGET_MACHINE_ENGINE_MANIFEST = "artifacts/main/engine-manifest.json"
MEMORY_PACKAGE_MODULES = (
    "src/avaturn_live_streamer/memory/__init__.py",
    "src/avaturn_live_streamer/memory/admin.py",
    "src/avaturn_live_streamer/memory/api.py",
    "src/avaturn_live_streamer/memory/extractor.py",
    "src/avaturn_live_streamer/memory/models.py",
    "src/avaturn_live_streamer/memory/paths.py",
    "src/avaturn_live_streamer/memory/schema.py",
    "src/avaturn_live_streamer/memory/service.py",
    "src/avaturn_live_streamer/memory/sqlite_store.py",
    "src/avaturn_live_streamer/memory/transfer.py",
    "src/avaturn_live_streamer/memory/worklet.py",
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the Tauri build contract"
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
    output = [line for line in result.stdout.splitlines() if line.strip()]
    assert output, "PlanOnly must emit a machine-readable build plan"
    return json.loads(output[-1])


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


def _write_file(root: Path, relative: str, content: str = "x") -> None:
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _write_runtime_manifest(runtime: Path, layout: str) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    portable_v2 = layout == "portable-v2"
    manifest: dict[str, object] = {
        "schemaVersion": 2 if portable_v2 else 1,
        "layout": layout,
        "paths": (
            {
                "python": "python/python.exe",
                "orchestrator": "scripts/run_local_stream.py",
                "source": "src",
                "artifacts": "artifacts/main",
                "models": "models",
            }
            if portable_v2
            else {
                "mainPython": "python-main/python.exe",
                "cosyvoicePython": "python-cosyvoice/python.exe",
                "feynobgPython": "python-feynobg/python.exe",
                "orchestrator": "scripts/run_local_stream.py",
                "source": "src",
                "artifacts": "artifacts/main",
                "models": "models",
            }
        ),
        "python": (
            {
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
            }
            if portable_v2
            else {"main": "3.12.9", "cosyvoice": "3.10.17", "feynobg": "3.12.9"}
        ),
        "components": {
            "dependenciesIncluded": True,
            "modelsIncluded": True,
            "frontendVendorIncluded": True,
            "tensorRtBuildInputsIncluded": True,
        },
        "privacy": {
            "userAssetsIncluded": False,
            "localVoiceCacheIncluded": False,
            "machineSpecificEnginesIncluded": False,
        },
    }
    (runtime / "runtime-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_standard_plan_builds_an_nsis_shell_without_a_runtime() -> None:
    plan = _plan("-Edition", "Standard")

    assert plan["schemaVersion"] == 1
    assert plan["kind"] == "avtr1-tauri-build-plan"
    assert plan["edition"] == "standard"
    assert plan["target"] == "installer"
    assert plan["bundleMode"] == "nsis"
    assert plan["config"] == "src-tauri/target/digibox-nsis-offline/tauri.offline.conf.json"
    assert plan["includesRuntime"] is False
    assert plan["buildsRuntime"] is False
    assert plan["runtimeResourceTarget"] is None
    assert plan["tauriArguments"][-2:] == ["--bundles", "nsis"]


def test_standard_plan_uses_a_pinned_local_offline_webview2_payload() -> None:
    plan = _plan("-Edition", "Standard")
    webview2 = plan["offlineWebView2"]

    assert webview2["enabled"] is True
    assert webview2["downloadsDuringBuild"] is False
    assert webview2["generatedConfig"] == (
        "src-tauri/target/digibox-nsis-offline/tauri.offline.conf.json"
    )
    assert webview2["fileName"] == WEBVIEW2_INSTALLER_NAME
    assert webview2["size"] == WEBVIEW2_INSTALLER_SIZE
    assert webview2["sha256"] == WEBVIEW2_INSTALLER_SHA256
    assert webview2["authenticodeStatus"] == "Valid"
    assert webview2["signerSubject"] == "CN=Microsoft Corporation"


def test_full_plan_uses_an_unbundled_shell_and_zip64_delivery() -> None:
    plan = _plan("-Edition", "Full")

    assert plan["edition"] == "full"
    assert plan["target"] == "archive"
    assert plan["bundleMode"] == "zip64"
    assert plan["config"] == "src-tauri/tauri.full.conf.json"
    assert plan["includesRuntime"] is True
    assert plan["buildsRuntime"] is True
    assert plan["runtimeResourceTarget"] == "avtr-runtime"
    assert "--no-bundle" in plan["tauriArguments"]
    assert plan["archiveFormat"] == "zip64"
    assert plan["singleFileInstaller"] is False
    assert plan["rootTensorRtLauncher"] == "DigiBox-TensorRT-Setup.cmd"


def test_root_tensorrt_launcher_forwards_sibling_runtime_and_exit_code(
    tmp_path: Path,
) -> None:
    assert ROOT_TENSORRT_LAUNCHER_SOURCE.is_file()
    launcher = tmp_path / "DigiBox-TensorRT-Setup.cmd"
    shutil.copy2(ROOT_TENSORRT_LAUNCHER_SOURCE, launcher)
    runtime = tmp_path / "avtr-runtime"
    helper = runtime / "scripts" / "desktop" / "DigiBox-TensorRT-Setup.cmd"
    observed = tmp_path / "observed-runtime.txt"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "@echo off\n"
        f'> "{observed}" echo %~f1\n'
        "exit /b 37\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "call", str(launcher)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 37, result.stdout + result.stderr
    assert Path(observed.read_text(encoding="utf-8").strip()) == runtime.resolve()


def test_full_delivery_stages_and_validates_root_tensorrt_launcher_before_archive() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '$rootTensorRtLauncherName = "DigiBox-TensorRT-Setup.cmd"' in source
    assert (
        '$rootTensorRtLauncherSource = Join-Path $root '
        '"scripts\\desktop\\DigiBox-TensorRT-Setup-Root.cmd"'
    ) in source
    assert (
        "Copy-Item -LiteralPath $rootTensorRtLauncherSource "
        "-Destination $rootTensorRtLauncherTarget"
    ) in source
    assert (
        "Assert-FileExists -Path $rootTensorRtLauncherTarget "
        '-Description "Root TensorRT setup launcher"'
    ) in source
    assert source.index(
        "Assert-FileExists -Path $rootTensorRtLauncherTarget"
    ) < source.index('& $tar "-a" "-c" "-f" $fullArchivePath')
    assert source.index(
        'Assert-NoForbiddenRuntimePayload -RuntimeRoot '
        '(Join-Path $fullUnpackedRoot "avtr-runtime")'
    ) < source.index('& $tar "-a" "-c" "-f" $fullArchivePath')


def test_tauri_plan_records_nonexistent_torch_wheelhouse_without_validating_it(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "future-wheelhouse"

    plan = _plan("-Edition", "Full", "-TorchWheelhouse", str(wheelhouse))

    assert Path(str(plan["torchWheelhouse"])) == wheelhouse.resolve()
    assert not wheelhouse.exists()


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_tauri_actual_runtime_build_requires_torch_wheelhouse_directory(
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
        "-OutputDirectory",
        str(tmp_path / "output"),
        "-TorchWheelhouse",
        str(wheelhouse),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "Torch wheelhouse" in message
    assert "directory" in message.lower()


def test_tauri_runtime_builder_receives_named_paths_and_switches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source root"
    runtime = tmp_path / "runtime destination"
    output = tmp_path / "delivery output"
    wheelhouse = tmp_path / "torch wheelhouse"
    wheelhouse.mkdir()

    for license_name in REQUIRED_LICENSES:
        _write_file(source, license_name)
    for relative in (
        "package.json",
        "scripts/desktop/DigiBox-TensorRT-Setup-Root.cmd",
        "src-tauri/Cargo.toml",
        "src-tauri/tauri.conf.json",
        "src-tauri/tauri.full.conf.json",
    ):
        _write_file(source, relative)

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
        "-OutputDirectory",
        str(output),
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


def test_full_source_rejects_target_machine_engine_manifest_before_staging(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    output = tmp_path / "output"
    engine_manifest = source.joinpath(*TARGET_MACHINE_ENGINE_MANIFEST.split("/"))
    engine_manifest.parent.mkdir(parents=True)
    engine_manifest.write_text('{"gpu":"target-machine"}', encoding="utf-8")

    result = _run_build(
        "-Edition",
        "Full",
        "-SourceRoot",
        str(source),
        "-RuntimeDestination",
        str(runtime),
        "-OutputDirectory",
        str(output),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "target-machine TensorRT engine manifest" in message
    assert engine_manifest.read_text(encoding="utf-8") == '{"gpu":"target-machine"}'
    assert not runtime.exists()
    assert not output.exists()


def test_full_tauri_validation_accepts_single_python_portable_v2() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '"portable-v2"' in source
    assert '"python\\python.exe"' in source
    assert '"packages\\main"' in source
    assert '"packages\\cosyvoice"' in source
    assert '"packages\\feynobg"' in source
    assert '"packages\\shared"' in source
    assert "AVTR1_COSYVOICE_PYTHONPATH" in source
    assert "AVTR1_FEYNOBG_PYTHONPATH" in source


def test_full_validation_uses_manifest_python_and_ordered_layers_for_all_probes() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "$Manifest.paths.python" in source
    assert "$Manifest.python.packageLayers" in source
    assert '"AVTR1_MAIN_PYTHONPATH"' in source
    assert '"AVTR1_COSYVOICE_PYTHONPATH"' in source
    assert '"AVTR1_FEYNOBG_PYTHONPATH"' in source
    assert "Assert-RuntimeInspection -RuntimeRoot $RuntimeRoot -Manifest $manifest" in source
    assert "Assert-RuntimeDependencyImports -RuntimeRoot $RuntimeRoot -Manifest $manifest" in source


def test_full_dependency_probes_match_runtime_pythonpath_bootstrap() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '[Environment]::SetEnvironmentVariable("PYTHONPATH"' in source
    assert '[Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", "1"' in source
    assert "assert 'sitecustomize' in sys.modules" in source
    assert "sys.path[:0]=sys.argv[1:]" not in source
    assert '"-I"' not in source
    assert "import avaturn_live_streamer.local_stream_cli" in source
    assert "import avaturn_live_streamer.memory.admin" in source
    assert "import avaturn_live_streamer.memory.api" in source


def test_full_validation_privacy_scan_trusts_v1_and_v2_managed_package_roots() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    for relative in (
        "python-main",
        "python-cosyvoice",
        "python-feynobg",
        "python",
        "packages",
    ):
        assert f'Join-Path $RuntimeRoot "{relative}"' in source


def test_full_runtime_scan_rejects_target_machine_engine_manifest(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    engine_manifest = runtime.joinpath(*TARGET_MACHINE_ENGINE_MANIFEST.split("/"))
    engine_manifest.parent.mkdir(parents=True)
    engine_manifest.write_text('{"gpu":"target-machine"}', encoding="utf-8")
    runner = tmp_path / "invoke-forbidden-runtime-scan.ps1"
    runner.write_text(
        r'''
param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot
)
$ErrorActionPreference = "Stop"
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
    throw "Assert-NoForbiddenRuntimePayload is missing"
}
Invoke-Expression $functionAst.Extent.Text
Assert-NoForbiddenRuntimePayload -RuntimeRoot $RuntimeRoot
'''.strip(),
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
            str(runner),
            "-BuildScript",
            str(BUILD_SCRIPT),
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
    assert "engine-manifest.json" in result.stdout + result.stderr
    assert engine_manifest.read_text(encoding="utf-8") == '{"gpu":"target-machine"}'


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
def test_full_runtime_scan_rejects_memory_persistence_payloads(
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
    runner = tmp_path / "invoke-memory-privacy-scan.ps1"
    runner.write_text(
        r'''
param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot
)
$ErrorActionPreference = "Stop"
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
    throw "Assert-NoForbiddenRuntimePayload is missing"
}
Invoke-Expression $functionAst.Extent.Text
Assert-NoForbiddenRuntimePayload -RuntimeRoot $RuntimeRoot
'''.strip(),
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
            str(runner),
            "-BuildScript",
            str(BUILD_SCRIPT),
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
    assert "forbidden" in (result.stdout + result.stderr).lower()


def test_incomplete_portable_v2_reports_the_single_python_and_all_four_layers(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "portable-v2"
    _write_runtime_manifest(runtime, "portable-v2")

    result = _run_build(
        "-Edition",
        "Full",
        "-SkipRuntimeBuild",
        "-RuntimeDestination",
        str(runtime),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    compact_message = "".join(message.split())
    assert "missing: scripts\\run_local_stream.py" in message
    for relative in REQUIRED_COSYVOICE_MODEL_FILES:
        assert relative.replace("/", "\\") in compact_message
    for relative in (
        "python\\python.exe",
        "packages\\main",
        "packages\\cosyvoice",
        "packages\\feynobg",
        "packages\\shared",
    ):
        assert relative in message
    assert "python-main\\python.exe" not in message


def test_incomplete_portable_v1_still_reports_three_legacy_interpreters(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "portable-v1"
    _write_runtime_manifest(runtime, "portable-v1")

    result = _run_build(
        "-Edition",
        "Full",
        "-SkipRuntimeBuild",
        "-RuntimeDestination",
        str(runtime),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "missing: scripts\\run_local_stream.py" in message
    for relative in (
        "python-main\\python.exe",
        "python-cosyvoice\\python.exe",
        "python-feynobg\\python.exe",
    ):
        assert relative in message
    assert "python\\python.exe" not in message


def test_plan_only_has_no_runtime_staging_side_effect(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime-does-not-exist"
    delivery = tmp_path / "delivery-does-not-exist"

    plan = _plan(
        "-Edition",
        "Full",
        "-RuntimeDestination",
        str(runtime),
        "-OutputDirectory",
        str(delivery),
    )

    assert Path(str(plan["runtimeRoot"])) == runtime.resolve()
    assert Path(str(plan["outputDirectory"])) == delivery.resolve()
    assert not runtime.exists()
    assert not delivery.exists()


def test_standard_plan_only_does_not_generate_the_offline_nsis_overlay() -> None:
    before = (
        STANDARD_OFFLINE_CONFIG.stat().st_mtime_ns
        if STANDARD_OFFLINE_CONFIG.exists()
        else None
    )

    _plan("-Edition", "Standard")

    after = (
        STANDARD_OFFLINE_CONFIG.stat().st_mtime_ns
        if STANDARD_OFFLINE_CONFIG.exists()
        else None
    )
    assert after == before


@pytest.mark.parametrize("target", ["Installer", "Msi"])
def test_full_refuses_single_file_installer_targets(target: str) -> None:
    result = _run_build("-Edition", "Full", "-Target", target, "-PlanOnly")

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "Full" in message
    assert "Archive or Unpacked" in message


def test_full_config_leaves_resources_to_the_unpacked_delivery_stage() -> None:
    config = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))
    bundle = config["bundle"]
    resources = bundle["resources"]
    builder = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert bundle["active"] is False
    assert "targets" not in bundle or not {
        str(value).lower() for value in bundle["targets"]
    }.intersection({"nsis", "msi"})
    assert resources == []
    assert (
        'Copy-DirectoryWithRobocopy -Source $runtimeRoot '
        '-Destination (Join-Path $fullUnpackedRoot "avtr-runtime")'
    ) in builder
    assert (
        'Copy-Item -LiteralPath (Join-Path $root $relativeLicense) '
        '-Destination (Join-Path $licenseRoot $relativeLicense)'
    ) in builder


def test_full_plan_with_custom_spaced_runtime_is_read_only_and_resource_free(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "custom runtime" / "avtr-runtime"
    delivery = tmp_path / "custom delivery"

    plan = _plan(
        "-Edition",
        "Full",
        "-RuntimeDestination",
        str(runtime),
        "-OutputDirectory",
        str(delivery),
    )
    config = json.loads(FULL_CONFIG.read_text(encoding="utf-8"))

    assert Path(str(plan["runtimeRoot"])) == runtime.resolve()
    assert plan["config"] == "src-tauri/tauri.full.conf.json"
    assert "--no-bundle" in plan["tauriArguments"]
    assert config["bundle"]["resources"] == []
    assert not runtime.exists()
    assert not delivery.exists()


def test_full_plan_declares_all_license_and_runtime_validation_gates() -> None:
    plan = _plan("-Edition", "Full")
    validation = plan["runtimeValidation"]

    assert set(plan["requiredLicenses"]) == REQUIRED_LICENSES
    assert validation["manifest"] == "runtime-manifest.json"
    assert validation["layout"] == "portable-v2"
    assert validation["defaultLayout"] == "portable-v2"
    assert validation["layouts"] == ["portable-v1", "portable-v2"]
    assert validation["inspector"] == "scripts/desktop/inspect_runtime.py"
    assert tuple(validation["requiredMemoryModules"]) == MEMORY_PACKAGE_MODULES
    assert set(validation["versionedHelpers"]) == TENSORRT_HELPERS
    assert set(validation["requiredComponents"]) == {
        "dependenciesIncluded",
        "modelsIncluded",
        "frontendVendorIncluded",
        "tensorRtBuildInputsIncluded",
    }


def test_full_plan_excludes_private_state_caches_and_machine_engines() -> None:
    plan = _plan("-Edition", "Full")
    forbidden = set(plan["runtimeValidation"]["forbiddenPayloads"])

    assert {
        "user_assets",
        "local_voices",
        "voice_clones",
        "reference_audio",
        "*.engine",
        "*.plan",
        "grid_sample_3d_plugin*.dll",
        "spk2info.pt",
        ".env*",
        "*.key",
        "*.pem",
        ".cache",
        "__pycache__",
        ".engine-staging",
        ".engine-backups",
        TARGET_MACHINE_ENGINE_MANIFEST,
        "memory.sqlite3",
        "memory.sqlite3-wal",
        "memory.sqlite3-shm",
        "memory/backups",
        "memory/pending-imports",
        "digibox-memory*.json",
    }.issubset(forbidden)


def test_tauri_memory_package_has_a_dedicated_runtime_gate() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "function Assert-MemoryPackageEntrypoint" in source
    for module_path in MEMORY_PACKAGE_MODULES:
        assert module_path.replace("/", "\\") in source
    assert "Assert-MemoryPackageEntrypoint -RuntimeRoot $RuntimeRoot" in source


@pytest.mark.parametrize("missing_module", MEMORY_PACKAGE_MODULES)
def test_tauri_memory_gate_rejects_each_missing_module(
    tmp_path: Path,
    missing_module: str,
) -> None:
    runtime = tmp_path / "runtime"
    for module_path in MEMORY_PACKAGE_MODULES:
        if module_path != missing_module:
            target = runtime.joinpath(*module_path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture", encoding="utf-8")

    result = _invoke_build_function(
        tmp_path,
        "Assert-MemoryPackageEntrypoint",
        f'Assert-MemoryPackageEntrypoint -RuntimeRoot "{runtime}"',
    )

    assert result.returncode != 0
    assert Path(missing_module).name in result.stdout + result.stderr


def test_script_uses_existing_portable_builder_and_hashes_tensorrt_helpers() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "scripts\\desktop\\build_portable_runtime.ps1" in source
    assert "runtime-manifest.json" in source
    assert "Assert-NoForbiddenRuntimePayload" in source
    assert "Assert-CompleteRuntimeComponents" in source
    assert "Get-FileHash" in source
    for helper in TENSORRT_HELPERS:
        assert helper.replace("/", "\\") in source


def test_standard_builder_validates_and_embeds_webview2_without_fwlink_head() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Get-AuthenticodeSignature" in source
    assert "Get-FileHash" in source
    assert WEBVIEW2_INSTALLER_NAME in source
    assert str(WEBVIEW2_INSTALLER_SIZE) in source
    assert WEBVIEW2_INSTALLER_SHA256 in source
    assert "CN=Microsoft Corporation" in source
    assert "NSIS_HOOK_PREINSTALL" in source
    assert 'type = "skip"' in source
    assert "Assert-OfflineWebView2Embedded" in source
    assert "go.microsoft.com/fwlink/?linkid=2124701" not in source.lower()


def test_standard_builder_verifies_the_same_nsis_escaped_webview2_path_it_writes() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "function ConvertTo-NsisLiteralPath" in source
    assert (
        "$escapedInstallerPath = ConvertTo-NsisLiteralPath -Path $InstallerPath"
        in source
    )
    assert (
        "$escapedVerifiedPayloadPath = "
        "ConvertTo-NsisLiteralPath -Path $VerifiedPayloadPath"
        in source
    )
    assert "$hook.Contains($escapedVerifiedPayloadPath)" in source


def test_standard_builder_rejects_an_untrusted_explicit_webview2_executable(
    tmp_path: Path,
) -> None:
    fake_installer = tmp_path / WEBVIEW2_INSTALLER_NAME
    fake_installer.write_bytes(b"not a Microsoft-signed WebView2 installer")

    result = _run_build(
        "-Edition",
        "Standard",
        "-Target",
        "Installer",
        "-WebView2OfflineInstaller",
        str(fake_installer),
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "WebView2" in message
    assert any(token in message for token in ("size", "SHA256", "Authenticode"))


def test_tauri_build_keeps_the_electron_builder_as_a_fallback() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

    assert (ROOT / "scripts" / "build_desktop_windows.ps1").is_file()
    assert (ROOT / "electron-builder.yml").is_file()
    assert (ROOT / "electron-builder-full.yml").is_file()
    assert "desktop:dev" in package["scripts"]
    assert "desktop:dist" in package["scripts"]
