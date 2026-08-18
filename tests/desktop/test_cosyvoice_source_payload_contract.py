from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_CASES = (
    pytest.param(
        ROOT / "scripts" / "build_tauri_windows.ps1",
        True,
        id="tauri",
    ),
    pytest.param(
        ROOT / "scripts" / "build_desktop_windows.ps1",
        False,
        id="electron",
    ),
)
COSYVOICE_PACKAGE_MARKER = "third_party/CosyVoice/cosyvoice/__init__.py"
COSYVOICE_RUNTIME_ENTRY = "third_party/CosyVoice/cosyvoice/cli/cosyvoice.py"

COMMON_REQUIRED_FILES = (
    "scripts/run_local_stream.py",
    "src/avaturn_live_streamer/__init__.py",
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
    "src/avtr1_renderer/__init__.py",
    "src/avaturn_live_streamer/vendor/preact.module.js",
    COSYVOICE_RUNTIME_ENTRY,
    "models/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml",
    "models/Fun-CosyVoice3-0.5B-2512/llm.pt",
    "models/Fun-CosyVoice3-0.5B-2512/flow.pt",
    "models/Fun-CosyVoice3-0.5B-2512/hift.pt",
    "models/Fun-CosyVoice3-0.5B-2512/campplus.onnx",
    "models/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/generation_config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/model.safetensors",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/tokenizer_config.json",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/merges.txt",
    "models/Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/vocab.json",
    "artifacts/main/avtr1_normalizer.safetensors",
    "artifacts/main/build_artifacts/avtr1.scripted.pt",
    "artifacts/main/build_artifacts/hubert-lbs-avtr1.onnx",
    "artifacts/main/build_artifacts/decoder.onnx",
    "artifacts/main/build_artifacts/modnet.onnx",
    "artifacts/main/build_artifacts/stitch_network.onnx",
    "artifacts/main/build_artifacts/warp_network.onnx",
)
ELECTRON_REQUIRED_FILES = (
    "src/avaturn_live_streamer/vendor/preact-hooks.module.js",
    "src/avaturn_live_streamer/vendor/htm.module.js",
    "third_party/CosyVoice/third_party/Matcha-TTS/matcha/__init__.py",
    "artifacts/main/avatars_artifacts/pasteback_mask.png",
    "artifacts/main/renderer_runtime_artifacts/appearance_extractor.onnx",
    "artifacts/main/renderer_runtime_artifacts/motion_extractor.onnx",
    "artifacts/main/renderer_runtime_artifacts/insightface_det.onnx",
    "artifacts/main/renderer_runtime_artifacts/landmark106.onnx",
    "artifacts/main/renderer_runtime_artifacts/landmark203.onnx",
    "artifacts/main/renderer_runtime_artifacts/blaze_face.onnx",
    "artifacts/main/renderer_runtime_artifacts/face_mesh.onnx",
    "artifacts/main/build_artifacts/warp_network_ori.onnx",
)


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required for the payload contract"
    return executable


def _write_file(root: Path, relative: str, content: str = "payload") -> None:
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _run_required_payload_check(
    tmp_path: Path,
    build_script: Path,
    is_tauri: bool,
    entry_state: str,
) -> subprocess.CompletedProcess[str]:
    runtime = tmp_path / f"runtime-{build_script.stem}-{entry_state}"
    required_files = COMMON_REQUIRED_FILES + (
        () if is_tauri else ELECTRON_REQUIRED_FILES
    )
    for relative in required_files:
        if relative == COSYVOICE_RUNTIME_ENTRY and entry_state == "missing":
            continue
        content = (
            ""
            if relative == COSYVOICE_RUNTIME_ENTRY and entry_state == "empty"
            else "payload"
        )
        _write_file(runtime, relative, content)

    # Both upstream package markers are intentionally empty in the checked-in
    # sources. A Full Runtime must validate executable source, not marker bytes.
    _write_file(runtime, COSYVOICE_PACKAGE_MARKER, "")
    if is_tauri:
        _write_file(runtime, "runtime-manifest.json", "{}")
        for relative in (
            "python-main/python.exe",
            "python-cosyvoice/python.exe",
            "python-feynobg/python.exe",
        ):
            _write_file(runtime, relative)

    harness = tmp_path / f"invoke-{build_script.stem}-{entry_state}.ps1"
    harness.write_text(
        r'''
param(
    [Parameter(Mandatory = $true)][string]$BuildScript,
    [Parameter(Mandatory = $true)][string]$RuntimeRoot,
    [Parameter(Mandatory = $true)]
    [ValidateSet("tauri", "electron")]
    [string]$BuildKind
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
            $node.Name -eq "Assert-RequiredRuntimePayload"
    },
    $true
)
if ($null -eq $functionAst) {
    throw "Assert-RequiredRuntimePayload was not found"
}
Invoke-Expression $functionAst.Extent.Text
if ($BuildKind -eq "tauri") {
    $versionedHelpers = @()
    $manifest = [pscustomobject]@{ layout = "portable-v1" }
    Assert-RequiredRuntimePayload -RuntimeRoot $RuntimeRoot -Manifest $manifest
} else {
    Assert-RequiredRuntimePayload -RuntimeRoot $RuntimeRoot
}
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
            str(harness),
            "-BuildScript",
            str(build_script),
            "-RuntimeRoot",
            str(runtime),
            "-BuildKind",
            "tauri" if is_tauri else "electron",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize("build_script,is_tauri", BUILD_CASES)
def test_full_runtime_validates_the_imported_cosyvoice_source(
    build_script: Path,
    is_tauri: bool,
) -> None:
    source = build_script.read_text(encoding="utf-8")

    assert COSYVOICE_RUNTIME_ENTRY.replace("/", "\\") in source
    assert COSYVOICE_PACKAGE_MARKER.replace("/", "\\") not in source


@pytest.mark.parametrize("build_script,is_tauri", BUILD_CASES)
def test_empty_cosyvoice_package_marker_is_valid_when_runtime_entry_is_nonempty(
    tmp_path: Path,
    build_script: Path,
    is_tauri: bool,
) -> None:
    result = _run_required_payload_check(
        tmp_path,
        build_script,
        is_tauri,
        entry_state="nonempty",
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("build_script,is_tauri", BUILD_CASES)
@pytest.mark.parametrize("entry_state", ("missing", "empty"))
def test_missing_or_empty_cosyvoice_runtime_entry_is_rejected(
    tmp_path: Path,
    build_script: Path,
    is_tauri: bool,
    entry_state: str,
) -> None:
    result = _run_required_payload_check(
        tmp_path,
        build_script,
        is_tauri,
        entry_state=entry_state,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "payload is incomplete" in output.lower()
    assert COSYVOICE_RUNTIME_ENTRY.replace("/", "\\").lower() in output.lower()
