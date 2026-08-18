from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build_warp_plugin_windows.ps1"
PATCH_FILE = ROOT / "scripts" / "patches" / "grid-sample3d-trt-plugin-windows.patch"
FP16_PATCH_FILE = (
    ROOT
    / "scripts"
    / "patches"
    / "0001-fix-FP16-crash-on-CUDA-12.x-kernel-optimizations.patch"
)
TENSORRT_BUILD_SCRIPT = ROOT / "scripts" / "build_tensorrt_windows.ps1"


def test_windows_warp_plugin_build_is_pinned_and_applies_runtime_shape_fix() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")
    patch = PATCH_FILE.read_text(encoding="utf-8")

    assert "f964750b8ce8d5453251a4036572d471a4c395e1" in script
    assert "9255eb39e6642787828a4c1f7fc1d09fe004e7a2" in script
    assert "Get-PluginPatchState" in script
    assert '"apply", "--whitespace=nowarn", $normalizedPatch' in script
    assert "inputDesc[0].dims" in patch
    assert "inputDesc[1].dims" in patch
    assert "mBatch, mInputChannel" in patch
    assert "getCreators" in patch
    assert "setLoggerFinder" in patch


def test_windows_warp_plugin_propagates_the_first_cuda_launch_error() -> None:
    patch = PATCH_FILE.read_text(encoding="utf-8")

    assert "return err != cudaSuccess;" in patch


def test_windows_warp_plugin_patches_do_not_add_trailing_whitespace() -> None:
    for patch_file in (FP16_PATCH_FILE, PATCH_FILE):
        added_lines = (
            line
            for line in patch_file.read_text(encoding="utf-8").splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        assert all(line.rstrip(" \t") == line for line in added_lines), patch_file


def test_windows_warp_plugin_normalizes_crlf_patches_before_git_apply() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "function New-NormalizedTextPatch" in script
    assert '[System.Text.UTF8Encoding]::new($false, $true)' in script
    assert '.Replace("`r`n", "`n")' in script
    assert 'if ($patchText.Contains("`r")) {' in script
    assert "function Get-PluginPatchState" in script
    assert "function Apply-PluginPatch" in script
    assert "Apply-PluginPatch -PluginSource $pluginSource -PatchFile $patchFile" in script


def test_windows_warp_plugin_patch_files_are_forced_to_lf_in_checkouts() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "scripts/patches/*.patch text eol=lf" in attributes


def test_windows_warp_plugin_builder_supports_windows_powershell_51() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "[System.Environment]::OSVersion.Platform" in script
    assert "[System.PlatformID]::Win32NT" in script
    assert "$IsWindows" not in script


def test_windows_warp_plugin_uses_a_short_unique_native_build_root() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "[System.IO.Path]::GetTempPath()" in script
    assert '[Guid]::NewGuid().ToString("N")' in script
    assert "$maxNativeBuildRootLength = 120" in script
    assert "$pluginBuild = [System.IO.Path]::GetFullPath(" in script
    assert (
        "$pluginBuild = Join-Path $BuildRoot "
        '"grid-sample3d-trt-plugin-build-sm$CudaArchitecture"' not in script
    )
    assert "LongPathsEnabled" not in script


def test_windows_warp_plugin_keeps_inputs_under_requested_build_root() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '$pluginSource = Join-Path $BuildRoot "grid-sample3d-trt-plugin"' in script
    assert '$tensorRTSource = Join-Path $BuildRoot "TensorRT-10.11-src"' in script
    assert '$importDir = Join-Path $BuildRoot "tensorrt-import-lib"' in script


def test_windows_warp_plugin_returns_a_stable_dll_under_build_root() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '$stableOutputDir = Join-Path $BuildRoot "native-output-sm$CudaArchitecture"' in script
    assert '$builtPluginDll = Join-Path $pluginBuild "grid_sample_3d_plugin.dll"' in script
    assert '$pluginDll = Join-Path $stableOutputDir "grid_sample_3d_plugin.dll"' in script
    assert '$incomingPluginDll = "$pluginDll.incoming.$([Guid]::NewGuid().ToString("N"))"' in script
    assert "Copy-Item -LiteralPath $builtPluginDll -Destination $incomingPluginDll" in script
    assert "Copy-Item -LiteralPath $builtPluginDll -Destination $pluginDll -Force" not in script
    assert "Get-FileHash -LiteralPath $builtPluginDll -Algorithm SHA256" in script
    assert "Get-FileHash -LiteralPath $incomingPluginDll -Algorithm SHA256" in script
    assert "$builtPlugin.Length -ne $incomingPlugin.Length" in script
    assert "$builtPluginHash.Hash -ne $incomingPluginHash.Hash" in script
    assert "Move-Item -LiteralPath $incomingPluginDll -Destination $pluginDll -Force" in script
    assert "finally {" in script
    assert "Remove-Item -LiteralPath $incomingPluginDll -Force" in script
    assert "$env:AVTR1_WARP_PLUGIN = $pluginDll" in script
    assert "Write-Output $pluginDll" in script


def test_windows_warp_plugin_accepts_an_explicit_output_path() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert '[string]$OutputPath = ""' in script
    assert "[string]::IsNullOrWhiteSpace($OutputPath)" in script
    assert "$pluginDll = [System.IO.Path]::GetFullPath($OutputPath)" in script
    assert "$stableOutputDir = Split-Path -Parent $pluginDll" in script


def test_windows_warp_plugin_cleans_native_temp_only_after_success() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "$nativeBuildSucceeded = $false" in script
    assert "$nativeBuildSucceeded = $true" in script
    assert "if ($nativeBuildSucceeded) {" in script
    assert "Remove-SafeNativeBuildDirectory -Path $pluginBuild" in script
    assert "Write-Warning" in script

    publish = script.index(
        "Move-Item -LiteralPath $incomingPluginDll -Destination $pluginDll -Force"
    )
    mark_success = script.index("$nativeBuildSucceeded = $true")
    cleanup = script.index("Remove-SafeNativeBuildDirectory -Path $pluginBuild")
    assert publish < mark_success < cleanup


def test_windows_warp_plugin_validates_native_temp_before_recursive_cleanup() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "function Remove-SafeNativeBuildDirectory" in script
    assert "$nativeRoot = [System.IO.Path]::GetFullPath($Path)" in script
    assert (
        "$temporaryBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()" in script
    )
    assert (
        "$nativeRoot.Equals($temporaryBase, [System.StringComparison]::OrdinalIgnoreCase" in script
    )
    assert (
        "$nativeRoot.StartsWith("
        "$temporaryPrefix, [System.StringComparison]::OrdinalIgnoreCase" in script
    )
    assert '$nativeLeaf.StartsWith("digibox-warp-sm", ' in script
    assert "Remove-Item -LiteralPath $nativeRoot -Recurse -Force" in script


def test_windows_warp_plugin_failure_names_the_short_native_build_path() -> None:
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert (
        'throw "GridSample3D Windows build failed in short native build path: '
        '$pluginBuild"' in script
    )
    assert (
        'throw "Build completed without producing $builtPluginDll in short '
        'native build path: $pluginBuild"' in script
    )
    assert (
        'throw "Plugin DLL is missing the TensorRT shared-library exports in '
        'short native build path: $pluginBuild"' in script
    )


def test_full_tensorrt_build_can_build_the_windows_warp_plugin() -> None:
    plugin_script = BUILD_SCRIPT.read_text(encoding="utf-8")
    tensorrt_script = TENSORRT_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "$env:AVTR1_WARP_PLUGIN = $pluginDll" in plugin_script
    assert "build_warp_plugin_windows.ps1" in tensorrt_script
    assert "smoke_warp_plugin_windows.py" in tensorrt_script
    assert "warp-b1-validation" in tensorrt_script


def test_full_tensorrt_build_versions_batch1_validation_by_plugin_hash() -> None:
    script = TENSORRT_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Get-FileHash -LiteralPath $env:AVTR1_WARP_PLUGIN -Algorithm SHA256" in script
    assert "warp-b1-validation-$pluginHash" in script


def test_full_tensorrt_build_backs_up_a_conflicting_formal_plugin() -> None:
    script = TENSORRT_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "$installedPluginHash -ne $pluginHash" in script
    assert 'ToString("yyyyMMddTHHmmssfffZ")' in script
    assert "while (Test-Path -LiteralPath $backupPath)" in script
    assert "Move-Item -LiteralPath $installedWarpPlugin -Destination $backupPath" in script
    assert "Backed up existing warp plugin: $backupPath" in script
    assert "Remove-Item" not in script
