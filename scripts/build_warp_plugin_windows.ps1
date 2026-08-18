param(
    [string]$Python = "",
    [string]$BuildRoot = "",
    [string]$OutputPath = "",
    [int]$CudaArchitecture = 0
)

$ErrorActionPreference = "Stop"

$pluginCommit = "f964750b8ce8d5453251a4036572d471a4c395e1"
$tensorRTCommit = "9255eb39e6642787828a4c1f7fc1d09fe004e7a2"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$fp16PatchFile = Join-Path $PSScriptRoot "patches\0001-fix-FP16-crash-on-CUDA-12.x-kernel-optimizations.patch"
$windowsPatchFile = Join-Path $PSScriptRoot "patches\grid-sample3d-trt-plugin-windows.patch"
$nvinferDef = Join-Path $PSScriptRoot "patches\nvinfer_10.def"

if (
    [System.Environment]::OSVersion.Platform -ne
    [System.PlatformID]::Win32NT
) {
    throw "This script builds the native Windows GridSample3D TensorRT plugin."
}

if (-not $Python) {
    throw (
        "Pass -Python with the Full portable-v2 interpreter, or run " +
        "scripts\build_tensorrt_windows.ps1 -IncludeWarp."
    )
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

if (-not $BuildRoot) {
    $BuildRoot = Join-Path $projectRoot "artifacts\build\windows-warp-plugin-fp16"
}
# Callers that may run concurrently must provide a unique BuildRoot.
$BuildRoot = [System.IO.Path]::GetFullPath($BuildRoot)
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

function Invoke-Git {
    param([string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed: git $($GitArguments -join ' ')"
    }
}

function New-NormalizedTextPatch {
    param(
        [Parameter(Mandatory = $true)][string]$PatchFile,
        [Parameter(Mandatory = $true)][string]$PatchDirectory
    )

    if (-not (Test-Path -LiteralPath $PatchFile -PathType Leaf)) {
        throw "Plugin patch file was not found: $PatchFile"
    }
    $patchBytes = [System.IO.File]::ReadAllBytes($PatchFile)
    if ($patchBytes -contains 0) {
        throw "Plugin patch must be UTF-8 text without NUL bytes: $PatchFile"
    }
    if (
        ($patchBytes.Length -ge 2 -and $patchBytes[0] -eq 0xFF -and $patchBytes[1] -eq 0xFE) -or
        ($patchBytes.Length -ge 2 -and $patchBytes[0] -eq 0xFE -and $patchBytes[1] -eq 0xFF) -or
        ($patchBytes.Length -ge 3 -and $patchBytes[0] -eq 0xEF -and $patchBytes[1] -eq 0xBB -and $patchBytes[2] -eq 0xBF)
    ) {
        throw "Plugin patch must be UTF-8 without a byte-order mark: $PatchFile"
    }

    try {
        $patchText = ([System.Text.UTF8Encoding]::new($false, $true)).GetString($patchBytes)
    }
    catch {
        throw "Plugin patch is not valid UTF-8 text: $PatchFile"
    }
    $patchText = $patchText.Replace("`r`n", "`n")
    if ($patchText.Contains("`r")) {
        throw "Plugin patch contains an unsupported standalone carriage return: $PatchFile"
    }

    $normalizedPatch = Join-Path $PatchDirectory (
        ".digibox-normalized-patch-$([Guid]::NewGuid().ToString('N')).patch"
    )
    [System.IO.File]::WriteAllText(
        $normalizedPatch,
        $patchText,
        [System.Text.UTF8Encoding]::new($false)
    )
    return $normalizedPatch
}

function Get-PluginPatchState {
    param(
        [Parameter(Mandatory = $true)][string]$PluginSource,
        [Parameter(Mandatory = $true)][string]$PatchFile
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & git -C $PluginSource apply --check --whitespace=nowarn $PatchFile 2>$null
        $patchCanApply = $LASTEXITCODE -eq 0
        & git -C $PluginSource apply --reverse --check --whitespace=nowarn $PatchFile 2>$null
        $patchAlreadyApplied = $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($patchCanApply -and -not $patchAlreadyApplied) {
        return "CanApply"
    }
    if (-not $patchCanApply -and $patchAlreadyApplied) {
        return "AlreadyApplied"
    }
    if ($patchCanApply -and $patchAlreadyApplied) {
        return "Ambiguous"
    }
    return "PartialOrDiverged"
}

function Apply-PluginPatch {
    param(
        [Parameter(Mandatory = $true)][string]$PluginSource,
        [Parameter(Mandatory = $true)][string]$PatchFile,
        [Parameter(Mandatory = $true)][string]$PatchDirectory
    )

    $normalizedPatch = New-NormalizedTextPatch `
        -PatchFile $PatchFile `
        -PatchDirectory $PatchDirectory
    try {
        $patchState = Get-PluginPatchState `
            -PluginSource $PluginSource `
            -PatchFile $normalizedPatch
        switch ($patchState) {
            "CanApply" {
                Invoke-Git @("-C", $PluginSource, "apply", "--whitespace=nowarn", $normalizedPatch)
            }
            "AlreadyApplied" {
                break
            }
            default {
                throw "Plugin patch state is $patchState and cannot be applied safely: $PatchFile"
            }
        }

        $verifiedState = Get-PluginPatchState `
            -PluginSource $PluginSource `
            -PatchFile $normalizedPatch
        if ($verifiedState -ne "AlreadyApplied") {
            throw "Plugin patch did not reach a verified applied state: $PatchFile"
        }
    }
    finally {
        if (Test-Path -LiteralPath $normalizedPatch) {
            Remove-Item -LiteralPath $normalizedPatch -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-SafeNativeBuildDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $nativeRoot = [System.IO.Path]::GetFullPath($Path)
    $nativeRoot = $nativeRoot.TrimEnd("\", "/")
    $temporaryBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    $temporaryBase = $temporaryBase.TrimEnd("\", "/")
    $temporaryPrefix = $temporaryBase + [System.IO.Path]::DirectorySeparatorChar
    $nativeLeaf = Split-Path -Leaf $nativeRoot

    if ($nativeRoot.Equals($temporaryBase, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove the temporary base directory: $nativeRoot"
    }
    if (-not $nativeRoot.StartsWith($temporaryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a native build path outside the temporary directory: $nativeRoot"
    }
    if (-not $nativeLeaf.StartsWith("digibox-warp-sm", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an unexpected native build directory: $nativeRoot"
    }
    if (Test-Path -LiteralPath $nativeRoot) {
        Remove-Item -LiteralPath $nativeRoot -Recurse -Force
    }
}

$trtInfoJson = & $Python -c 'import json, pathlib, tensorrt, tensorrt_libs; print(json.dumps(dict(version=tensorrt.__version__, libs=str(pathlib.Path(tensorrt_libs.__file__).parent))))'
if ($LASTEXITCODE -ne 0) {
    throw "TensorRT 10.11 is not importable from $Python"
}
$trtInfo = $trtInfoJson | ConvertFrom-Json
if ($trtInfo.version -ne "10.11.0.33") {
    throw "Expected TensorRT 10.11.0.33, found $($trtInfo.version)"
}
$nvinferDll = Join-Path $trtInfo.libs "nvinfer_10.dll"
if (-not (Test-Path -LiteralPath $nvinferDll)) {
    throw "TensorRT runtime DLL not found: $nvinferDll"
}

if ($CudaArchitecture -le 0) {
    $CudaArchitecture = [int](& $Python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(major * 10 + minor)')
    if ($LASTEXITCODE -ne 0 -or $CudaArchitecture -le 0) {
        throw "Could not detect the CUDA architecture. Pass -CudaArchitecture explicitly."
    }
}

$pluginSource = Join-Path $BuildRoot "grid-sample3d-trt-plugin"
if (-not (Test-Path -LiteralPath (Join-Path $pluginSource ".git"))) {
    Invoke-Git @("clone", "https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git", $pluginSource)
}
$pluginHead = (& git -C $pluginSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect $pluginSource"
}
if ($pluginHead -ne $pluginCommit) {
    Invoke-Git @("-C", $pluginSource, "checkout", "--detach", $pluginCommit)
}

foreach ($patchFile in @($fp16PatchFile, $windowsPatchFile)) {
    Apply-PluginPatch -PluginSource $pluginSource -PatchFile $patchFile -PatchDirectory $BuildRoot
}

$tensorRTSource = Join-Path $BuildRoot "TensorRT-10.11-src"
if (-not (Test-Path -LiteralPath (Join-Path $tensorRTSource ".git"))) {
    Invoke-Git @(
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        "https://github.com/NVIDIA/TensorRT.git",
        $tensorRTSource
    )
    Invoke-Git @("-C", $tensorRTSource, "sparse-checkout", "init", "--cone")
    Invoke-Git @("-C", $tensorRTSource, "sparse-checkout", "set", "include")
    Invoke-Git @("-C", $tensorRTSource, "checkout", "--detach", $tensorRTCommit)
}
$tensorRTHead = (& git -C $tensorRTSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $tensorRTHead -ne $tensorRTCommit) {
    throw "TensorRT headers are not pinned to 10.11.0.33: $tensorRTSource"
}
$tensorRTInclude = Join-Path $tensorRTSource "include"

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Build Tools were not found."
}
$vsInstall = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $vsInstall) {
    throw "Visual Studio C++ Build Tools were not found."
}
$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
$msvcRoot = Get-ChildItem -LiteralPath (Join-Path $vsInstall "VC\Tools\MSVC") -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
$libExe = Join-Path $msvcRoot.FullName "bin\Hostx64\x64\lib.exe"
$dumpbinExe = Join-Path $msvcRoot.FullName "bin\Hostx64\x64\dumpbin.exe"

$exports = & $dumpbinExe /exports $nvinferDll
$exportText = $exports -join "`n"
if ($LASTEXITCODE -ne 0 -or $exportText -notmatch "\bgetPluginRegistry\b") {
    throw "$nvinferDll does not export getPluginRegistry."
}

$importDir = Join-Path $BuildRoot "tensorrt-import-lib"
New-Item -ItemType Directory -Force -Path $importDir | Out-Null
$nvinferLib = Join-Path $importDir "nvinfer_10.lib"
& $libExe "/def:$nvinferDef" "/machine:x64" "/out:$nvinferLib"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $nvinferLib)) {
    throw "Failed to generate the TensorRT import library."
}

$cmake = (Get-Command cmake -ErrorAction Stop).Source
$nvcc = (Get-Command nvcc -ErrorAction Stop).Source
$ninja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if (-not (Test-Path -LiteralPath $ninja)) {
    throw "Ninja was not found in Visual Studio Build Tools."
}

$nativeBuildName = (
    "digibox-warp-sm{0}-{1}-{2}" -f
    $CudaArchitecture,
    $PID,
    [Guid]::NewGuid().ToString("N")
)
$pluginBuild = [System.IO.Path]::GetFullPath(
    (Join-Path ([System.IO.Path]::GetTempPath()) $nativeBuildName)
)
$maxNativeBuildRootLength = 120
if ($pluginBuild.Length -gt $maxNativeBuildRootLength) {
    throw (
        "The short native build path is too long " +
        "($($pluginBuild.Length) characters; maximum " +
        "$maxNativeBuildRootLength): $pluginBuild"
    )
}
New-Item -ItemType Directory -Force -Path $pluginBuild | Out-Null

$configure = (
    '"{0}" -S "{1}" -B "{2}" -G Ninja ' +
    '-DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="{3}" ' +
    '-DCMAKE_CUDA_COMPILER="{4}" -DCMAKE_CUDA_ARCHITECTURES={5} ' +
    '-DTensorRT_INCLUDE_DIR="{6}" -DTensorRT_LIBRARY="{7}"'
) -f $cmake, $pluginSource, $pluginBuild, $ninja, $nvcc, $CudaArchitecture, $tensorRTInclude, $nvinferLib
$build = '"{0}" --build "{1}" --target grid_sample_3d_plugin --config Release --parallel' -f $cmake, $pluginBuild
$developerCommand = '"{0}" -arch=x64 -host_arch=x64 >nul && {1} && {2}' -f $vsDevCmd, $configure, $build

$nativeBuildSucceeded = $false
try {
    & $env:ComSpec /d /s /c $developerCommand
    if ($LASTEXITCODE -ne 0) {
        throw "GridSample3D Windows build failed in short native build path: $pluginBuild"
    }

    $builtPluginDll = Join-Path $pluginBuild "grid_sample_3d_plugin.dll"
    if (-not (Test-Path -LiteralPath $builtPluginDll)) {
        throw "Build completed without producing $builtPluginDll in short native build path: $pluginBuild"
    }

    $pluginExports = & $dumpbinExe /exports $builtPluginDll
    $pluginExportText = $pluginExports -join "`n"
    if (
        $LASTEXITCODE -ne 0 -or
        $pluginExportText -notmatch "\bgetCreators\b" -or
        $pluginExportText -notmatch "\bsetLoggerFinder\b"
    ) {
        throw "Plugin DLL is missing the TensorRT shared-library exports in short native build path: $pluginBuild"
    }

    if ([string]::IsNullOrWhiteSpace($OutputPath)) {
        $stableOutputDir = Join-Path $BuildRoot "native-output-sm$CudaArchitecture"
        $pluginDll = Join-Path $stableOutputDir "grid_sample_3d_plugin.dll"
    }
    else {
        $pluginDll = [System.IO.Path]::GetFullPath($OutputPath)
        $stableOutputDir = Split-Path -Parent $pluginDll
    }
    New-Item -ItemType Directory -Force -Path $stableOutputDir | Out-Null
    $incomingPluginDll = "$pluginDll.incoming.$([Guid]::NewGuid().ToString("N"))"
    try {
        Copy-Item -LiteralPath $builtPluginDll -Destination $incomingPluginDll
        $builtPlugin = Get-Item -LiteralPath $builtPluginDll
        $incomingPlugin = Get-Item -LiteralPath $incomingPluginDll
        if ($builtPlugin.Length -ne $incomingPlugin.Length) {
            throw "Incoming plugin length does not match the built DLL: $incomingPluginDll"
        }
        $builtPluginHash = Get-FileHash -LiteralPath $builtPluginDll -Algorithm SHA256
        $incomingPluginHash = Get-FileHash -LiteralPath $incomingPluginDll -Algorithm SHA256
        if ($builtPluginHash.Hash -ne $incomingPluginHash.Hash) {
            throw "Incoming plugin SHA256 does not match the built DLL: $incomingPluginDll"
        }
        Move-Item -LiteralPath $incomingPluginDll -Destination $pluginDll -Force
    }
    finally {
        if (Test-Path -LiteralPath $incomingPluginDll) {
            try {
                Remove-Item -LiteralPath $incomingPluginDll -Force
            }
            catch {
                Write-Warning "Could not remove incoming plugin file: $incomingPluginDll"
            }
        }
    }
    $nativeBuildSucceeded = $true
}
finally {
    if ($nativeBuildSucceeded) {
        try {
            Remove-SafeNativeBuildDirectory -Path $pluginBuild
        }
        catch {
            Write-Warning (
                "The plugin was published, but DigiBox could not clean " +
                "the short native build path $pluginBuild. " +
                $_.Exception.Message
            )
        }
    }
}

Write-Host "Windows GridSample3D plugin built successfully:"
Write-Host "  $pluginDll"
$env:AVTR1_WARP_PLUGIN = $pluginDll
Write-Host "AVTR1_WARP_PLUGIN is set for the current PowerShell process."
Write-Output $pluginDll
