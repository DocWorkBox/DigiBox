# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$Destination = "",
    [ValidateSet("PortableV2", "LegacyV1")]
    [string]$Layout = "PortableV2",
    [string]$PythonVersion = "3.12.9",
    [string]$MainPythonVersion = "3.12.9",
    [string]$CosyVoicePythonVersion = "3.10.17",
    [string]$FeyNoBgPythonVersion = "3.12.9",
    [ValidateSet("cpu", "cuda")]
    [string]$FeyNoBgDevice = "cpu",
    [string]$UvExecutable = "",
    [string]$TorchWheelhouse = "",
    [switch]$SkipDependencies,
    [switch]$SkipModels,
    [switch]$Clean,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultSourceRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\.."))
$resolvedSourceRoot = [System.IO.Path]::GetFullPath(
    $(if ($SourceRoot) { $SourceRoot } else { $defaultSourceRoot })
).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$defaultDestination = Join-Path $resolvedSourceRoot "desktop\staging\avtr-runtime"
$resolvedDestination = [System.IO.Path]::GetFullPath(
    $(if ($Destination) { $Destination } else { $defaultDestination })
).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)

$markerName = ".avtr-portable-runtime.json"
$markerKind = "avtr1-portable-runtime"
$isPortableV2 = $Layout -eq "PortableV2"
$runtimeNames = if ($isPortableV2) {
    @("python")
} else {
    @("python-main", "python-cosyvoice", "python-feynobg")
}
$packageLayerPaths = [ordered]@{
    main = @("packages/main", "packages/shared", "src")
    cosyvoice = @(
        "packages/cosyvoice",
        "packages/shared",
        "third_party/CosyVoice",
        "third_party/CosyVoice/third_party/Matcha-TTS",
        "src"
    )
    feynobg = @("packages/feynobg", "packages/shared", "src")
}
$dependencyRequirements = [ordered]@{
    main = @(
        "requirements-windows.txt",
        "requirements-windows-nvidia-vfx.txt",
        "requirements-windows-tensorrt.txt"
    )
    cosyvoice = @("requirements-windows-cosyvoice.txt")
    feynobg = @("requirements-windows-feynobg.txt")
}
$consolidatorRelative = "scripts/desktop/consolidate_python_layers.py"
$portableTorchVersions = [ordered]@{
    torch = "2.7.1+cu128"
    torchaudio = "2.7.1+cu128"
    torchvision = "0.22.1+cu128"
}
$resolvedTorchWheelhouse = ""
$nvidiaVfxWheelName = "nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl"
$nvidiaVfxWheelSize = 490396952
$nvidiaVfxWheelSha256 = "b6cfaff5f435ad18329a1e1c1ac3ceb36f2aa6cfb0774d271c0bcc3aeaf31c53"
$resolvedNvidiaVfxWheel = ""
$payloadDirectoryExclusions = @(
    ".cache",
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "user_assets",
    "local_voices",
    "voice_clones",
    "reference_audio",
    "memory\backups",
    "memory\pending-imports",
    ".trash",
    ".engine-staging",
    ".engine-backups"
)
$payloadFileExclusions = @(
    "*.engine",
    "*.plan",
    "engine-manifest.json",
    "*grid_sample_3d_plugin*.dll",
    "spk2info.pt",
    "*.incomplete",
    "*.metadata",
    "*.pyc",
    "*.pyo",
    ".env*",
    "*.key",
    "*.pem",
    "memory.sqlite3",
    "memory.sqlite3-wal",
    "memory.sqlite3-shm",
    "digibox-memory*.json"
)

function Test-DirectoryHasEntries {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }
    return $null -ne (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
}

function Assert-SourceTree {
    $required = @(
        (Join-Path $resolvedSourceRoot "src"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\__init__.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\admin.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\api.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\extractor.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\models.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\paths.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\schema.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\service.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\sqlite_store.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\transfer.py"),
        (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\memory\worklet.py"),
        (Join-Path $resolvedSourceRoot "scripts\run_local_stream.py"),
        (Join-Path $resolvedSourceRoot "scripts\desktop\build_tensorrt.ps1"),
        (Join-Path $resolvedSourceRoot "scripts\desktop\DigiBox-TensorRT-Setup.cmd"),
        (Join-Path $resolvedSourceRoot "scripts\desktop\consolidate_python_layers.py"),
        (Join-Path $resolvedSourceRoot "third_party\CosyVoice\cosyvoice"),
        (Join-Path $resolvedSourceRoot "third_party\CosyVoice\third_party\Matcha-TTS"),
        (Join-Path $resolvedSourceRoot "artifacts\main"),
        (Join-Path $resolvedSourceRoot "LICENSE.md"),
        (Join-Path $resolvedSourceRoot "LICENSE-MODEL.md"),
        (Join-Path $resolvedSourceRoot "LICENSE-RENDERER.md"),
        (Join-Path $resolvedSourceRoot "LICENSE-STREAMER.md")
    )
    if (-not $PlanOnly) {
        $required += @(
            (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\vendor\preact.module.js"),
            (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\vendor\preact-hooks.module.js"),
            (Join-Path $resolvedSourceRoot "src\avaturn_live_streamer\vendor\htm.module.js")
        )
    }
    $tensorRtBuildInputs = @(
        "avtr1.scripted.pt",
        "hubert-lbs-avtr1.onnx",
        "decoder.onnx",
        "modnet.onnx",
        "stitch_network.onnx",
        "warp_network.onnx",
        "warp_network_ori.onnx"
    )
    foreach ($fileName in $tensorRtBuildInputs) {
        $required += Join-Path $resolvedSourceRoot "artifacts\main\build_artifacts\$fileName"
    }
    $runtimeArtifactFiles = @(
        "avtr1_normalizer.safetensors",
        "avatars_artifacts\pasteback_mask.png",
        "renderer_runtime_artifacts\appearance_extractor.onnx",
        "renderer_runtime_artifacts\motion_extractor.onnx",
        "renderer_runtime_artifacts\insightface_det.onnx",
        "renderer_runtime_artifacts\landmark106.onnx",
        "renderer_runtime_artifacts\landmark203.onnx",
        "renderer_runtime_artifacts\blaze_face.onnx",
        "renderer_runtime_artifacts\face_mesh.onnx"
    )
    foreach ($relativeArtifactPath in $runtimeArtifactFiles) {
        $required += Join-Path $resolvedSourceRoot "artifacts\main\$relativeArtifactPath"
    }
    if (-not $SkipModels) {
        $cosyVoiceModelFiles = @(
            "cosyvoice3.yaml",
            "llm.pt",
            "flow.pt",
            "hift.pt",
            "campplus.onnx",
            "speech_tokenizer_v3.onnx",
            "CosyVoice-BlankEN\config.json",
            "CosyVoice-BlankEN\generation_config.json",
            "CosyVoice-BlankEN\merges.txt",
            "CosyVoice-BlankEN\model.safetensors",
            "CosyVoice-BlankEN\tokenizer_config.json",
            "CosyVoice-BlankEN\vocab.json"
        )
        foreach ($relativeModelPath in $cosyVoiceModelFiles) {
            $required += Join-Path $resolvedSourceRoot (
                "models\Fun-CosyVoice3-0.5B-2512\$relativeModelPath"
            )
        }
    }
    $missing = @(
        $required | Where-Object {
            if (-not (Test-Path -LiteralPath $_)) {
                $true
            } else {
                $item = Get-Item -LiteralPath $_ -Force
                -not $item.PSIsContainer -and $item.Length -le 0
            }
        }
    )
    if ($missing.Count -gt 0) {
        throw "Portable Runtime source files are missing: $($missing -join ', ')"
    }
}

function Assert-SafeDestination {
    $rootPath = [System.IO.Path]::GetPathRoot($resolvedDestination).TrimEnd("\", "/")
    $destinationComparable = $resolvedDestination.TrimEnd("\", "/")
    if ([string]::Equals($destinationComparable, $rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to use a drive root as the Runtime destination: $resolvedDestination"
    }
    if (-not [string]::Equals(
        (Split-Path -Leaf $resolvedDestination),
        "avtr-runtime",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime destination leaf must be exactly 'avtr-runtime': $resolvedDestination"
    }
    if ([string]::Equals(
        $resolvedDestination,
        $resolvedSourceRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime destination cannot be the source repository root."
    }
    $sourceWithSeparator = "$resolvedSourceRoot$([System.IO.Path]::DirectorySeparatorChar)"
    $destinationWithSeparator = "$resolvedDestination$([System.IO.Path]::DirectorySeparatorChar)"
    if ($sourceWithSeparator.StartsWith(
        $destinationWithSeparator,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Runtime destination cannot be an ancestor of the source repository."
    }
    if (Test-Path -LiteralPath $resolvedDestination -PathType Leaf) {
        throw "Runtime destination is a file: $resolvedDestination"
    }
    $pathToInspect = $resolvedDestination
    while ($pathToInspect) {
        if (Test-Path -LiteralPath $pathToInspect) {
            $item = Get-Item -LiteralPath $pathToInspect -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Runtime destination and its ancestors must not be reparse points: $($item.FullName)"
            }
        }
        $parent = Split-Path -Parent $pathToInspect
        if (-not $parent -or [string]::Equals(
            $parent,
            $pathToInspect,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            break
        }
        $pathToInspect = $parent
    }
}

function Assert-CleanMarker {
    $markerPath = Join-Path $resolvedDestination $markerName
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "-Clean refused: non-empty target has no AVTR portable Runtime marker: $markerPath"
    }
    try {
        $marker = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "-Clean refused: Runtime marker is not valid JSON: $markerPath"
    }
    if ($marker.schemaVersion -ne 1 -or $marker.kind -ne $markerKind) {
        throw "-Clean refused: Runtime marker does not identify an AVTR portable Runtime: $markerPath"
    }
    try {
        $markedDestination = [System.IO.Path]::GetFullPath([string]$marker.destination).TrimEnd("\", "/")
    }
    catch {
        throw "-Clean refused: Runtime marker has no valid destination identity: $markerPath"
    }
    if (-not [string]::Equals(
        $markedDestination,
        $resolvedDestination,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "-Clean refused: Runtime marker belongs to another destination: $markedDestination"
    }
}

function Assert-NativeSuccess {
    param([Parameter(Mandatory = $true)][string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Invoke-RobocopyFiltered {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target,
        [string[]]$ExcludeDirectories = @(),
        [string[]]$ExcludeFiles = @()
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Copy source directory is missing: $Source"
    }
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    $arguments = @(
        $Source,
        $Target,
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:2",
        "/W:1",
        "/XJ",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    )
    if ($ExcludeDirectories.Count -gt 0) {
        $arguments += "/XD"
        $arguments += $ExcludeDirectories
    }
    if ($ExcludeFiles.Count -gt 0) {
        $arguments += "/XF"
        $arguments += $ExcludeFiles
    }
    & robocopy.exe @arguments | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code while copying $Source"
    }
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [string]$DestinationRelativePath = ""
    )

    $source = Join-Path $resolvedSourceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required Runtime file is missing: $source"
    }
    $targetRelative = if ($DestinationRelativePath) { $DestinationRelativePath } else { $RelativePath }
    $target = Join-Path $resolvedDestination $targetRelative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
}

function Write-Utf8Json {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value,
        [int]$Depth = 10
    )

    $json = ($Value | ConvertTo-Json -Depth $Depth) + [Environment]::NewLine
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

function Resolve-UvPath {
    $candidate = $null
    if ($UvExecutable) {
        $candidate = [System.IO.Path]::GetFullPath($UvExecutable)
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "uv executable does not exist: $candidate"
        }
    } else {
        $command = Get-Command uv -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
        }
        $candidate = $command.Source
    }
    $versionText = (& $candidate --version | Select-Object -Last 1)
    Assert-NativeSuccess "Inspecting uv version"
    $versionMatch = [regex]::Match([string]$versionText, "(\d+\.\d+\.\d+)")
    if (-not $versionMatch.Success) {
        throw "Could not parse uv version: $versionText"
    }
    $minimumVersion = [version]"0.8.0"
    $actualVersion = [version]$versionMatch.Groups[1].Value
    if ($actualVersion -lt $minimumVersion) {
        throw "uv 0.8.0 or newer is required to isolate Windows registry registration; found $actualVersion."
    }
    return $candidate
}

function Resolve-TorchWheelhouse {
    if (-not $TorchWheelhouse) {
        return ""
    }
    if (-not $isPortableV2) {
        throw "-TorchWheelhouse is supported only by the PortableV2 layout."
    }
    $candidate = [System.IO.Path]::GetFullPath($TorchWheelhouse).TrimEnd("\", "/")
    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        throw "Torch wheelhouse does not exist: $candidate"
    }
    $item = Get-Item -LiteralPath $candidate -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Torch wheelhouse must not be a reparse point: $candidate"
    }
    $requiredWheels = @(
        "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl",
        "torchaudio-2.7.1+cu128-cp312-cp312-win_amd64.whl",
        "torchvision-0.22.1+cu128-cp312-cp312-win_amd64.whl"
    )
    $availableNames = @(
        Get-ChildItem -LiteralPath $candidate -File -Force | Select-Object -ExpandProperty Name
    )
    $missing = @($requiredWheels | Where-Object { $_ -notin $availableNames })
    if ($missing.Count -gt 0) {
        throw "Torch wheelhouse is missing exact CPython 3.12 CUDA wheels: $($missing -join ', ')"
    }
    return $candidate
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $sha256.ComputeHash($stream)
            return [System.BitConverter]::ToString($bytes).Replace("-", "").ToLowerInvariant()
        }
        finally {
            $sha256.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Resolve-NvidiaVfxWheel {
    $candidate = Join-Path $resolvedSourceRoot ".downloads\$nvidiaVfxWheelName"
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        return ""
    }
    $item = Get-Item -LiteralPath $candidate -Force
    if ($item.Length -ne $nvidiaVfxWheelSize) {
        Write-Warning (
            "Ignoring local NVIDIA VFX wheel with unexpected size; " +
            "the pinned requirements URL will be used: $candidate"
        )
        return ""
    }
    $actualHash = Get-Sha256Hex -Path $candidate
    if ($actualHash -ne $nvidiaVfxWheelSha256) {
        Write-Warning (
            "Ignoring local NVIDIA VFX wheel with unexpected SHA256; " +
            "the pinned requirements URL will be used: $candidate"
        )
        return ""
    }
    return [System.IO.Path]::GetFullPath($candidate)
}

function Resolve-ManagedPythonExecutable {
    param([Parameter(Mandatory = $true)][string]$ManagedRoot)

    $managedComparable = [System.IO.Path]::GetFullPath($ManagedRoot).TrimEnd("\", "/") + "\"
    $candidates = @(
        Get-ChildItem -LiteralPath $ManagedRoot -Directory -Force |
            Where-Object {
                ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0
            } |
            ForEach-Object {
                $candidate = Join-Path $_.FullName "python.exe"
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    $candidateItem = Get-Item -LiteralPath $candidate -Force
                    if (($candidateItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                        throw "Managed Python executable must not be a reparse point: $candidate"
                    }
                    $candidatePath = [System.IO.Path]::GetFullPath($candidateItem.FullName)
                    if (-not $candidatePath.StartsWith(
                        $managedComparable,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )) {
                        throw "Resolved standalone Python escaped its isolated install root: $candidatePath"
                    }
                    $candidatePath
                }
            } |
            Sort-Object -Unique
    )
    if ($candidates.Count -ne 1) {
        throw "Expected exactly one standalone Python installation under $ManagedRoot; found $($candidates.Count)."
    }
    return $candidates[0]
}

function Install-StandalonePython {
    param(
        [Parameter(Mandatory = $true)][string]$UvPath,
        [Parameter(Mandatory = $true)][string]$Version,
        [Parameter(Mandatory = $true)][string]$RuntimeName,
        [Parameter(Mandatory = $true)][string]$DownloadRoot
    )

    $managedRoot = Join-Path $DownloadRoot $RuntimeName
    New-Item -ItemType Directory -Path $managedRoot -Force | Out-Null
    $previousInstallRoot = $env:UV_PYTHON_INSTALL_DIR
    $previousBinRoot = $env:UV_PYTHON_BIN_DIR
    $previousRegistrySetting = $env:UV_PYTHON_INSTALL_REGISTRY
    $binRoot = Join-Path $DownloadRoot "$RuntimeName-bin"
    New-Item -ItemType Directory -Path $binRoot -Force | Out-Null
    try {
        $env:UV_PYTHON_INSTALL_DIR = $managedRoot
        $env:UV_PYTHON_BIN_DIR = $binRoot
        $env:UV_PYTHON_INSTALL_REGISTRY = "0"
        & $UvPath python install $Version --install-dir $managedRoot --reinstall --managed-python --no-config |
            Out-Host
        Assert-NativeSuccess "Installing standalone CPython $Version for $RuntimeName"
    }
    finally {
        $env:UV_PYTHON_INSTALL_DIR = $previousInstallRoot
        $env:UV_PYTHON_BIN_DIR = $previousBinRoot
        $env:UV_PYTHON_INSTALL_REGISTRY = $previousRegistrySetting
    }
    $basePython = Resolve-ManagedPythonExecutable -ManagedRoot $managedRoot
    $managedComparable = [System.IO.Path]::GetFullPath($managedRoot).TrimEnd("\", "/") + "\"
    if (-not $basePython.StartsWith($managedComparable, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved standalone Python escaped its isolated install root: $basePython"
    }
    if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
        throw "uv did not return a usable standalone Python executable for ${RuntimeName}: $basePython"
    }
    $probeJson = & $basePython -I -c `
        "import json,sys; print(json.dumps({'version':'.'.join(map(str,sys.version_info[:3])),'prefix':sys.prefix}))"
    Assert-NativeSuccess "Probing standalone CPython $Version for $RuntimeName"
    $probe = ($probeJson | Select-Object -Last 1) | ConvertFrom-Json
    $baseRoot = Split-Path -Parent $basePython
    if ($probe.version -ne $Version -or -not [string]::Equals(
        [System.IO.Path]::GetFullPath([string]$probe.prefix).TrimEnd("\", "/"),
        [System.IO.Path]::GetFullPath($baseRoot).TrimEnd("\", "/"),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Standalone Python identity mismatch for ${RuntimeName}: version=$($probe.version), prefix=$($probe.prefix)"
    }
    $targetRoot = Join-Path $resolvedDestination $RuntimeName
    Invoke-RobocopyFiltered -Source $baseRoot -Target $targetRoot
    $targetPython = Join-Path $targetRoot "python.exe"
    if (-not (Test-Path -LiteralPath $targetPython -PathType Leaf)) {
        throw "Standalone Python copy is incomplete: $targetPython"
    }
    & $targetPython -I -c "import encodings,json,site,sqlite3,ssl,sys; assert '.'.join(map(str,sys.version_info[:3])) == '$Version'"
    Assert-NativeSuccess "Verifying relocated standalone CPython $Version for $RuntimeName"
    return $targetPython
}

function Invoke-UvPipInstall {
    param(
        [Parameter(Mandatory = $true)][string]$UvPath,
        [Parameter(Mandatory = $true)][string]$Python,
        [string]$Target = "",
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Step
    )

    $targetArguments = if ($Target) { @("--target", $Target) } else { @() }
    & $UvPath pip install --python $Python --link-mode copy @targetArguments @Arguments
    Assert-NativeSuccess $Step
}

function Install-RuntimeDependencies {
    param(
        [Parameter(Mandatory = $true)][string]$UvPath,
        [Parameter(Mandatory = $true)][hashtable]$PythonPaths
    )

    $previousLinkMode = $env:UV_LINK_MODE
    $previousPythonPath = $env:PYTHONPATH
    $torchConstraintPath = $null
    try {
        # Runtime files must not be hardlinked to the build machine's uv cache.
        $env:UV_LINK_MODE = "copy"
        # Host package layers must never leak into dependency resolution or builds.
        $env:PYTHONPATH = $null
        $mainPython = if ($isPortableV2) { $PythonPaths["python"] } else { $PythonPaths["python-main"] }
        $cosyvoicePython = if ($isPortableV2) { $PythonPaths["python"] } else { $PythonPaths["python-cosyvoice"] }
        $feynobgPython = if ($isPortableV2) { $PythonPaths["python"] } else { $PythonPaths["python-feynobg"] }
        $mainTarget = if ($isPortableV2) { Join-Path $resolvedDestination "packages\main" } else { "" }
        $cosyvoiceTarget = if ($isPortableV2) { Join-Path $resolvedDestination "packages\cosyvoice" } else { "" }
        $feynobgTarget = if ($isPortableV2) { Join-Path $resolvedDestination "packages\feynobg" } else { "" }
        $portableTorchSpecs = @(
            "torch==2.7.1+cu128",
            "torchaudio==2.7.1+cu128",
            "torchvision==0.22.1+cu128"
        )
        $portableSourceArguments = @()
        $portableConstraintArguments = @()
        if ($isPortableV2) {
            $torchConstraintPath = Join-Path ([System.IO.Path]::GetTempPath()) (
                "avtr-portable-torch-constraints-" + [guid]::NewGuid().ToString("N") + ".txt"
            )
            $constraintText = ($portableTorchSpecs -join [Environment]::NewLine) + [Environment]::NewLine
            [System.IO.File]::WriteAllText(
                $torchConstraintPath,
                $constraintText,
                [System.Text.UTF8Encoding]::new($false)
            )
            $portableConstraintArguments = @("-c", $torchConstraintPath)
            $portableSourceArguments = if ($resolvedTorchWheelhouse) {
                @(
                    "--find-links", $resolvedTorchWheelhouse,
                    "--index-url", "https://pypi.org/simple"
                )
            } else {
                @(
                    "--index-url", "https://download.pytorch.org/whl/cu128",
                    "--extra-index-url", "https://pypi.org/simple"
                )
            }
        }
        $portableResolutionArguments = $portableSourceArguments + $portableConstraintArguments
        # Exact +cu128 versions exist only in the selected local wheelhouse, while
        # PyPI remains available for Torch's own transitive dependencies.
        $portableTorchArguments = $portableResolutionArguments + $portableTorchSpecs
        $mainTorchArguments = if ($isPortableV2) { $portableTorchArguments } else {
            @(
                "--index-url", "https://download.pytorch.org/whl/cu128",
                "torch==2.7.1"
            )
        }
        $cosyvoiceTorchArguments = if ($isPortableV2) { $portableTorchArguments } else {
            @(
                "--index-url", "https://download.pytorch.org/whl/cu128",
                "torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1"
            )
        }
        $feyTorchIndex = if ($FeyNoBgDevice -eq "cuda") {
            "https://download.pytorch.org/whl/cu128"
        } else {
            "https://download.pytorch.org/whl/cpu"
        }
        $feynobgTorchArguments = if ($isPortableV2) { $portableTorchArguments } else {
            @(
                "--index-url", $feyTorchIndex,
                "torch==2.7.1", "torchvision==0.22.1"
            )
        }

        $mainRequirementsArguments = if ($isPortableV2) {
            $portableResolutionArguments + @("-r", (Join-Path $resolvedSourceRoot "requirements-windows.txt"))
        } else {
            @("-r", (Join-Path $resolvedSourceRoot "requirements-windows.txt"))
        }
        $tensorRtArguments = if ($isPortableV2) {
            $portableResolutionArguments + @(
                "-r", (Join-Path $resolvedSourceRoot "requirements-windows-tensorrt.txt")
            )
        } else {
            @("-r", (Join-Path $resolvedSourceRoot "requirements-windows-tensorrt.txt"))
        }
        $nvidiaVfxArguments = if ($resolvedNvidiaVfxWheel) {
            $portableResolutionArguments + @($resolvedNvidiaVfxWheel)
        } else {
            $portableResolutionArguments + @(
                "-r", (Join-Path $resolvedSourceRoot "requirements-windows-nvidia-vfx.txt")
            )
        }
        if (-not $isPortableV2 -and $resolvedNvidiaVfxWheel) {
            $nvidiaVfxArguments = @($resolvedNvidiaVfxWheel)
        } elseif (-not $isPortableV2) {
            $nvidiaVfxArguments = @(
                "-r", (Join-Path $resolvedSourceRoot "requirements-windows-nvidia-vfx.txt")
            )
        }
        $cosyvoiceRequirementsArguments = if ($isPortableV2) {
            $portableResolutionArguments + @(
                "-r", (Join-Path $resolvedSourceRoot "requirements-windows-cosyvoice.txt")
            )
        } else {
            @("-r", (Join-Path $resolvedSourceRoot "requirements-windows-cosyvoice.txt"))
        }
        $cosyvoiceBuildToolsArguments = if ($isPortableV2) {
            $portableResolutionArguments + @("setuptools<81", "wheel")
        } else {
            @("setuptools<81", "wheel")
        }
        $whisperArguments = if ($isPortableV2) {
            $portableResolutionArguments + @(
                "--no-build-isolation", "openai-whisper==20231117"
            )
        } else {
            @("--no-build-isolation", "openai-whisper==20231117")
        }
        $feynobgRequirementsArguments = if ($isPortableV2) {
            $portableResolutionArguments + @(
                "-r", (Join-Path $resolvedSourceRoot "requirements-windows-feynobg.txt")
            )
        } else {
            @("-r", (Join-Path $resolvedSourceRoot "requirements-windows-feynobg.txt"))
        }

        $mainTorchStep = if ($isPortableV2) {
            "Installing portable CUDA PyTorch into the main package layer"
        } else {
            "Installing CUDA PyTorch into python-main"
        }
        $cosyvoiceTorchStep = if ($isPortableV2) {
            "Installing portable CUDA PyTorch into the CosyVoice package layer"
        } else {
            "Installing CUDA PyTorch into python-cosyvoice"
        }
        $feynobgTorchStep = if ($isPortableV2) {
            "Installing portable CUDA PyTorch into the FeyNoBg package layer"
        } else {
            "Installing PyTorch into python-feynobg"
        }

        Invoke-UvPipInstall -UvPath $UvPath -Python $mainPython -Target $mainTarget `
            -Arguments $mainTorchArguments -Step $mainTorchStep
        Invoke-UvPipInstall -UvPath $UvPath -Python $mainPython -Target $mainTarget `
            -Arguments $mainRequirementsArguments `
            -Step "Installing main Runtime dependencies"
        Invoke-UvPipInstall -UvPath $UvPath -Python $mainPython -Target $mainTarget `
            -Arguments $nvidiaVfxArguments `
            -Step $(if ($isPortableV2) {
                "Installing NVIDIA VFX Runtime into the main package layer"
            } else {
                "Installing NVIDIA VFX Runtime into python-main"
            })
        Invoke-UvPipInstall -UvPath $UvPath -Python $mainPython -Target $mainTarget `
            -Arguments $tensorRtArguments `
            -Step "Installing TensorRT builder/runtime bindings"

        Invoke-UvPipInstall -UvPath $UvPath -Python $cosyvoicePython -Target $cosyvoiceTarget `
            -Arguments $cosyvoiceTorchArguments -Step $cosyvoiceTorchStep
        Invoke-UvPipInstall -UvPath $UvPath -Python $cosyvoicePython -Target $cosyvoiceTarget `
            -Arguments $cosyvoiceRequirementsArguments `
            -Step "Installing CosyVoice Runtime dependencies"
        Invoke-UvPipInstall -UvPath $UvPath -Python $cosyvoicePython -Target $cosyvoiceTarget `
            -Arguments $cosyvoiceBuildToolsArguments `
            -Step "Installing CosyVoice legacy build support"
        try {
            # In portable-v2 the legacy build tools live in the target layer,
            # not in base Python. Expose them only while the no-isolation build runs.
            if ($isPortableV2) {
                $env:PYTHONPATH = $cosyvoiceTarget
            }
            Invoke-UvPipInstall -UvPath $UvPath -Python $cosyvoicePython -Target $cosyvoiceTarget `
                -Arguments $whisperArguments `
                -Step "Installing the CosyVoice Whisper frontend"
        }
        finally {
            $env:PYTHONPATH = $null
        }

        Invoke-UvPipInstall -UvPath $UvPath -Python $feynobgPython -Target $feynobgTarget `
            -Arguments $feynobgTorchArguments -Step $feynobgTorchStep
        Invoke-UvPipInstall -UvPath $UvPath -Python $feynobgPython -Target $feynobgTarget `
            -Arguments $feynobgRequirementsArguments `
            -Step "Installing FeyNoBg Runtime dependencies"
    }
    finally {
        $env:UV_LINK_MODE = $previousLinkMode
        $env:PYTHONPATH = $previousPythonPath
        if ($torchConstraintPath -and (Test-Path -LiteralPath $torchConstraintPath -PathType Leaf)) {
            Remove-Item -LiteralPath $torchConstraintPath -Force
        }
    }
}

function Assert-SharedPythonDistributions {
    param(
        [Parameter(Mandatory = $true)][string]$InventoryPath,
        [Parameter(Mandatory = $true)][string[]]$Names
    )

    $inventory = Get-Content -LiteralPath $InventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($name in $Names) {
        $matches = @($inventory.distributions | Where-Object { $_.name -eq $name })
        if ($matches.Count -ne 1 -or $matches[0].layer -ne "shared") {
            throw "PortableV2 expected one shared '$name' distribution; inspect $InventoryPath."
        }
    }
}

function Assert-PortableLayerBootstrap {
    param([Parameter(Mandatory = $true)][string]$Python)

    $previousPythonPath = $env:PYTHONPATH
    $mainLayer = Join-Path $resolvedDestination "packages\main"
    $sharedLayer = Join-Path $resolvedDestination "packages\shared"
    try {
        $env:PYTHONPATH = "$mainLayer$([System.IO.Path]::PathSeparator)$sharedLayer"
        & $Python -B -c "import sys; assert 'sitecustomize' in sys.modules"
        Assert-NativeSuccess "Verifying portable Python layer bootstrap"
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }
}

function Assert-NvidiaVfxImport {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [string[]]$PythonPathEntries = @()
    )

    $previousPythonPath = $env:PYTHONPATH
    $previousNoUserSite = $env:PYTHONNOUSERSITE
    $previousDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
    try {
        $env:PYTHONPATH = $PythonPathEntries -join [System.IO.Path]::PathSeparator
        $env:PYTHONNOUSERSITE = "1"
        $env:PYTHONDONTWRITEBYTECODE = "1"
        & $Python -B -c "import nvvfx; import avaturn_live_streamer.local_stream_cli; import avaturn_live_streamer.memory.admin; import avaturn_live_streamer.memory.api"
        Assert-NativeSuccess "Verifying main Runtime and memory application imports"
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:PYTHONNOUSERSITE = $previousNoUserSite
        $env:PYTHONDONTWRITEBYTECODE = $previousDontWriteBytecode
    }
}

function Assert-NvidiaVfxLayerInventory {
    param([Parameter(Mandatory = $true)][string]$InventoryPath)

    if (-not $isPortableV2 -or -not (Test-Path -LiteralPath $InventoryPath -PathType Leaf)) {
        return
    }
    $inventory = Get-Content -LiteralPath $InventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $matches = @($inventory.distributions | Where-Object { $_.name -eq "nvidia-vfx" })
    if (
        $matches.Count -ne 1 -or
        $matches[0].layer -ne "main" -or
        $matches[0].profile -ne "main"
    ) {
        throw "PortableV2 expected one retained 'nvidia-vfx' distribution in the main layer."
    }
}

function Remove-GeneratedCaches {
    $cacheDirectories = @(Get-ChildItem -LiteralPath $resolvedDestination -Recurse -Force -Directory |
        Where-Object { $_.Name -in @("__pycache__", ".cache", ".pytest_cache", ".ruff_cache") })
    foreach ($directory in ($cacheDirectories | Sort-Object { $_.FullName.Length } -Descending)) {
        if ($directory.FullName.StartsWith(
            "$resolvedDestination$([System.IO.Path]::DirectorySeparatorChar)",
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            Remove-Item -LiteralPath $directory.FullName -Recurse -Force
        }
    }
    $cacheFiles = @(Get-ChildItem -LiteralPath $resolvedDestination -Recurse -Force -File |
        Where-Object { $_.Extension -in @(".pyc", ".pyo") })
    foreach ($file in $cacheFiles) {
        Remove-Item -LiteralPath $file.FullName -Force
    }
}

function Assert-NoForbiddenPayload {
    $forbidden = @()
    $pythonRootPrefixes = if ($isPortableV2) {
        @(
            ((Join-Path $resolvedDestination "python").ToLowerInvariant() + "\"),
            ((Join-Path $resolvedDestination "packages").ToLowerInvariant() + "\")
        )
    } else {
        @(
            ((Join-Path $resolvedDestination "python-main").ToLowerInvariant() + "\"),
            ((Join-Path $resolvedDestination "python-cosyvoice").ToLowerInvariant() + "\"),
            ((Join-Path $resolvedDestination "python-feynobg").ToLowerInvariant() + "\")
        )
    }
    foreach ($entry in (Get-ChildItem -LiteralPath $resolvedDestination -Recurse -Force)) {
        $name = $entry.Name.ToLowerInvariant()
        $full = $entry.FullName.ToLowerInvariant()
        $underManagedPython = $null -ne ($pythonRootPrefixes | Where-Object {
            $full.StartsWith($_)
        } | Select-Object -First 1)
        $trustedCaBundle = $underManagedPython -and (
            $full.EndsWith("\certifi\cacert.pem") -or
            $full.EndsWith("\lib\site-packages\pip\_vendor\certifi\cacert.pem")
        )
        if (
            ($entry.PSIsContainer -and $name -in @(
                "__pycache__", ".cache", ".pytest_cache", ".ruff_cache",
                "user_assets", "local_voices", "voice_clones", "reference_audio", ".trash"
            )) -or
            ($entry.PSIsContainer -and $full.EndsWith(
                "\artifacts\main\avatars_artifacts\backgrounds"
            )) -or
            ($entry.PSIsContainer -and (
                $full.EndsWith("\memory\backups") -or
                $full.EndsWith("\memory\pending-imports")
            ))
        ) {
            $forbidden += $entry.FullName
            continue
        }
        if (-not $entry.PSIsContainer -and (
            $entry.Extension -ieq ".engine" -or
            $entry.Extension -ieq ".plan" -or
            ($name.Contains("grid_sample_3d_plugin") -and $entry.Extension -ieq ".dll") -or
            $name -eq "spk2info.pt" -or
            $entry.Extension -ieq ".incomplete" -or
            $entry.Extension -ieq ".metadata" -or
            ($name.StartsWith(".env") -and $name -ne ".env.example") -or
            $entry.Extension -ieq ".key" -or
            ($entry.Extension -ieq ".pem" -and -not $trustedCaBundle) -or
            $name -in @("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm") -or
            ($name.StartsWith("digibox-memory") -and $entry.Extension -ieq ".json") -or
            $full.Contains("\user_assets\")
        )) {
            $forbidden += $entry.FullName
        }
    }
    if ($forbidden.Count -gt 0) {
        throw "Portable Runtime contains forbidden private/cache/engine payloads: $($forbidden -join ', ')"
    }
    $requiredMemoryModules = @(
        "src\avaturn_live_streamer\memory\__init__.py",
        "src\avaturn_live_streamer\memory\admin.py",
        "src\avaturn_live_streamer\memory\api.py",
        "src\avaturn_live_streamer\memory\extractor.py",
        "src\avaturn_live_streamer\memory\models.py",
        "src\avaturn_live_streamer\memory\paths.py",
        "src\avaturn_live_streamer\memory\schema.py",
        "src\avaturn_live_streamer\memory\service.py",
        "src\avaturn_live_streamer\memory\sqlite_store.py",
        "src\avaturn_live_streamer\memory\transfer.py",
        "src\avaturn_live_streamer\memory\worklet.py"
    )
    $missingMemoryModules = @($requiredMemoryModules | Where-Object {
        $candidate = Join-Path $resolvedDestination $_
        -not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
        (Get-Item -LiteralPath $candidate).Length -le 0
    })
    if ($missingMemoryModules.Count -gt 0) {
        throw "Portable Runtime memory package is incomplete; missing: $($missingMemoryModules -join ', ')"
    }
    foreach ($runtimeName in $runtimeNames) {
        $python = Join-Path $resolvedDestination "$runtimeName\python.exe"
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
            throw "Portable Runtime is missing $runtimeName\python.exe"
        }
    }
}

Assert-SourceTree
Assert-SafeDestination
if ($isPortableV2 -and -not $PythonVersion.StartsWith("3.12.")) {
    throw "PortableV2 requires one CPython 3.12 runtime; requested $PythonVersion."
}
$resolvedTorchWheelhouse = Resolve-TorchWheelhouse
$resolvedNvidiaVfxWheel = Resolve-NvidiaVfxWheel
$destinationHasEntries = Test-DirectoryHasEntries -Path $resolvedDestination
if ($destinationHasEntries) {
    if (-not $Clean) {
        throw "Runtime destination is non-empty. Choose another path or rerun with -Clean: $resolvedDestination"
    }
    Assert-CleanMarker
}

$plan = [ordered]@{
    schemaVersion = $(if ($isPortableV2) { 2 } else { 1 })
    kind = "avtr1-portable-runtime-build-plan"
    layout = $(if ($isPortableV2) { "portable-v2" } else { "portable-v1" })
    sourceRoot = $resolvedSourceRoot
    destination = $resolvedDestination
    clean = [bool]$Clean
    dependencyLinkMode = "copy"
    includeDependencies = -not [bool]$SkipDependencies
    includeModels = -not [bool]$SkipModels
    feynobgDevice = $FeyNoBgDevice
    effectiveFeynobgTorch = $(if ($isPortableV2) { "cuda" } else { $FeyNoBgDevice })
    dependencyRequirements = $dependencyRequirements
    nvidiaVfx = [ordered]@{
        distribution = "nvidia-vfx==0.1.0.1"
        import = "nvvfx"
        layer = $(if ($isPortableV2) { "packages/main" } else { "python-main" })
        source = $(if ($resolvedNvidiaVfxWheel) { "verified-local-wheel" } else { "pinned-requirement-url" })
        wheel = $(if ($resolvedNvidiaVfxWheel) { $resolvedNvidiaVfxWheel } else { $null })
        sha256 = $nvidiaVfxWheelSha256
        licenses = "nvidia_vfx-0.1.0.1.dist-info/licenses/packaging"
    }
    excludedDirectories = $payloadDirectoryExclusions
    excludedFiles = $payloadFileExclusions
}
if ($isPortableV2) {
    $plan.torch = [ordered]@{
        source = $(if ($resolvedTorchWheelhouse) { "wheelhouse" } else { "pytorch-cu128" })
        wheelhouse = $(if ($resolvedTorchWheelhouse) { $resolvedTorchWheelhouse } else { $null })
        versions = $portableTorchVersions
    }
    $plan.python = [ordered]@{
        version = $PythonVersion
        executable = "python/python.exe"
        packageLayers = $packageLayerPaths
        layerInventory = "packages/python-layer-inventory.json"
    }
    $plan.consolidator = $consolidatorRelative
} else {
    $plan.runtimes = @(
        [ordered]@{ name = "python-main"; pythonVersion = $MainPythonVersion },
        [ordered]@{ name = "python-cosyvoice"; pythonVersion = $CosyVoicePythonVersion },
        [ordered]@{ name = "python-feynobg"; pythonVersion = $FeyNoBgPythonVersion }
    )
}
if ($PlanOnly) {
    Write-Output ($plan | ConvertTo-Json -Depth 8 -Compress)
    return
}

if ($destinationHasEntries) {
    # Assert-CleanMarker above validated the exact, resolved target before this
    # recursive operation. No unresolved variable, wildcard or broad parent is used.
    Remove-Item -LiteralPath $resolvedDestination -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedDestination -Force | Out-Null
$marker = [ordered]@{
    schemaVersion = 1
    kind = $markerKind
    layout = $plan.layout
    destination = $resolvedDestination
}
Write-Utf8Json -Path (Join-Path $resolvedDestination $markerName) -Value $marker -Depth 4

$downloadRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "avtr-runtime-python-" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
try {
    $uvPath = Resolve-UvPath
    $pythonPaths = @{}
    if ($isPortableV2) {
        $pythonPaths["python"] = Install-StandalonePython `
            -UvPath $uvPath `
            -Version $PythonVersion `
            -RuntimeName "python" `
            -DownloadRoot $downloadRoot
        foreach ($layerName in @("main", "cosyvoice", "feynobg", "shared")) {
            New-Item -ItemType Directory `
                -Path (Join-Path $resolvedDestination "packages\$layerName") `
                -Force | Out-Null
        }
    } else {
        foreach ($runtime in $plan.runtimes) {
            $pythonPaths[$runtime.name] = Install-StandalonePython `
                -UvPath $uvPath `
                -Version $runtime.pythonVersion `
                -RuntimeName $runtime.name `
                -DownloadRoot $downloadRoot
        }
    }

    Invoke-RobocopyFiltered `
        -Source (Join-Path $resolvedSourceRoot "src") `
        -Target (Join-Path $resolvedDestination "src") `
        -ExcludeDirectories $payloadDirectoryExclusions `
        -ExcludeFiles $payloadFileExclusions
    Invoke-RobocopyFiltered `
        -Source (Join-Path $resolvedSourceRoot "scripts") `
        -Target (Join-Path $resolvedDestination "scripts") `
        -ExcludeDirectories $payloadDirectoryExclusions `
        -ExcludeFiles $payloadFileExclusions
    Invoke-RobocopyFiltered `
        -Source (Join-Path $resolvedSourceRoot "artifacts\main") `
        -Target (Join-Path $resolvedDestination "artifacts\main") `
        -ExcludeDirectories (
            $payloadDirectoryExclusions + @(
                "build",
                (Join-Path $resolvedSourceRoot "artifacts\main\avatars_artifacts\backgrounds")
            )
        ) `
        -ExcludeFiles $payloadFileExclusions

    if (-not $SkipModels) {
        Invoke-RobocopyFiltered `
            -Source (Join-Path $resolvedSourceRoot "models") `
            -Target (Join-Path $resolvedDestination "models") `
            -ExcludeDirectories $payloadDirectoryExclusions `
            -ExcludeFiles ($payloadFileExclusions + @("*.wav", "*.flac", "*.ogg", "*.mp3"))
    } else {
        New-Item -ItemType Directory -Path (Join-Path $resolvedDestination "models") -Force | Out-Null
    }

    Invoke-RobocopyFiltered `
        -Source (Join-Path $resolvedSourceRoot "third_party\CosyVoice\cosyvoice") `
        -Target (Join-Path $resolvedDestination "third_party\CosyVoice\cosyvoice") `
        -ExcludeDirectories $payloadDirectoryExclusions `
        -ExcludeFiles $payloadFileExclusions
    Invoke-RobocopyFiltered `
        -Source (Join-Path $resolvedSourceRoot "third_party\CosyVoice\third_party\Matcha-TTS") `
        -Target (Join-Path $resolvedDestination "third_party\CosyVoice\third_party\Matcha-TTS") `
        -ExcludeDirectories ($payloadDirectoryExclusions + @("notebooks")) `
        -ExcludeFiles ($payloadFileExclusions + @("*.ipynb"))
    Copy-RequiredFile -RelativePath "third_party\CosyVoice\LICENSE"
    Copy-RequiredFile -RelativePath "third_party\CosyVoice\README.md"

    $rootFiles = @(
        "LICENSE.md",
        "LICENSE-MODEL.md",
        "LICENSE-RENDERER.md",
        "LICENSE-STREAMER.md",
        "PATENTS.md",
        "THIRD-PARTY-NOTICES.md",
        "README.md",
        ".env.example",
        "pyproject.toml",
        "requirements-windows.txt",
        "requirements-windows-nvidia-vfx.txt",
        "requirements-windows-tensorrt.txt",
        "requirements-windows-cosyvoice.txt",
        "requirements-windows-feynobg.txt"
    )
    foreach ($relativePath in $rootFiles) {
        Copy-RequiredFile -RelativePath $relativePath
    }

    if (-not $SkipDependencies) {
        Install-RuntimeDependencies -UvPath $uvPath -PythonPaths $pythonPaths
    }

    Remove-GeneratedCaches

    if ($isPortableV2) {
        $consolidator = Join-Path $resolvedDestination ($consolidatorRelative.Replace("/", "\"))
        & $pythonPaths["python"] -B $consolidator --runtime-root $resolvedDestination | Out-Host
        Assert-NativeSuccess "Consolidating identical Python package layers"
        Assert-PortableLayerBootstrap -Python $pythonPaths["python"]
        if (-not $SkipDependencies) {
            Assert-NvidiaVfxLayerInventory `
                -InventoryPath (Join-Path $resolvedDestination "packages\python-layer-inventory.json")
            Assert-SharedPythonDistributions `
                -InventoryPath (Join-Path $resolvedDestination "packages\python-layer-inventory.json") `
                -Names @("torch", "torchaudio", "torchvision")
            Assert-NvidiaVfxImport `
                -Python $pythonPaths["python"] `
                -PythonPathEntries @(
                    (Join-Path $resolvedDestination "packages\main"),
                    (Join-Path $resolvedDestination "packages\shared"),
                    (Join-Path $resolvedDestination "src")
                )
        }
    } elseif (-not $SkipDependencies) {
        Assert-NvidiaVfxImport `
            -Python $pythonPaths["python-main"] `
            -PythonPathEntries @((Join-Path $resolvedDestination "src"))
    }

    Assert-NoForbiddenPayload

    $package = Get-Content -LiteralPath (Join-Path $resolvedSourceRoot "package.json") -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $runtimeManifest = [ordered]@{
        schemaVersion = $(if ($isPortableV2) { 2 } else { 1 })
        layout = $plan.layout
        platform = "win32"
        arch = "x64"
        runtimeId = "avtr1-$($package.version)-win64"
        createdAt = (Get-Date).ToUniversalTime().ToString("o")
        components = [ordered]@{
            dependenciesIncluded = -not [bool]$SkipDependencies
            modelsIncluded = -not [bool]$SkipModels
            frontendVendorIncluded = $true
            tensorRtBuildInputsIncluded = $true
        }
        tensorrt = [ordered]@{
            version = "10.11.0.33"
            cudaMajor = 12
            computeCapability = $null
            engines = @()
            status = "build-required-on-target-machine"
        }
        privacy = [ordered]@{
            userAssetsIncluded = $false
            localVoiceCacheIncluded = $false
            machineSpecificEnginesIncluded = $false
            exclusions = @(
                "artifacts/main/user_assets",
                "**/*.engine",
                "**/grid_sample_3d_plugin.dll",
                "models/**/spk2info.pt",
                "**/.cache",
                "**/*.incomplete",
                "**/memory.sqlite3",
                "**/memory.sqlite3-wal",
                "**/memory.sqlite3-shm",
                "**/memory/backups",
                "**/memory/pending-imports",
                "**/digibox-memory*.json"
            )
        }
    }
    if ($isPortableV2) {
        $runtimeManifest.paths = [ordered]@{
            python = "python/python.exe"
            orchestrator = "scripts/run_local_stream.py"
            source = "src"
            artifacts = "artifacts/main"
            models = "models"
            tensorRtAssistant = "scripts/desktop/DigiBox-TensorRT-Setup.cmd"
        }
        $runtimeManifest.python = [ordered]@{
            version = $PythonVersion
            packageLayers = $packageLayerPaths
            layerInventory = "packages/python-layer-inventory.json"
        }
    } else {
        $runtimeManifest.paths = [ordered]@{
            mainPython = "python-main/python.exe"
            cosyvoicePython = "python-cosyvoice/python.exe"
            feynobgPython = "python-feynobg/python.exe"
            orchestrator = "scripts/run_local_stream.py"
            source = "src"
            artifacts = "artifacts/main"
            models = "models"
            tensorRtAssistant = "scripts/desktop/DigiBox-TensorRT-Setup.cmd"
        }
        $runtimeManifest.python = [ordered]@{
            main = $MainPythonVersion
            cosyvoice = $CosyVoicePythonVersion
            feynobg = $FeyNoBgPythonVersion
        }
    }
    Write-Utf8Json `
        -Path (Join-Path $resolvedDestination "runtime-manifest.json") `
        -Value $runtimeManifest `
        -Depth 10

    Write-Host "Portable DigiBox Runtime created: $resolvedDestination" -ForegroundColor Green
    Write-Host "TensorRT engines and private user data were intentionally excluded."
}
finally {
    $temporaryRoot = [System.IO.Path]::GetFullPath($downloadRoot)
    $expectedTemporaryPrefix = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if (
        (Test-Path -LiteralPath $temporaryRoot) -and
        $temporaryRoot.StartsWith($expectedTemporaryPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $temporaryRoot).StartsWith("avtr-runtime-python-", [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
