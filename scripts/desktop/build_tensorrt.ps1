[CmdletBinding()]
param(
    [string]$RuntimeRoot = "",
    [ValidateSet("Standard", "Full")]
    [string]$Mode = "Standard"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Resolve-PortableRuntimePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ([string]::IsNullOrWhiteSpace($RelativePath)) {
        throw "$Label must be a non-empty relative path."
    }
    $normalized = $RelativePath.Replace("/", "\")
    if ([System.IO.Path]::IsPathRooted($normalized)) {
        throw "$Label must be relative to the Runtime root: $RelativePath"
    }
    $resolved = Get-FullPath (Join-Path $Root $normalized)
    $rootPrefix = $Root.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith(
        $rootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escapes the Runtime root: $RelativePath"
    }
    return $resolved
}

function Resolve-PortableV2MainRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$Manifest
    )

    if ($Manifest.schemaVersion -ne 2 -or $Manifest.layout -ne "portable-v2") {
        throw "Portable-v2 TensorRT builds require schemaVersion 2 and layout portable-v2."
    }
    $pythonRelative = ([string]$Manifest.paths.python).Replace("\", "/")
    if ($pythonRelative -ne "python/python.exe") {
        throw "Portable-v2 TensorRT builds require paths.python to be python/python.exe."
    }
    $python = Resolve-PortableRuntimePath `
        -Root $Root `
        -RelativePath ([string]$Manifest.paths.python) `
        -Label "paths.python"
    $layers = @($Manifest.python.packageLayers.main)
    if ($layers.Count -eq 0) {
        throw "Portable-v2 TensorRT builds require python.packageLayers.main."
    }
    $resolvedLayers = @($layers | ForEach-Object {
        Resolve-PortableRuntimePath `
            -Root $Root `
            -RelativePath ([string]$_) `
            -Label "python.packageLayers.main"
    })
    $missingLayers = @($resolvedLayers | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Container)
    })
    if ($missingLayers.Count -gt 0) {
        throw "Portable-v2 main Python layers are missing: $($missingLayers -join ', ')"
    }
    return [pscustomobject]@{
        Python = $python
        PythonPath = $resolvedLayers -join ";"
    }
}

function Get-ArtifactRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$ArtifactRoot,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $base = (Get-FullPath $ArtifactRoot).TrimEnd("\") + "\"
    $full = Get-FullPath $Path
    if (-not $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact path escapes the active artifact root: $full"
    }
    return $full.Substring($base.Length)
}

function Assert-AvtrServicesStopped {
    $ports = @(7860, 8000, 8767, 8768)
    $listeners = @()
    foreach ($port in $ports) {
        $connections = @(
            Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        )
        foreach ($connection in $connections) {
            $listeners += [pscustomobject]@{
                Port = $port
                ProcessId = $connection.OwningProcess
            }
        }
    }
    if ($listeners.Count -gt 0) {
        $summary = ($listeners | ForEach-Object {
            "port $($_.Port) (PID $($_.ProcessId))"
        }) -join ", "
        throw (
            "DigiBox is still running: $summary. " +
            "Close DigiBox and all compatible backend services before building engines."
        )
    }
}

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-WarpPluginBuild {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$BuildRoot,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    $expectedPlugin = [System.IO.Path]::GetFullPath($OutputPath)
    if (Test-Path -LiteralPath $expectedPlugin) {
        throw "Refusing to reuse a pre-existing staged Warp plugin: $expectedPlugin"
    }
    & $ScriptPath `
        -Python $Python `
        -BuildRoot $BuildRoot `
        -OutputPath $expectedPlugin
    if (-not (Test-Path -LiteralPath $expectedPlugin -PathType Leaf)) {
        throw (
            "Warp plugin builder returned successfully but the expected DLL is missing: " +
            $expectedPlugin
        )
    }
}

function Invoke-EngineInspection {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Root,
        [string[]]$EnginePaths = @()
    )
    $arguments = @(
        (Join-Path $Root "scripts\desktop\inspect_runtime.py"),
        "--runtime-root",
        $Root,
        "--probe-engines",
        "--json"
    )
    foreach ($enginePath in $EnginePaths) {
        $arguments += @("--engine-path", $enginePath)
    }
    $json = & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "TensorRT engine inspection failed with exit code $LASTEXITCODE"
    }
    $inspection = $json | ConvertFrom-Json
    $failed = @(
        $inspection.engine_probe.PSObject.Properties | Where-Object {
            -not $_.Value.ok
        }
    )
    if ($failed.Count -gt 0) {
        $names = ($failed | ForEach-Object { $_.Name }) -join ", "
        throw "TensorRT engine deserialization failed: $names"
    }
    return $inspection
}

function Backup-ActiveArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$ArtifactRoot,
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [Parameter(Mandatory = $true)][object[]]$Targets,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )
    $state = @()
    foreach ($target in $Targets) {
        $relative = Get-ArtifactRelativePath -ArtifactRoot $ArtifactRoot -Path $target
        $backup = Join-Path $BackupRoot $relative
        $hadOriginal = Test-Path -LiteralPath $target
        if ($hadOriginal) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
            Copy-Item -LiteralPath $target -Destination $backup
        }
        $state += [pscustomobject]@{
            Target = $target
            HadOriginal = $hadOriginal
            Backup = $backup
        }
    }

    $manifestBackup = Join-Path $BackupRoot "engine-manifest.json"
    $manifestExisted = Test-Path -LiteralPath $ManifestPath
    if ($manifestExisted) {
        New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
        Copy-Item -LiteralPath $ManifestPath -Destination $manifestBackup
    }
    return [pscustomobject]@{
        Targets = $state
        ManifestPath = $ManifestPath
        ManifestExisted = $manifestExisted
        ManifestBackup = $manifestBackup
    }
}

function Install-ActiveArtifacts {
    param(
        [Parameter(Mandatory = $true)][object[]]$InstallEntries,
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$RetireTargets,
        [Parameter(Mandatory = $true)][string]$Timestamp
    )
    $incoming = @()
    foreach ($entry in $InstallEntries) {
        $targetDirectory = Split-Path -Parent $entry.Target
        New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
        $temporary = "$($entry.Target).incoming.$Timestamp"
        Copy-Item -LiteralPath $entry.Source -Destination $temporary -Force
        $incoming += [pscustomobject]@{
            Temporary = $temporary
            Target = $entry.Target
        }
    }

    try {
        foreach ($item in $incoming) {
            Move-Item -LiteralPath $item.Temporary -Destination $item.Target -Force
        }
        foreach ($target in $RetireTargets) {
            if (Test-Path -LiteralPath $target) {
                Remove-Item -LiteralPath $target -Force
            }
        }
    }
    finally {
        foreach ($item in $incoming) {
            if (Test-Path -LiteralPath $item.Temporary) {
                Remove-Item -LiteralPath $item.Temporary -Force
            }
        }
    }
}

function Restore-ActiveArtifacts {
    param([Parameter(Mandatory = $true)][object]$BackupState)
    foreach ($item in $BackupState.Targets) {
        if ($item.HadOriginal) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $item.Target) | Out-Null
            Copy-Item -LiteralPath $item.Backup -Destination $item.Target -Force
        }
        elseif (Test-Path -LiteralPath $item.Target) {
            Remove-Item -LiteralPath $item.Target -Force
        }
    }
    if ($BackupState.ManifestExisted) {
        Copy-Item -LiteralPath $BackupState.ManifestBackup `
            -Destination $BackupState.ManifestPath -Force
    }
    elseif (Test-Path -LiteralPath $BackupState.ManifestPath) {
        Remove-Item -LiteralPath $BackupState.ManifestPath -Force
    }
}

function Invoke-ArtifactInstallTransaction {
    param(
        [Parameter(Mandatory = $true)][object]$BackupState,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    try {
        & $Action
    }
    catch {
        $installError = $_
        Write-Warning `
            "Engine installation failed; starting automatic rollback." `
            -WarningAction Continue
        try {
            Restore-ActiveArtifacts -BackupState $BackupState
        }
        catch {
            $rollbackError = $_
            $message = @(
                "Engine installation failed and automatic rollback also failed."
                "Install error: $($installError.Exception.Message)"
                "Rollback error: $($rollbackError.Exception.Message)"
            ) -join " "
            $innerErrors = [System.Collections.Generic.List[System.Exception]]::new()
            $innerErrors.Add($installError.Exception)
            $innerErrors.Add($rollbackError.Exception)
            throw [System.AggregateException]::new($message, $innerErrors)
        }
        Write-Warning `
            "Rollback completed. The previous active engine set was restored." `
            -WarningAction Continue
        throw
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultRoot = Get-FullPath (Join-Path $scriptRoot "..\..")
$root = if ($RuntimeRoot) { Get-FullPath $RuntimeRoot } else { $defaultRoot }
$runtimeManifestPath = Join-Path $root "runtime-manifest.json"
$portableV2Main = $null
if (Test-Path -LiteralPath $runtimeManifestPath -PathType Leaf) {
    $runtimeManifest = Get-Content `
        -LiteralPath $runtimeManifestPath `
        -Raw `
        -Encoding UTF8 | ConvertFrom-Json
    $layoutProperty = $runtimeManifest.PSObject.Properties["layout"]
    $manifestLayout = if ($null -eq $layoutProperty) {
        $null
    }
    else {
        [string]$layoutProperty.Value
    }
    if ($runtimeManifest.schemaVersion -eq 2 -or $manifestLayout -eq "portable-v2") {
        $portableV2Main = Resolve-PortableV2MainRuntime `
            -Root $root `
            -Manifest $runtimeManifest
    }
    elseif (
        $runtimeManifest.schemaVersion -ne 1 -or
        ($null -ne $manifestLayout -and $manifestLayout -ne "portable-v1")
    ) {
        throw "Unsupported DigiBox Runtime manifest: $runtimeManifestPath"
    }
}

$python = if ($null -ne $portableV2Main) {
    Join-Path $root "python\python.exe"
}
else {
    $pythonCandidates = @(
        (Join-Path $root "python-main\python.exe"),
        (Join-Path $root ".venv\Scripts\python.exe")
    )
    $pythonCandidates | Where-Object {
        Test-Path -LiteralPath $_
    } | Select-Object -First 1
}
if (-not $python) {
    throw "DigiBox Python runtime was not found under: $root"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "DigiBox Python runtime does not exist: $python"
}

$artifacts = Join-Path $root "artifacts\main"
$buildInputs = Join-Path $artifacts "build_artifacts"
$requiredInputs = @(
    "avtr1.scripted.pt",
    "hubert-lbs-avtr1.onnx",
    "decoder.onnx",
    "modnet.onnx",
    "stitch_network.onnx"
)
if ($Mode -eq "Full") {
    $requiredInputs += "warp_network.onnx"
}
$missing = @($requiredInputs | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $buildInputs $_))
})
if ($missing.Count -gt 0) {
    throw (
        "TensorRT portable build inputs are missing: " +
        ($missing -join ", ") +
        ". Install the offline artifact payload first."
    )
}

Assert-AvtrServicesStopped

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$stageRoot = Join-Path $artifacts ".engine-staging\$timestamp"
$speechStage = Join-Path $stageRoot "speech"
$rendererStage = Join-Path $stageRoot "renderer"
$backupRoot = Join-Path $artifacts ".engine-backups\$timestamp"
New-Item -ItemType Directory -Force -Path $speechStage, $rendererStage | Out-Null

$environmentNames = @(
    "PYTHONPATH",
    "AVTR1_MAIN_PYTHONPATH",
    "AVTR1_LOCAL_STORAGE",
    "PYTHONNOUSERSITE",
    "PYTHONUTF8"
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$sourcePath = Join-Path $root "src"
$isManagedRuntime = Test-Path -LiteralPath (Join-Path $root "python-main\python.exe")
if ($null -ne $portableV2Main) {
    $env:PYTHONPATH = $portableV2Main.PythonPath
    $env:AVTR1_MAIN_PYTHONPATH = $portableV2Main.PythonPath
}
elseif ($isManagedRuntime -or -not $previousEnvironment["PYTHONPATH"]) {
    $env:PYTHONPATH = $sourcePath
}
else {
    $env:PYTHONPATH = "$sourcePath;$($previousEnvironment['PYTHONPATH'])"
}
$env:AVTR1_LOCAL_STORAGE = Join-Path $root "artifacts"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"

try {
    Invoke-NativeStep "Check NVIDIA, CUDA and TensorRT runtime" {
        & $python (Join-Path $root "scripts\windows_diagnostics.py") --require-tensorrt
    }
    Invoke-NativeStep "Build AVTR-1 speech-to-motion engines" {
        & $python (Join-Path $root "scripts\build_avtr1_engines.py") `
            --ckpt (Join-Path $buildInputs "avtr1.scripted.pt") `
            --out-dir $speechStage
    }
    Invoke-NativeStep "Build HuBERT engine" {
        & $python (Join-Path $root "scripts\build_hubert_engine.py") `
            --onnx (Join-Path $buildInputs "hubert-lbs-avtr1.onnx") `
            --out (Join-Path $speechStage "hubert_lbs_fp16.engine")
    }
    Invoke-NativeStep "Build standard renderer engines" {
        & $python (Join-Path $root "scripts\build_renderer_engines.py") `
            decoder modnet stitch `
            --decoder-onnx (Join-Path $buildInputs "decoder.onnx") `
            --modnet-onnx (Join-Path $buildInputs "modnet.onnx") `
            --stitch-onnx (Join-Path $buildInputs "stitch_network.onnx") `
            --out-dir $rendererStage
    }

    if ($Mode -eq "Full") {
        Write-Warning (
            "Full mode requires Visual Studio 2022 C++ Build Tools, CMake, " +
            "Ninja, Git and the CUDA Toolkit."
        )
        $pluginBuildRoot = Join-Path $stageRoot "warp-plugin"
        $stagedPlugin = Join-Path $rendererStage "grid_sample_3d_plugin.dll"
        Invoke-WarpPluginBuild `
            -ScriptPath (Join-Path $root "scripts\build_warp_plugin_windows.ps1") `
            -Python $python `
            -BuildRoot $pluginBuildRoot `
            -OutputPath $stagedPlugin
        Invoke-NativeStep "Validate the Warp plugin" {
            & $python (Join-Path $root "scripts\smoke_warp_plugin_windows.py") `
                --plugin $stagedPlugin --max-batch 5
        }
        Invoke-NativeStep "Build the Warp TensorRT engine" {
            & $python (Join-Path $root "scripts\build_renderer_engines.py") `
                warp `
                --warp-onnx (Join-Path $buildInputs "warp_network.onnx") `
                --warp-plugin $stagedPlugin `
                --out-dir $rendererStage
        }
    }

    $speechTarget = Join-Path $artifacts "speech2motion_runtime_artifacts_cc_win64"
    $rendererTarget = Join-Path $artifacts "renderer_runtime_artifacts_cc_win64"
    $installEntries = @(
        [pscustomobject]@{
            Source = Join-Path $speechStage "avtr1_encode_fp16.engine"
            Target = Join-Path $speechTarget "avtr1_encode_fp16.engine"
        },
        [pscustomobject]@{
            Source = Join-Path $speechStage "avtr1_decode_fp16.engine"
            Target = Join-Path $speechTarget "avtr1_decode_fp16.engine"
        },
        [pscustomobject]@{
            Source = Join-Path $speechStage "avtr1_normalizer.safetensors"
            Target = Join-Path $speechTarget "avtr1_normalizer.safetensors"
        },
        [pscustomobject]@{
            Source = Join-Path $speechStage "hubert_lbs_fp16.engine"
            Target = Join-Path $speechTarget "hubert_lbs_fp16.engine"
        },
        [pscustomobject]@{
            Source = Join-Path $rendererStage "decoder_b5_fp16.engine"
            Target = Join-Path $rendererTarget "decoder_b5_fp16.engine"
        },
        [pscustomobject]@{
            Source = Join-Path $rendererStage "modnet_b5_fp16.engine"
            Target = Join-Path $rendererTarget "modnet_b5_fp16.engine"
        },
        [pscustomobject]@{
            Source = Join-Path $rendererStage "stitch_network_b5_fp16.engine"
            Target = Join-Path $rendererTarget "stitch_network_b5_fp16.engine"
        }
    )
    $warpTarget = Join-Path $rendererTarget "warp_network_b5_fp16.engine"
    $pluginTarget = Join-Path $rendererTarget "grid_sample_3d_plugin.dll"
    if ($Mode -eq "Full") {
        $installEntries += @(
            [pscustomobject]@{
                Source = Join-Path $rendererStage "warp_network_b5_fp16.engine"
                Target = $warpTarget
            },
            [pscustomobject]@{
                Source = Join-Path $rendererStage "grid_sample_3d_plugin.dll"
                Target = $pluginTarget
            }
        )
    }

    foreach ($entry in $installEntries) {
        if (-not (Test-Path -LiteralPath $entry.Source)) {
            throw "Build output is missing: $($entry.Source)"
        }
    }

    Write-Host ""
    Write-Host "==> Validate staged TensorRT engines" -ForegroundColor Cyan
    $stageEnginePaths = @(
        $installEntries | Where-Object {
            $_.Source.EndsWith(".engine", [System.StringComparison]::OrdinalIgnoreCase)
        } | ForEach-Object { $_.Source }
    )
    $stagedInspection = Invoke-EngineInspection `
        -Python $python `
        -Root $root `
        -EnginePaths $stageEnginePaths

    Assert-AvtrServicesStopped

    $allManagedTargets = @($installEntries | ForEach-Object { $_.Target })
    foreach ($optionalTarget in @($warpTarget, $pluginTarget)) {
        if ($allManagedTargets -notcontains $optionalTarget) {
            $allManagedTargets += $optionalTarget
        }
    }
    $retireTargets = @()
    if ($Mode -eq "Standard") {
        $retireTargets = @($warpTarget, $pluginTarget)
    }
    $manifestPath = Join-Path $artifacts "engine-manifest.json"
    $backupState = Backup-ActiveArtifacts `
        -ArtifactRoot $artifacts `
        -BackupRoot $backupRoot `
        -Targets $allManagedTargets `
        -ManifestPath $manifestPath

    Invoke-ArtifactInstallTransaction -BackupState $backupState -Action {
        Install-ActiveArtifacts `
            -InstallEntries $installEntries `
            -RetireTargets $retireTargets `
            -Timestamp $timestamp

        $activeInspection = Invoke-EngineInspection -Python $python -Root $root
        $manifest = [ordered]@{
            schemaVersion = 1
            status = "complete"
            createdAt = (Get-Date).ToUniversalTime().ToString("o")
            mode = $Mode.ToLowerInvariant()
            runtimeId = "avtr1-local-$timestamp"
            stagedInspection = $stagedInspection
            activeInspection = $activeInspection
        }
        $manifestTemporary = "$manifestPath.$timestamp.tmp"
        $manifest | ConvertTo-Json -Depth 12 | Set-Content `
            -LiteralPath $manifestTemporary `
            -Encoding UTF8
        Move-Item -LiteralPath $manifestTemporary -Destination $manifestPath -Force
    }

    Write-Host ""
    Write-Host "TensorRT $Mode acceleration is ready." -ForegroundColor Green
    Write-Host "Manifest: $manifestPath"
    if (Test-Path -LiteralPath $backupRoot) {
        Write-Host "Previous artifacts were preserved at: $backupRoot"
    }
}
finally {
    foreach ($name in $environmentNames) {
        $value = $previousEnvironment[$name]
        $environmentPath = "Env:$name"
        if ($null -eq $value) {
            Remove-Item -Path $environmentPath -ErrorAction SilentlyContinue
        }
        else {
            Set-Item -Path $environmentPath -Value $value
        }
    }
}
