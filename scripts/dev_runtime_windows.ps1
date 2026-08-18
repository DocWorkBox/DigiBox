# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

<#
.SYNOPSIS
Resolve the checked-out AVTR-1 sources against a DigiBox portable-v2 Runtime.

.DESCRIPTION
This helper is intended to be dot-sourced by the Windows development entrypoints.
Repository source overrides the packaged copy while dependencies, models, and
artifacts come from the selected Full Runtime. Model repair and TensorRT build
commands may therefore update that extracted Runtime.
#>

$script:DigiBoxDevRuntimeScriptRoot = $PSScriptRoot

function Resolve-DigiBoxDevContainedPath {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (
        [string]::IsNullOrWhiteSpace($RelativePath) -or
        [System.IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath.Contains(":") -or
        $RelativePath.Contains([char]0)
    ) {
        throw "$Description must be a non-empty relative portable-v2 path."
    }

    $root = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\", "/")
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    $prefix = $root + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description escapes the portable-v2 Runtime root: $RelativePath"
    }
    return $resolved
}

function Resolve-DigiBoxMemoryRoot {
    $localAppData = [string]$env:LOCALAPPDATA
    if (
        [string]::IsNullOrWhiteSpace($localAppData) -or
        -not [System.IO.Path]::IsPathRooted($localAppData) -or
        -not $localAppData.Contains(":")
    ) {
        return $null
    }
    return [System.IO.Path]::GetFullPath(
        (Join-Path $localAppData "DigiBox\memory")
    )
}

function ConvertTo-DigiBoxDevRuntimeRoot {
    param([Parameter(Mandatory = $true)][string]$Candidate)

    $root = [System.IO.Path]::GetFullPath($Candidate)
    $nested = Join-Path $root "avtr-runtime"
    if (Test-Path -LiteralPath (Join-Path $nested "runtime-manifest.json") -PathType Leaf) {
        return [System.IO.Path]::GetFullPath($nested)
    }
    return $root
}

function Assert-DigiBoxDevelopmentPortsAvailable {
    [CmdletBinding()]
    param([int[]]$Ports = @(7860, 8000, 8767, 8768))

    $requested = @($Ports | Sort-Object -Unique)
    $properties = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
    $listeners = @($properties.GetActiveTcpListeners())
    $occupied = @($requested | Where-Object {
        $port = $_
        $listeners | Where-Object { $_.Port -eq $port } | Select-Object -First 1
    })
    if ($occupied.Count -gt 0) {
        throw (
            "DigiBox source development ports are already in use: " +
            ($occupied -join ", ") +
            ". Stop the existing service before starting repository source."
        )
    }
}

function Find-DigiBoxDevRuntimeRoot {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$RuntimeRoot = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($RuntimeRoot)) {
        return ConvertTo-DigiBoxDevRuntimeRoot -Candidate $RuntimeRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($env:AVTR1_DEV_RUNTIME_ROOT)) {
        return ConvertTo-DigiBoxDevRuntimeRoot -Candidate $env:AVTR1_DEV_RUNTIME_ROOT
    }

    $candidates = @(
        (Join-Path $RepoRoot "desktop\dist-tauri\DigiBox-Full-win64\avtr-runtime")
    )
    $buildsRoot = Join-Path $RepoRoot "desktop\builds"
    if (Test-Path -LiteralPath $buildsRoot -PathType Container) {
        $builds = @(Get-ChildItem -LiteralPath $buildsRoot -Directory -ErrorAction Stop |
            Sort-Object -Property LastWriteTimeUtc -Descending)
        foreach ($build in $builds) {
            $candidates += Join-Path $build.FullName "dist\DigiBox-Full-win64\avtr-runtime"
        }
    }
    $candidates += Join-Path $RepoRoot "desktop\staging\avtr-runtime"
    if (Test-Path -LiteralPath $buildsRoot -PathType Container) {
        foreach ($build in $builds) {
            $candidates += Join-Path $build.FullName "staging\avtr-runtime"
        }
    }

    $seen = @{}
    foreach ($candidate in $candidates) {
        $normalized = [System.IO.Path]::GetFullPath($candidate)
        $key = $normalized.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        if (Test-Path -LiteralPath (Join-Path $normalized "runtime-manifest.json") -PathType Leaf) {
            return $normalized
        }
    }

    throw (
        "No DigiBox Full portable-v2 Runtime was found. Pass -RuntimeRoot or set " +
        "AVTR1_DEV_RUNTIME_ROOT."
    )
}

function Get-DigiBoxDevManifestLayers {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)]
        [ValidateSet("main", "cosyvoice", "feynobg")]
        [string]$Profile
    )

    $packageLayersProperty = $Manifest.python.PSObject.Properties["packageLayers"]
    if ($null -eq $packageLayersProperty) {
        throw "portable-v2 manifest is missing python.packageLayers."
    }
    $profileProperty = $packageLayersProperty.Value.PSObject.Properties[$Profile]
    if ($null -eq $profileProperty) {
        throw "portable-v2 manifest is missing python.packageLayers.$Profile."
    }
    $relativeLayers = @($profileProperty.Value)
    if ($relativeLayers.Count -eq 0) {
        throw "portable-v2 manifest has no python.packageLayers.$Profile entries."
    }

    $resolvedLayers = @()
    for ($index = 0; $index -lt $relativeLayers.Count; $index += 1) {
        $layer = Resolve-DigiBoxDevContainedPath `
            -RuntimeRoot $RuntimeRoot `
            -RelativePath ([string]$relativeLayers[$index]) `
            -Description "python.packageLayers.$Profile[$index]"
        $resolvedLayers += $layer
    }
    return $resolvedLayers
}

function Get-DigiBoxDevProfilePath {
    param(
        [Parameter(Mandatory = $true)][string[]]$PreferredPaths,
        [Parameter(Mandatory = $true)][string[]]$RuntimePaths,
        [string[]]$ExcludedRuntimePaths = @()
    )

    $excluded = @{}
    foreach ($path in $ExcludedRuntimePaths) {
        $excluded[[System.IO.Path]::GetFullPath($path).ToLowerInvariant()] = $true
    }
    $seen = @{}
    $resolved = @()
    foreach ($path in @($PreferredPaths) + @($RuntimePaths)) {
        $fullPath = [System.IO.Path]::GetFullPath($path)
        $key = $fullPath.ToLowerInvariant()
        if ($excluded.ContainsKey($key) -or $seen.ContainsKey($key)) {
            continue
        }
        if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
            throw "Development Python path is missing: $fullPath"
        }
        $seen[$key] = $true
        $resolved += $fullPath
    }
    return [string]::Join([System.IO.Path]::PathSeparator, $resolved)
}

function Resolve-DigiBoxSourceRuntime {
    [CmdletBinding()]
    param(
        [string]$RepoRoot = "",
        [string]$RuntimeRoot = ""
    )

    if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
        $RepoRoot = Split-Path -Parent $script:DigiBoxDevRuntimeScriptRoot
    }
    $resolvedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
    $repoSource = Join-Path $resolvedRepoRoot "src"
    $repoCosyVoice = Join-Path $resolvedRepoRoot "third_party\CosyVoice"
    $repoMatcha = Join-Path $repoCosyVoice "third_party\Matcha-TTS"
    foreach ($requiredRepoDirectory in @($repoSource, $repoCosyVoice, $repoMatcha)) {
        if (-not (Test-Path -LiteralPath $requiredRepoDirectory -PathType Container)) {
            throw "Repository development source is missing: $requiredRepoDirectory"
        }
    }

    $resolvedRuntimeRoot = Find-DigiBoxDevRuntimeRoot `
        -RepoRoot $resolvedRepoRoot `
        -RuntimeRoot $RuntimeRoot
    $manifestPath = Join-Path $resolvedRuntimeRoot "runtime-manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "DigiBox portable-v2 manifest is missing: $manifestPath"
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 -ErrorAction Stop |
            ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "DigiBox portable-v2 manifest is invalid JSON: $manifestPath"
    }
    if ($manifest.schemaVersion -ne 2 -or $manifest.layout -ne "portable-v2") {
        throw "DigiBox source development requires a schemaVersion 2 portable-v2 Runtime."
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.runtimeId)) {
        throw "portable-v2 manifest is missing runtimeId."
    }
    if (
        [string]::IsNullOrWhiteSpace([string]$manifest.python.version) -or
        $null -eq $manifest.python.packageLayers
    ) {
        throw "portable-v2 manifest has incomplete Python metadata."
    }
    foreach ($component in @(
        "dependenciesIncluded",
        "modelsIncluded",
        "frontendVendorIncluded",
        "tensorRtBuildInputsIncluded"
    )) {
        $property = $manifest.components.PSObject.Properties[$component]
        if ($null -eq $property -or $property.Value -ne $true) {
            throw "portable-v2 manifest requires components.$component=true."
        }
    }

    foreach ($pathProperty in $manifest.paths.PSObject.Properties) {
        $null = Resolve-DigiBoxDevContainedPath `
            -RuntimeRoot $resolvedRuntimeRoot `
            -RelativePath ([string]$pathProperty.Value) `
            -Description "paths.$($pathProperty.Name)"
    }
    $python = Resolve-DigiBoxDevContainedPath `
        -RuntimeRoot $resolvedRuntimeRoot `
        -RelativePath ([string]$manifest.paths.python) `
        -Description "paths.python"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "DigiBox portable-v2 Python is missing: $python"
    }
    $modelsRoot = Resolve-DigiBoxDevContainedPath `
        -RuntimeRoot $resolvedRuntimeRoot `
        -RelativePath ([string]$manifest.paths.models) `
        -Description "paths.models"
    if (-not (Test-Path -LiteralPath $modelsRoot -PathType Container)) {
        throw "DigiBox portable-v2 models are missing: $modelsRoot"
    }
    $artifactsDir = Resolve-DigiBoxDevContainedPath `
        -RuntimeRoot $resolvedRuntimeRoot `
        -RelativePath ([string]$manifest.paths.artifacts) `
        -Description "paths.artifacts"
    if (-not (Test-Path -LiteralPath $artifactsDir -PathType Container)) {
        throw "DigiBox portable-v2 artifacts are missing: $artifactsDir"
    }
    $artifactsRoot = Split-Path -Parent $artifactsDir

    $mainLayers = @(Get-DigiBoxDevManifestLayers `
        -RuntimeRoot $resolvedRuntimeRoot -Manifest $manifest -Profile "main")
    $cosyVoiceLayers = @(Get-DigiBoxDevManifestLayers `
        -RuntimeRoot $resolvedRuntimeRoot -Manifest $manifest -Profile "cosyvoice")
    $feyNoBgLayers = @(Get-DigiBoxDevManifestLayers `
        -RuntimeRoot $resolvedRuntimeRoot -Manifest $manifest -Profile "feynobg")
    $runtimeSource = Resolve-DigiBoxDevContainedPath `
        -RuntimeRoot $resolvedRuntimeRoot `
        -RelativePath ([string]$manifest.paths.source) `
        -Description "paths.source"
    $runtimeCosyVoice = Join-Path $resolvedRuntimeRoot "third_party\CosyVoice"
    $runtimeMatcha = Join-Path $runtimeCosyVoice "third_party\Matcha-TTS"

    $mainPythonPath = Get-DigiBoxDevProfilePath `
        -PreferredPaths @($repoSource) `
        -RuntimePaths $mainLayers `
        -ExcludedRuntimePaths @($runtimeSource)
    $cosyVoicePythonPath = Get-DigiBoxDevProfilePath `
        -PreferredPaths @($repoSource, $repoCosyVoice, $repoMatcha) `
        -RuntimePaths $cosyVoiceLayers `
        -ExcludedRuntimePaths @($runtimeSource, $runtimeCosyVoice, $runtimeMatcha)
    $feyNoBgPythonPath = Get-DigiBoxDevProfilePath `
        -PreferredPaths @($repoSource) `
        -RuntimePaths $feyNoBgLayers `
        -ExcludedRuntimePaths @($runtimeSource)

    $localAppData = [Environment]::GetFolderPath(
        [System.Environment+SpecialFolder]::LocalApplicationData
    )
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        $localAppData = Join-Path $resolvedRepoRoot ".digibox-dev"
    }
    $desktopData = Join-Path $localAppData "live.avaturn.avtr1.desktop"
    $memoryRoot = Resolve-DigiBoxMemoryRoot
    $pythonPath = [ordered]@{
        Main = $mainPythonPath
        CosyVoice = $cosyVoicePythonPath
        FeyNoBg = $feyNoBgPythonPath
    }
    $environment = [ordered]@{
        AVTR1_RUNTIME_ROOT = $resolvedRuntimeRoot
        AVTR1_APP_ROOT = $resolvedRepoRoot
        AVTR1_MODELS_ROOT = $modelsRoot
        AVTR1_LOCAL_STORAGE = $artifactsRoot
        AVTR1_USER_ASSETS_ROOT = (Join-Path $desktopData "user_assets")
        AVTR1_COSYVOICE_SPEAKER_CACHE = (Join-Path $desktopData "cosyvoice\spk2info.pt")
        AVTR1_SINGLE_ENV = "1"
        AVTR1_COSYVOICE_PYTHON = $python
        AVTR1_FEYNOBG_PYTHON = $python
        AVTR1_MAIN_PYTHONPATH = $mainPythonPath
        AVTR1_COSYVOICE_PYTHONPATH = $cosyVoicePythonPath
        AVTR1_FEYNOBG_PYTHONPATH = $feyNoBgPythonPath
        PYTHONPATH = $mainPythonPath
        PYTHONNOUSERSITE = "1"
        PYTHONDONTWRITEBYTECODE = "1"
        PYTHONUNBUFFERED = "1"
        PYTHONUTF8 = "1"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$memoryRoot)) {
        $environment["AVTR1_MEMORY_ROOT"] = $memoryRoot
    }

    return [pscustomobject]@{
        RepoRoot = $resolvedRepoRoot
        RuntimeRoot = $resolvedRuntimeRoot
        ManifestPath = $manifestPath
        Python = $python
        ModelsRoot = $modelsRoot
        ArtifactsRoot = $artifactsRoot
        PythonPath = [pscustomobject]$pythonPath
        Environment = $environment
    }
}

function Set-DigiBoxSourceRuntimeEnvironment {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Runtime)

    $environment = $Runtime.Environment
    if ($null -eq $environment) {
        throw "Runtime.Environment is required."
    }
    $managedKeys = @(
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CONDA_PREFIX_1",
        "CONDA_PROMPT_MODIFIER",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "AVTR1_MEMORY_ROOT",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT"
    )
    $values = [ordered]@{}
    if ($environment -is [System.Collections.IDictionary]) {
        foreach ($key in $environment.Keys) {
            $values[[string]$key] = [string]$environment[$key]
        }
    }
    else {
        foreach ($property in $environment.PSObject.Properties) {
            $values[$property.Name] = [string]$property.Value
        }
    }

    $keys = @($managedKeys) + @($values.Keys)
    $seen = @{}
    $snapshot = [ordered]@{}
    foreach ($key in $keys) {
        $normalized = ([string]$key).ToUpperInvariant()
        if ($seen.ContainsKey($normalized)) {
            continue
        }
        $seen[$normalized] = $true
        $previous = [Environment]::GetEnvironmentVariable([string]$key, "Process")
        $snapshot[[string]$key] = [pscustomobject]@{
            WasSet = ($null -ne $previous)
            Value = $previous
        }
    }

    foreach ($key in $managedKeys) {
        [Environment]::SetEnvironmentVariable($key, $null, "Process")
    }
    foreach ($key in $values.Keys) {
        [Environment]::SetEnvironmentVariable([string]$key, [string]$values[$key], "Process")
    }
    return $snapshot
}

function Restore-DigiBoxSourceRuntimeEnvironment {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Snapshot)

    if ($Snapshot -is [System.Collections.IDictionary]) {
        $keys = @($Snapshot.Keys)
    }
    else {
        $keys = @($Snapshot.PSObject.Properties.Name)
    }
    foreach ($key in $keys) {
        $state = if ($Snapshot -is [System.Collections.IDictionary]) {
            $Snapshot[$key]
        }
        else {
            $Snapshot.PSObject.Properties[[string]$key].Value
        }
        $value = if ($state.WasSet) { [string]$state.Value } else { $null }
        [Environment]::SetEnvironmentVariable([string]$key, $value, "Process")
    }
}
