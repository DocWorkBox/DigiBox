# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

[CmdletBinding()]
param(
    [ValidateSet("Standard", "Full")]
    [string]$Edition = "Standard",
    [ValidateSet("Auto", "Installer", "Archive", "Unpacked")]
    [string]$Target = "Auto",
    [string]$SourceRoot = "",
    [string]$RuntimeDestination = "",
    [ValidateSet("PortableV2", "LegacyV1")]
    [string]$RuntimeLayout = "PortableV2",
    [string]$NpmExecutable = "",
    [string]$TorchWheelhouse = "",
    [switch]$SkipRuntimeBuild,
    [switch]$CleanRuntime,
    [switch]$SkipRuntimeDependencies,
    [switch]$SkipModels,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$defaultRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$root = [System.IO.Path]::GetFullPath(
    $(if ($SourceRoot) { $SourceRoot } else { $defaultRoot })
).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$runtimeRoot = [System.IO.Path]::GetFullPath(
    $(if ($RuntimeDestination) {
        $RuntimeDestination
    } else {
        Join-Path $root "desktop\staging\avtr-runtime"
    })
).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$resolvedTorchWheelhouse = if ([string]::IsNullOrWhiteSpace($TorchWheelhouse)) {
    $null
} else {
    [System.IO.Path]::GetFullPath($TorchWheelhouse).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}
$configName = if ($Edition -eq "Full") {
    "electron-builder-full.yml"
} else {
    "electron-builder.yml"
}
$configPath = Join-Path $root $configName
$runtimeBuilder = Join-Path $root "scripts\desktop\build_portable_runtime.ps1"
$resolvedTarget = if ($Target -eq "Auto") {
    if ($Edition -eq "Full") { "Archive" } else { "Installer" }
} else {
    $Target
}

if ($Edition -eq "Full" -and $resolvedTarget -eq "Installer") {
    throw "The Full Runtime is too large for a reliable NSIS installer. Use -Target Archive or -Target Unpacked."
}
if ($Edition -eq "Standard" -and $resolvedTarget -eq "Archive") {
    throw "The Standard edition supports -Target Installer or -Target Unpacked."
}
if ($resolvedTorchWheelhouse -and $RuntimeLayout -ne "PortableV2") {
    throw "-TorchWheelhouse is supported only with -RuntimeLayout PortableV2."
}

function Assert-NoForbiddenRuntimePayload {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

    $forbidden = @()
    $managedDependencyPrefixes = @(
        ((Join-Path $RuntimeRoot "python-main").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python-cosyvoice").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python-feynobg").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "packages").ToLowerInvariant() + "\")
    )
    foreach ($entry in (Get-ChildItem -LiteralPath $RuntimeRoot -Recurse -Force)) {
        $name = $entry.Name.ToLowerInvariant()
        $full = $entry.FullName.ToLowerInvariant()
        $underManagedPython = $false
        foreach ($prefix in $managedDependencyPrefixes) {
            if ($full.StartsWith($prefix)) {
                $underManagedPython = $true
                break
            }
        }
        $trustedCaBundle = $underManagedPython -and (
            $full.EndsWith("\certifi\cacert.pem") -or
            $full.EndsWith("\pip\_vendor\certifi\cacert.pem")
        )
        if ($entry.PSIsContainer -and $name -in @(
            "user_assets", "local_voices", "voice_clones", "reference_audio",
            ".trash", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache",
            ".engine-staging", ".engine-backups"
        ) -or ($entry.PSIsContainer -and $full.EndsWith(
            "\artifacts\main\avatars_artifacts\backgrounds"
        )) -or ($entry.PSIsContainer -and (
            $full.EndsWith("\memory\backups") -or
            $full.EndsWith("\memory\pending-imports")
        ))) {
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
            ($name.StartsWith("digibox-memory") -and $entry.Extension -ieq ".json")
        )) {
            $forbidden += $entry.FullName
        }
    }
    if ($forbidden.Count -gt 0) {
        throw "Full desktop staged Runtime contains forbidden private/cache/engine payloads: $($forbidden -join ', ')"
    }
}

function Resolve-RuntimeManifestPath {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
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
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $RuntimeRoot $normalized))
    $rootPrefix = $RuntimeRoot.TrimEnd("\", "/") + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith(
        $rootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label escapes the Runtime root: $RelativePath"
    }
    return $resolved
}

function Resolve-NpmPath {
    if ($NpmExecutable) {
        $candidate = [System.IO.Path]::GetFullPath($NpmExecutable)
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "npm executable does not exist: $candidate"
        }
        return $candidate
    }
    $command = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $command = Get-Command npm -ErrorAction SilentlyContinue
    }
    if ($null -eq $command) {
        throw "Node.js/npm is required to build the desktop application."
    }
    return $command.Source
}

function Invoke-FrontendVendoring {
    param([Parameter(Mandatory = $true)][string]$NpmPath)

    Push-Location $root
    try {
        & $NpmPath "run" "vendor:frontend"
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend vendoring failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

function Assert-CompleteRuntimeComponents {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )

    $components = $Manifest.components
    if (
        $null -eq $components -or
        $components.dependenciesIncluded -ne $true -or
        $components.modelsIncluded -ne $true -or
        $components.frontendVendorIncluded -ne $true -or
        $components.tensorRtBuildInputsIncluded -ne $true
    ) {
        throw (
            "Full desktop Runtime is incomplete. The manifest must set " +
            "components.dependenciesIncluded, components.modelsIncluded, " +
            "components.frontendVendorIncluded, and components.tensorRtBuildInputsIncluded to true: " +
            $ManifestPath
        )
    }
}

function Assert-MemoryPackageEntrypoint {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

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
    $missing = @($requiredMemoryModules | Where-Object {
        $candidate = Join-Path $RuntimeRoot $_
        -not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
        (Get-Item -LiteralPath $candidate).Length -le 0
    })
    if ($missing.Count -gt 0) {
        throw "Full desktop Runtime memory package is incomplete; missing: $($missing -join ', ')"
    }
}

function Assert-RequiredRuntimePayload {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

    $requiredRelativePaths = @(
        "scripts\run_local_stream.py",
        "src\avaturn_live_streamer\__init__.py",
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
        "src\avaturn_live_streamer\memory\worklet.py",
        "src\avtr1_renderer\__init__.py",
        "src\avaturn_live_streamer\vendor\preact.module.js",
        "src\avaturn_live_streamer\vendor\preact-hooks.module.js",
        "src\avaturn_live_streamer\vendor\htm.module.js",
        "third_party\CosyVoice\cosyvoice\cli\cosyvoice.py",
        "third_party\CosyVoice\third_party\Matcha-TTS\matcha\__init__.py",
        "models\Fun-CosyVoice3-0.5B-2512\cosyvoice3.yaml",
        "models\Fun-CosyVoice3-0.5B-2512\llm.pt",
        "models\Fun-CosyVoice3-0.5B-2512\flow.pt",
        "models\Fun-CosyVoice3-0.5B-2512\hift.pt",
        "models\Fun-CosyVoice3-0.5B-2512\campplus.onnx",
        "models\Fun-CosyVoice3-0.5B-2512\speech_tokenizer_v3.onnx",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\config.json",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\generation_config.json",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\model.safetensors",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\tokenizer_config.json",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\merges.txt",
        "models\Fun-CosyVoice3-0.5B-2512\CosyVoice-BlankEN\vocab.json",
        "artifacts\main\avtr1_normalizer.safetensors",
        "artifacts\main\avatars_artifacts\pasteback_mask.png",
        "artifacts\main\renderer_runtime_artifacts\appearance_extractor.onnx",
        "artifacts\main\renderer_runtime_artifacts\motion_extractor.onnx",
        "artifacts\main\renderer_runtime_artifacts\insightface_det.onnx",
        "artifacts\main\renderer_runtime_artifacts\landmark106.onnx",
        "artifacts\main\renderer_runtime_artifacts\landmark203.onnx",
        "artifacts\main\renderer_runtime_artifacts\blaze_face.onnx",
        "artifacts\main\renderer_runtime_artifacts\face_mesh.onnx",
        "artifacts\main\build_artifacts\avtr1.scripted.pt",
        "artifacts\main\build_artifacts\hubert-lbs-avtr1.onnx",
        "artifacts\main\build_artifacts\decoder.onnx",
        "artifacts\main\build_artifacts\modnet.onnx",
        "artifacts\main\build_artifacts\stitch_network.onnx",
        "artifacts\main\build_artifacts\warp_network.onnx",
        "artifacts\main\build_artifacts\warp_network_ori.onnx"
    )
    $missing = @($requiredRelativePaths | Where-Object {
        $candidate = Join-Path $RuntimeRoot $_
        -not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
        (Get-Item -LiteralPath $candidate).Length -le 0
    })
    if ($missing.Count -gt 0) {
        throw "Full desktop Runtime payload is incomplete; missing: $($missing -join ', ')"
    }
}

function Assert-RuntimeDependencyImports {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest
    )

    $isPortableV2 = $Manifest.schemaVersion -eq 2 -and $Manifest.layout -eq "portable-v2"
    if ($isPortableV2) {
        $python = Resolve-RuntimeManifestPath `
            -RuntimeRoot $RuntimeRoot `
            -RelativePath ([string]$Manifest.paths.python) `
            -Label "paths.python"
        $mainPaths = @($Manifest.python.packageLayers.main | ForEach-Object {
            Resolve-RuntimeManifestPath `
                -RuntimeRoot $RuntimeRoot `
                -RelativePath ([string]$_) `
                -Label "python.packageLayers.main"
        })
        $cosyVoicePaths = @($Manifest.python.packageLayers.cosyvoice | ForEach-Object {
            Resolve-RuntimeManifestPath `
                -RuntimeRoot $RuntimeRoot `
                -RelativePath ([string]$_) `
                -Label "python.packageLayers.cosyvoice"
        })
        $feyNoBgPaths = @($Manifest.python.packageLayers.feynobg | ForEach-Object {
            Resolve-RuntimeManifestPath `
                -RuntimeRoot $RuntimeRoot `
                -RelativePath ([string]$_) `
                -Label "python.packageLayers.feynobg"
        })
        $probes = @(
            [ordered]@{
                name = "main"
                python = $python
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,tensorrt,nvvfx,fastapi,uvicorn,aiortc,cv2,avtr1_renderer,avaturn_live_streamer; import avaturn_live_streamer.local_stream_cli; import avaturn_live_streamer.memory.admin; import avaturn_live_streamer.memory.api"
                paths = $mainPaths
            },
            [ordered]@{
                name = "cosyvoice"
                python = $python
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,torchaudio,onnxruntime,transformers,matcha; from cosyvoice.cli.cosyvoice import CosyVoice3; import avaturn_live_streamer.integrations.cosyvoice_server"
                paths = $cosyVoicePaths
            },
            [ordered]@{
                name = "feynobg"
                python = $python
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,torchvision,nobg,avaturn_live_streamer.integrations.feynobg_server"
                paths = $feyNoBgPaths
            }
        )
    }
    else {
        $probes = @(
            [ordered]@{
                name = "python-main"
                python = Join-Path $RuntimeRoot "python-main\python.exe"
                code = "import torch,tensorrt,nvvfx,fastapi,uvicorn,aiortc,cv2,avtr1_renderer,avaturn_live_streamer; import avaturn_live_streamer.local_stream_cli; import avaturn_live_streamer.memory.admin; import avaturn_live_streamer.memory.api"
                paths = @((Join-Path $RuntimeRoot "src"))
            },
            [ordered]@{
                name = "python-cosyvoice"
                python = Join-Path $RuntimeRoot "python-cosyvoice\python.exe"
                code = "import torch,torchaudio,onnxruntime,transformers,matcha; from cosyvoice.cli.cosyvoice import CosyVoice3; import avaturn_live_streamer.integrations.cosyvoice_server"
                paths = @(
                    (Join-Path $RuntimeRoot "src"),
                    (Join-Path $RuntimeRoot "third_party\CosyVoice"),
                    (Join-Path $RuntimeRoot "third_party\CosyVoice\third_party\Matcha-TTS")
                )
            },
            [ordered]@{
                name = "python-feynobg"
                python = Join-Path $RuntimeRoot "python-feynobg\python.exe"
                code = "import torch,torchvision,nobg,avaturn_live_streamer.integrations.feynobg_server"
                paths = @((Join-Path $RuntimeRoot "src"))
            }
        )
    }
    foreach ($probe in $probes) {
        $previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
        $previousNoUserSite = [Environment]::GetEnvironmentVariable("PYTHONNOUSERSITE", "Process")
        $previousDontWriteBytecode = [Environment]::GetEnvironmentVariable(
            "PYTHONDONTWRITEBYTECODE",
            "Process"
        )
        try {
            [Environment]::SetEnvironmentVariable("PYTHONPATH", [string]::Join([System.IO.Path]::PathSeparator, @($probe.paths)), "Process")
            [Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", "1", "Process")
            [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", "Process")
            & $probe.python -B -c $probe.code
            if ($LASTEXITCODE -ne 0) {
                throw "Full desktop Runtime dependency import probe failed for $($probe.name)."
            }
        }
        finally {
            [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath, "Process")
            [Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", $previousNoUserSite, "Process")
            [Environment]::SetEnvironmentVariable(
                "PYTHONDONTWRITEBYTECODE",
                $previousDontWriteBytecode,
                "Process"
            )
        }
    }
}

if (
    -not $PlanOnly -and
    $Edition -eq "Full" -and
    -not $SkipRuntimeBuild -and
    $resolvedTorchWheelhouse -and
    -not (Test-Path -LiteralPath $resolvedTorchWheelhouse -PathType Container)
) {
    throw "Torch wheelhouse must be an existing directory: $resolvedTorchWheelhouse"
}

if (-not (Test-Path -LiteralPath (Join-Path $root "package.json") -PathType Leaf)) {
    throw "package.json is missing from the desktop source root: $root"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "electron-builder configuration is missing: $configPath"
}

$plan = [ordered]@{
    schemaVersion = 1
    kind = "avtr1-desktop-build-plan"
    edition = $Edition.ToLowerInvariant()
    target = $resolvedTarget.ToLowerInvariant()
    config = $configName
    includesRuntime = $Edition -eq "Full"
    runtimeRoot = $runtimeRoot
    buildsRuntime = ($Edition -eq "Full" -and -not $SkipRuntimeBuild)
    runtimeBuildLayout = $RuntimeLayout
    acceptedRuntimeLayouts = @("portable-v1", "portable-v2")
    cleanRuntime = [bool]$CleanRuntime
    includesModels = ($Edition -eq "Full" -and -not [bool]$SkipModels)
    torchWheelhouse = $resolvedTorchWheelhouse
    packagedRuntimeSource = $(if ($Edition -eq "Full") { $runtimeRoot } else { $null })
}
if ($PlanOnly) {
    Write-Output ($plan | ConvertTo-Json -Depth 6 -Compress)
    return
}

if ($Edition -eq "Full" -and -not $SkipRuntimeBuild -and ($SkipRuntimeDependencies -or $SkipModels)) {
    throw "A Full desktop distribution cannot skip Runtime dependencies or models. Use the Runtime builder directly for incomplete diagnostic layouts."
}
$npmPath = Resolve-NpmPath
if ($Edition -eq "Standard" -or -not $SkipRuntimeBuild) {
    Invoke-FrontendVendoring -NpmPath $npmPath
}

if ($Edition -eq "Full") {
    if (-not $SkipRuntimeBuild) {
        if (-not (Test-Path -LiteralPath $runtimeBuilder -PathType Leaf)) {
            throw "Portable Runtime builder is missing: $runtimeBuilder"
        }
        $runtimeArguments = @{
            SourceRoot = $root
            Destination = $runtimeRoot
            Layout = $RuntimeLayout
        }
        if ($CleanRuntime) { $runtimeArguments["Clean"] = $true }
        if ($SkipRuntimeDependencies) { $runtimeArguments["SkipDependencies"] = $true }
        if ($SkipModels) { $runtimeArguments["SkipModels"] = $true }
        if ($resolvedTorchWheelhouse) {
            $runtimeArguments["TorchWheelhouse"] = $resolvedTorchWheelhouse
        }
        & $runtimeBuilder @runtimeArguments
        if (-not $?) {
            throw "Portable Runtime build failed."
        }
    }

    $runtimeManifest = Join-Path $runtimeRoot "runtime-manifest.json"
    $tensorRtAssistant = Join-Path $runtimeRoot "scripts\desktop\build_tensorrt.ps1"
    $tensorRtLauncher = Join-Path $runtimeRoot "scripts\desktop\DigiBox-TensorRT-Setup.cmd"
    $runtimeInspector = Join-Path $runtimeRoot "scripts\desktop\inspect_runtime.py"
    $missingRuntimeFiles = @(
        @(
            $runtimeManifest,
            $tensorRtAssistant,
            $tensorRtLauncher,
            $runtimeInspector
        ) | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    )
    if ($missingRuntimeFiles.Count -gt 0) {
        throw "Full desktop build requires a complete staged Runtime; missing: $($missingRuntimeFiles -join ', ')"
    }

    $manifest = Get-Content -LiteralPath $runtimeManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    $isPortableV1 = $manifest.schemaVersion -eq 1 -and $manifest.layout -eq "portable-v1"
    $isPortableV2 = $manifest.schemaVersion -eq 2 -and $manifest.layout -eq "portable-v2"
    if (-not $isPortableV1 -and -not $isPortableV2) {
        throw (
            "Unsupported portable Runtime manifest; expected portable-v1 or portable-v2: " +
            $runtimeManifest
        )
    }

    $layoutFiles = @()
    $layoutDirectories = @()
    if ($isPortableV2) {
        if ($null -eq $manifest.paths -or $null -eq $manifest.python) {
            throw "Portable-v2 Runtime manifest is missing paths or python metadata: $runtimeManifest"
        }
        $normalizedPythonPath = ([string]$manifest.paths.python).Replace("\", "/")
        if ($normalizedPythonPath -ne "python/python.exe") {
            throw "Portable-v2 Full Runtime requires paths.python to be python/python.exe."
        }
        $layoutFiles += Resolve-RuntimeManifestPath `
            -RuntimeRoot $runtimeRoot `
            -RelativePath ([string]$manifest.paths.python) `
            -Label "paths.python"

        $requiredLayerDirectories = @(
            "packages/main",
            "packages/cosyvoice",
            "packages/feynobg",
            "packages/shared"
        )
        foreach ($relativeLayerDirectory in $requiredLayerDirectories) {
            $layoutDirectories += Resolve-RuntimeManifestPath `
                -RuntimeRoot $runtimeRoot `
                -RelativePath $relativeLayerDirectory `
                -Label "required package layer"
        }
        foreach ($profile in @("main", "cosyvoice", "feynobg")) {
            $profileLayers = @($manifest.python.packageLayers.$profile)
            if ($profileLayers.Count -eq 0) {
                throw "Portable-v2 Runtime manifest requires python.packageLayers.$profile."
            }
            foreach ($relativeLayer in $profileLayers) {
                $layoutDirectories += Resolve-RuntimeManifestPath `
                    -RuntimeRoot $runtimeRoot `
                    -RelativePath ([string]$relativeLayer) `
                    -Label "python.packageLayers.$profile"
            }
        }
    }
    else {
        $layoutFiles += @(
            (Join-Path $runtimeRoot "python-main\python.exe"),
            (Join-Path $runtimeRoot "python-cosyvoice\python.exe"),
            (Join-Path $runtimeRoot "python-feynobg\python.exe")
        )
    }

    $missingLayoutFiles = @($layoutFiles | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    })
    $missingLayoutDirectories = @($layoutDirectories | Select-Object -Unique | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Container)
    })
    if ($missingLayoutFiles.Count -gt 0 -or $missingLayoutDirectories.Count -gt 0) {
        $missingLayoutPayload = @($missingLayoutFiles) + @($missingLayoutDirectories)
        throw "Full desktop Runtime layout is incomplete; missing: $($missingLayoutPayload -join ', ')"
    }
    Assert-CompleteRuntimeComponents -Manifest $manifest -ManifestPath $runtimeManifest
    Assert-NoForbiddenRuntimePayload -RuntimeRoot $runtimeRoot
    Assert-RequiredRuntimePayload -RuntimeRoot $runtimeRoot
    Assert-MemoryPackageEntrypoint -RuntimeRoot $runtimeRoot
    $versionedHelperPaths = @(
        "scripts\desktop\build_tensorrt.ps1",
        "scripts\desktop\DigiBox-TensorRT-Setup.cmd",
        "scripts\desktop\inspect_runtime.py"
    )
    foreach ($relativeHelperPath in $versionedHelperPaths) {
        $sourceHelper = Join-Path $root $relativeHelperPath
        $stagedHelper = Join-Path $runtimeRoot $relativeHelperPath
        $sourceHash = (Get-FileHash -LiteralPath $sourceHelper -Algorithm SHA256).Hash
        $stagedHash = (Get-FileHash -LiteralPath $stagedHelper -Algorithm SHA256).Hash
        if ($sourceHash -ne $stagedHash) {
            throw "Staged Runtime helper differs from this desktop source; rebuild the Runtime: $relativeHelperPath"
        }
        if ([System.IO.Path]::GetExtension($stagedHelper) -ieq ".ps1") {
            $parseErrors = $null
            [System.Management.Automation.Language.Parser]::ParseFile(
                $stagedHelper,
                [ref]$null,
                [ref]$parseErrors
            ) | Out-Null
            if ($parseErrors.Count -gt 0) {
                throw "Staged Runtime PowerShell helper does not parse in Windows PowerShell 5.1: $stagedHelper"
            }
        }
    }
    Assert-RuntimeDependencyImports -RuntimeRoot $runtimeRoot -Manifest $manifest
}

$electronBuilderPath = Join-Path $root "node_modules\.bin\electron-builder.cmd"
if (-not (Test-Path -LiteralPath $electronBuilderPath -PathType Leaf)) {
    throw "The pinned electron-builder is not installed. Run 'npm install' in $root before building."
}

$builderArguments = @(
    "--config", $configPath,
    "--win",
    "--x64",
    "--publish", "never"
)
if ($resolvedTarget -eq "Unpacked") {
    $builderArguments += "--dir"
}

$previousRuntimeSource = $env:AVTR_PORTABLE_RUNTIME_SOURCE
Push-Location $root
try {
    if ($Edition -eq "Full") {
        $env:AVTR_PORTABLE_RUNTIME_SOURCE = $runtimeRoot
    }
    & $electronBuilderPath @builderArguments
    if ($LASTEXITCODE -ne 0) {
        throw "electron-builder failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:AVTR_PORTABLE_RUNTIME_SOURCE = $previousRuntimeSource
    Pop-Location
}

$label = if ($Edition -eq "Full") { "Full desktop edition" } else { "Standard desktop edition" }
Write-Host "$label build completed." -ForegroundColor Green
