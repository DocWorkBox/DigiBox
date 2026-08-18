# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

[CmdletBinding()]
param(
    [ValidateSet("Standard", "Full")]
    [string]$Edition = "Standard",
    [ValidateSet("Auto", "Installer", "Msi", "Archive", "Unpacked")]
    [string]$Target = "Auto",
    [string]$SourceRoot = "",
    [string]$RuntimeDestination = "",
    [string]$OutputDirectory = "",
    [string]$TauriExecutable = "",
    [string]$WebView2OfflineInstaller = "",
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
).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$runtimeRoot = [System.IO.Path]::GetFullPath(
    $(if ($RuntimeDestination) {
        $RuntimeDestination
    } else {
        Join-Path $root "desktop\staging\avtr-runtime"
    })
).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$deliveryRoot = [System.IO.Path]::GetFullPath(
    $(if ($OutputDirectory) {
        $OutputDirectory
    } else {
        Join-Path $root "desktop\dist-tauri"
    })
).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$resolvedTorchWheelhouse = if ([string]::IsNullOrWhiteSpace($TorchWheelhouse)) {
    $null
} else {
    [System.IO.Path]::GetFullPath($TorchWheelhouse).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

$runtimeBuilderRelative = "scripts\desktop\build_portable_runtime.ps1"
$runtimeBuilder = Join-Path $root $runtimeBuilderRelative
$rootTensorRtLauncherName = "DigiBox-TensorRT-Setup.cmd"
$rootTensorRtLauncherSource = Join-Path $root "scripts\desktop\DigiBox-TensorRT-Setup-Root.cmd"
$baseConfigRelative = "src-tauri/tauri.conf.json"
$fullConfigRelative = "src-tauri/tauri.full.conf.json"
$standardOfflineBuildRelative = "src-tauri/target/digibox-nsis-offline"
$standardOfflineConfigRelative = "$standardOfflineBuildRelative/tauri.offline.conf.json"
$standardOfflineHookRelative = "$standardOfflineBuildRelative/webview2-offline-installer.nsh"
$baseConfigPath = Join-Path $root ($baseConfigRelative.Replace("/", "\"))
$fullConfigPath = Join-Path $root ($fullConfigRelative.Replace("/", "\"))
$standardOfflineConfigPath = Join-Path $root ($standardOfflineConfigRelative.Replace("/", "\"))
$standardOfflineHookPath = Join-Path $root ($standardOfflineHookRelative.Replace("/", "\"))
$resolvedTarget = if ($Target -eq "Auto") {
    if ($Edition -eq "Full") { "Archive" } else { "Installer" }
} else {
    $Target
}
$configRelative = if ($Edition -eq "Full") {
    $fullConfigRelative
} elseif ($resolvedTarget -eq "Installer") {
    $standardOfflineConfigRelative
} else {
    $baseConfigRelative
}

if ($Edition -eq "Full" -and $resolvedTarget -in @("Installer", "Msi")) {
    throw "The Full Runtime cannot be emitted as a single-file NSIS/MSI installer. Use -Target Archive or Unpacked."
}
if ($Edition -eq "Full" -and $resolvedTarget -notin @("Archive", "Unpacked")) {
    throw "The Full edition supports only -Target Archive or -Target Unpacked."
}
if ($Edition -eq "Standard" -and $resolvedTarget -notin @("Installer", "Unpacked")) {
    throw "The Standard edition supports an NSIS Installer or Unpacked shell."
}

$requiredLicenses = @(
    "LICENSE.md",
    "LICENSE-MODEL.md",
    "LICENSE-RENDERER.md",
    "LICENSE-STREAMER.md",
    "PATENTS.md",
    "THIRD-PARTY-NOTICES.md"
)
$versionedHelpers = @(
    "scripts\desktop\build_tensorrt.ps1",
    "scripts\desktop\DigiBox-TensorRT-Setup.cmd",
    "scripts\desktop\inspect_runtime.py"
)
$requiredComponents = @(
    "dependenciesIncluded",
    "modelsIncluded",
    "frontendVendorIncluded",
    "tensorRtBuildInputsIncluded"
)
$forbiddenPayloads = @(
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
    "artifacts/main/engine-manifest.json",
    "memory.sqlite3",
    "memory.sqlite3-wal",
    "memory.sqlite3-shm",
    "memory/backups",
    "memory/pending-imports",
    "digibox-memory*.json"
)
$fullUnpackedRoot = Join-Path $deliveryRoot "DigiBox-Full-win64"
$fullArchivePath = Join-Path $deliveryRoot "DigiBox-Full-win64.zip"
$webView2InstallerName = "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
$webView2InstallerSize = 209653456
$webView2InstallerSha256 = "F8D4AB074C22A0CD136434F37C6B34DFB64EBF8A32CE42E03BD8F2A6B51A3892"
$webView2SignerSubject = "CN=Microsoft Corporation"
$tauriArguments = @("build", "--config", $configRelative)
if ($Edition -eq "Standard" -and $resolvedTarget -eq "Installer") {
    $tauriArguments += @("--bundles", "nsis")
} else {
    $tauriArguments += "--no-bundle"
}
$bundleMode = if ($Edition -eq "Standard" -and $resolvedTarget -eq "Installer") {
    "nsis"
} elseif ($Edition -eq "Full" -and $resolvedTarget -eq "Archive") {
    "zip64"
} else {
    "unpacked"
}

$plan = [ordered]@{
    schemaVersion = 1
    kind = "avtr1-tauri-build-plan"
    edition = $Edition.ToLowerInvariant()
    target = $resolvedTarget.ToLowerInvariant()
    bundleMode = $bundleMode
    config = $configRelative
    includesRuntime = $Edition -eq "Full"
    buildsRuntime = ($Edition -eq "Full" -and -not [bool]$SkipRuntimeBuild)
    runtimeRoot = $runtimeRoot
    outputDirectory = $deliveryRoot
    unpackedDirectory = $(if ($Edition -eq "Full") { $fullUnpackedRoot } else { $null })
    runtimeResourceTarget = $(if ($Edition -eq "Full") { "avtr-runtime" } else { $null })
    rootTensorRtLauncher = $(if ($Edition -eq "Full") { $rootTensorRtLauncherName } else { $null })
    archiveFormat = $(if ($Edition -eq "Full" -and $resolvedTarget -eq "Archive") { "zip64" } else { $null })
    archivePath = $(if ($Edition -eq "Full" -and $resolvedTarget -eq "Archive") { $fullArchivePath } else { $null })
    singleFileInstaller = $Edition -eq "Standard" -and $resolvedTarget -eq "Installer"
    cleanRuntime = [bool]$CleanRuntime
    includesModels = $Edition -eq "Full" -and -not [bool]$SkipModels
    torchWheelhouse = $resolvedTorchWheelhouse
    tauriArguments = $tauriArguments
    offlineWebView2 = [ordered]@{
        enabled = $Edition -eq "Standard" -and $resolvedTarget -eq "Installer"
        downloadsDuringBuild = $false
        generatedConfig = $standardOfflineConfigRelative
        fileName = $webView2InstallerName
        size = $webView2InstallerSize
        sha256 = $webView2InstallerSha256
        authenticodeStatus = "Valid"
        signerSubject = $webView2SignerSubject
    }
    requiredLicenses = $requiredLicenses
    runtimeValidation = [ordered]@{
        manifest = "runtime-manifest.json"
        layout = "portable-v2"
        defaultLayout = "portable-v2"
        layouts = @("portable-v1", "portable-v2")
        inspector = "scripts/desktop/inspect_runtime.py"
        requiredMemoryPackage = "src/avaturn_live_streamer/memory/__init__.py"
        requiredMemoryModules = @(
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
            "src/avaturn_live_streamer/memory/worklet.py"
        )
        requiredComponents = $requiredComponents
        versionedHelpers = @($versionedHelpers | ForEach-Object { $_.Replace("\", "/") })
        forbiddenPayloads = $forbiddenPayloads
    }
}

if ($PlanOnly) {
    Write-Output ($plan | ConvertTo-Json -Depth 8 -Compress)
    return
}

if ($Edition -eq "Full") {
    $sourceEngineManifest = Join-Path $root "artifacts\main\engine-manifest.json"
    if (Test-Path -LiteralPath $sourceEngineManifest -PathType Leaf) {
        throw "Full source tree contains a target-machine TensorRT engine manifest: $sourceEngineManifest"
    }
}

if (
    $Edition -eq "Full" -and
    -not $SkipRuntimeBuild -and
    $resolvedTorchWheelhouse -and
    -not (Test-Path -LiteralPath $resolvedTorchWheelhouse -PathType Container)
) {
    throw "Torch wheelhouse must be an existing directory: $resolvedTorchWheelhouse"
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description is missing: $Path"
    }
}

function Assert-TrustedOfflineWebView2Installer {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = [System.IO.Path]::GetFullPath($Path)
    Assert-FileExists -Path $resolved -Description "WebView2 offline installer"
    $file = Get-Item -LiteralPath $resolved
    if ($file.Name -cne $webView2InstallerName) {
        throw "WebView2 offline installer must be named '$webView2InstallerName': $resolved"
    }
    if ($file.Length -ne $webView2InstallerSize) {
        throw "WebView2 offline installer size mismatch (expected $webView2InstallerSize bytes): $resolved"
    }
    $hash = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
    if ($hash -ne $webView2InstallerSha256) {
        throw "WebView2 offline installer SHA256 mismatch: $resolved"
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $resolved
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "WebView2 offline installer Authenticode signature is not Valid ($($signature.Status)): $resolved"
    }
    $signer = $signature.SignerCertificate.Subject
    if (
        -not $signer.Equals($webView2SignerSubject, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $signer.StartsWith("$webView2SignerSubject,", [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "WebView2 offline installer signer is not Microsoft Corporation ('$signer'): $resolved"
    }
    $versionInfo = $file.VersionInfo
    if (
        $versionInfo.CompanyName -ne "Microsoft Corporation" -or
        $versionInfo.ProductName -ne "Microsoft Edge Update" -or
        $versionInfo.OriginalFilename -ne "MicrosoftEdgeUpdateSetup.exe"
    ) {
        throw "WebView2 offline installer version metadata is unexpected: $resolved"
    }
    return $resolved
}

function Resolve-TrustedOfflineWebView2Installer {
    if ($WebView2OfflineInstaller) {
        return Assert-TrustedOfflineWebView2Installer -Path $WebView2OfflineInstaller
    }
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is unavailable; pass -WebView2OfflineInstaller with the pinned Microsoft offline installer."
    }
    $cacheRoot = Join-Path $env:LOCALAPPDATA "tauri\x64"
    if (Test-Path -LiteralPath $cacheRoot -PathType Container) {
        foreach ($candidate in Get-ChildItem -LiteralPath $cacheRoot -Filter $webView2InstallerName -File -Recurse) {
            if ($candidate.Length -ne $webView2InstallerSize) {
                continue
            }
            $candidateHash = (Get-FileHash -LiteralPath $candidate.FullName -Algorithm SHA256).Hash
            if ($candidateHash -eq $webView2InstallerSha256) {
                return Assert-TrustedOfflineWebView2Installer -Path $candidate.FullName
            }
        }
    }
    throw (
        "Pinned WebView2 offline installer was not found under '$cacheRoot'. " +
        "Pass -WebView2OfflineInstaller with the Microsoft-signed file matching SHA256 $webView2InstallerSha256."
    )
}

function ConvertTo-NsisLiteralPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return $Path.Replace('$', '$$')
}

function Write-StandardOfflineWebView2Overlay {
    param([Parameter(Mandatory = $true)][string]$InstallerPath)

    $generatedRoot = Split-Path -Parent $standardOfflineConfigPath
    New-Item -ItemType Directory -Path $generatedRoot -Force | Out-Null
    $escapedInstallerPath = ConvertTo-NsisLiteralPath -Path $InstallerPath
    $hookTemplate = @'
; Generated by scripts/build_tauri_windows.ps1 from a pinned, verified local payload.
!macro NSIS_HOOK_PREINSTALL
  ${If} ${RunningX64}
    ReadRegStr $4 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${Else}
    ReadRegStr $4 HKLM "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${EndIf}
  ${If} $4 == ""
    ReadRegStr $4 HKCU "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2APPGUID}" "pv"
  ${EndIf}
  ${If} $4 == ""
  ${AndIf} $UpdateMode <> 1
    Delete "$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe"
    File "/oname=$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe" "__WEBVIEW2_SOURCE__"
    DetailPrint "Installing Microsoft Edge WebView2 Runtime..."
    ExecWait '$\"$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe$\" /silent /install' $1
    Delete "$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe"
    ${If} $1 <> 0
      Abort "Microsoft Edge WebView2 Runtime installation failed with exit code $1."
    ${EndIf}
  ${EndIf}
!macroend
'@
    $hookContent = $hookTemplate.Replace("__WEBVIEW2_SOURCE__", $escapedInstallerPath)
    [System.IO.File]::WriteAllText(
        $standardOfflineHookPath,
        $hookContent,
        [System.Text.UTF8Encoding]::new($false)
    )
    $overlay = [ordered]@{
        bundle = [ordered]@{
            windows = [ordered]@{
                webviewInstallMode = [ordered]@{ type = "skip" }
                nsis = [ordered]@{ installerHooks = $standardOfflineHookPath }
            }
        }
    }
    [System.IO.File]::WriteAllText(
        $standardOfflineConfigPath,
        ($overlay | ConvertTo-Json -Depth 8),
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Resolve-BuiltNsisInstaller {
    $baseConfig = Get-Content -LiteralPath $baseConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $fileName = "{0}_{1}_x64-setup.exe" -f $baseConfig.productName, $baseConfig.version
    $candidate = Join-Path $root "src-tauri\target\release\bundle\nsis\$fileName"
    Assert-FileExists -Path $candidate -Description "Built DigiBox NSIS installer"
    return $candidate
}

function Assert-OfflineWebView2Embedded {
    param(
        [Parameter(Mandatory = $true)][string]$InstallerPath,
        [Parameter(Mandatory = $true)][string]$VerifiedPayloadPath
    )

    $installer = Get-Item -LiteralPath $InstallerPath
    if ($installer.Length -le $webView2InstallerSize) {
        throw "DigiBox NSIS output is too small to contain the offline WebView2 payload: $InstallerPath"
    }
    $hook = Get-Content -LiteralPath $standardOfflineHookPath -Raw -Encoding UTF8
    $escapedVerifiedPayloadPath = ConvertTo-NsisLiteralPath -Path $VerifiedPayloadPath
    if (
        -not $hook.Contains($escapedVerifiedPayloadPath) -or
        -not $hook.Contains('File "/oname=$TEMP\MicrosoftEdgeWebView2RuntimeInstaller.exe"')
    ) {
        throw "Generated NSIS hook does not embed the verified WebView2 payload."
    }

    $sevenZip = Get-Command "7z.exe" -ErrorAction SilentlyContinue
    if (-not $sevenZip) {
        $programFiles7z = Join-Path $env:ProgramFiles "7-Zip\7z.exe"
        if (Test-Path -LiteralPath $programFiles7z -PathType Leaf) {
            $sevenZip = Get-Item -LiteralPath $programFiles7z
        }
    }
    if ($sevenZip) {
        $sevenZipPath = if ($sevenZip -is [System.Management.Automation.CommandInfo]) {
            $sevenZip.Source
        } else {
            $sevenZip.FullName
        }
        $listing = @(& $sevenZipPath "l" "-slt" $InstallerPath)
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip could not inspect the built DigiBox NSIS installer."
        }
        $listingText = $listing -join "`n"
        $payloadPattern = '(?ms)^Path = \$TEMP\\MicrosoftEdgeWebView2RuntimeInstaller\.exe\r?$.*?^Size = ' + $webView2InstallerSize + '\r?$'
        if ($listingText -notmatch $payloadPattern) {
            throw "Built DigiBox NSIS installer does not contain the expected offline WebView2 entry and size."
        }
    } else {
        Write-Warning "7-Zip is unavailable; offline WebView2 embedding was verified from the successful NSIS File directive and output size."
    }
}

function Assert-CompleteRuntimeComponents {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )
    foreach ($componentName in $requiredComponents) {
        $property = $Manifest.components.PSObject.Properties[$componentName]
        if ($null -eq $property -or $property.Value -ne $true) {
            throw "Full Tauri Runtime component '$componentName' is incomplete: $ManifestPath"
        }
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
        throw "Full Tauri Runtime memory package is incomplete; missing: $($missing -join ', ')"
    }
}

function Assert-NoForbiddenRuntimePayload {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

    $forbidden = @()
    $engineManifest = Join-Path $RuntimeRoot "artifacts\main\engine-manifest.json"
    if (Test-Path -LiteralPath $engineManifest -PathType Leaf) {
        $forbidden += $engineManifest
    }
    $managedPythonPrefixes = @(
        ((Join-Path $RuntimeRoot "python-main").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python-cosyvoice").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python-feynobg").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "python").ToLowerInvariant() + "\"),
        ((Join-Path $RuntimeRoot "packages").ToLowerInvariant() + "\")
    )
    foreach ($entry in (Get-ChildItem -LiteralPath $RuntimeRoot -Recurse -Force)) {
        $name = $entry.Name.ToLowerInvariant()
        $full = $entry.FullName.ToLowerInvariant()
        if ($entry.PSIsContainer) {
            if (
                $name -in @(
                    "user_assets", "local_voices", "voice_clones", "reference_audio",
                    ".trash", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache",
                    ".engine-staging", ".engine-backups"
                ) -or
                $full.EndsWith("\artifacts\main\avatars_artifacts\backgrounds") -or
                $full.EndsWith("\memory\backups") -or
                $full.EndsWith("\memory\pending-imports")
            ) {
                $forbidden += $entry.FullName
            }
            continue
        }

        $underManagedPython = $false
        foreach ($prefix in $managedPythonPrefixes) {
            if ($full.StartsWith($prefix)) {
                $underManagedPython = $true
                break
            }
        }
        $trustedCaBundle = $underManagedPython -and (
            $full.EndsWith("\certifi\cacert.pem") -or
            $full.EndsWith("\pip\_vendor\certifi\cacert.pem")
        )
        if (
            $entry.Extension -ieq ".engine" -or
            $entry.Extension -ieq ".plan" -or
            ($name.StartsWith("grid_sample_3d_plugin") -and $entry.Extension -ieq ".dll") -or
            $name -eq "spk2info.pt" -or
            $entry.Extension -ieq ".incomplete" -or
            $entry.Extension -ieq ".metadata" -or
            ($name.StartsWith(".env") -and $name -ne ".env.example") -or
            $entry.Extension -ieq ".key" -or
            ($entry.Extension -ieq ".pem" -and -not $trustedCaBundle) -or
            $name -in @("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm") -or
            ($name.StartsWith("digibox-memory") -and $entry.Extension -ieq ".json")
        ) {
            $forbidden += $entry.FullName
        }
    }
    if ($forbidden.Count -gt 0) {
        throw "Full Tauri Runtime contains forbidden private/cache/engine payloads: $($forbidden -join ', ')"
    }
}

function Resolve-RuntimeManifestPath {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if ([string]::IsNullOrWhiteSpace($RelativePath) -or [System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "$Description must be a non-empty relative Runtime path: $RelativePath"
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd("\", "/")
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot $RelativePath))
    $prefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description escapes the portable Runtime root: $RelativePath"
    }
    return $resolved
}

function Assert-SupportedRuntimeManifest {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )

    $supported = (
        ($Manifest.schemaVersion -eq 1 -and $Manifest.layout -eq "portable-v1") -or
        ($Manifest.schemaVersion -eq 2 -and $Manifest.layout -eq "portable-v2")
    )
    if (-not $supported) {
        throw "Unsupported portable Runtime manifest: $ManifestPath"
    }
}

function Resolve-RuntimePythonPath {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest
    )

    $relativePath = if ($Manifest.layout -eq "portable-v2") {
        [string]$Manifest.paths.python
    } else {
        [string]$Manifest.paths.mainPython
    }
    return Resolve-RuntimeManifestPath `
        -RuntimeRoot $RuntimeRoot `
        -RelativePath $relativePath `
        -Description "Manifest Python path"
}

function Resolve-RuntimePackageLayers {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][ValidateSet("main", "cosyvoice", "feynobg")][string]$Profile
    )

    $packageLayers = $Manifest.python.packageLayers
    $property = $packageLayers.PSObject.Properties[$Profile]
    if ($null -eq $property -or @($property.Value).Count -eq 0) {
        throw "Portable-v2 manifest is missing python.packageLayers.$Profile"
    }
    return @(
        foreach ($relativePath in @($property.Value)) {
            Resolve-RuntimeManifestPath `
                -RuntimeRoot $RuntimeRoot `
                -RelativePath ([string]$relativePath) `
                -Description "python.packageLayers.$Profile"
        }
    )
}

function Assert-RequiredRuntimePayload {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest
    )

    $requiredRelativePaths = @(
        "runtime-manifest.json",
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
        "third_party\CosyVoice\cosyvoice\cli\cosyvoice.py",
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
        "artifacts\main\build_artifacts\avtr1.scripted.pt",
        "artifacts\main\build_artifacts\hubert-lbs-avtr1.onnx",
        "artifacts\main\build_artifacts\decoder.onnx",
        "artifacts\main\build_artifacts\modnet.onnx",
        "artifacts\main\build_artifacts\stitch_network.onnx",
        "artifacts\main\build_artifacts\warp_network.onnx"
    ) + $versionedHelpers
    $requiredDirectories = @()
    if ($Manifest.layout -eq "portable-v2") {
        $requiredRelativePaths += "python\python.exe"
        $requiredDirectories += @(
            "packages\main",
            "packages\cosyvoice",
            "packages\feynobg",
            "packages\shared"
        )
        foreach ($profile in @("main", "cosyvoice", "feynobg")) {
            $requiredDirectories += @(
                Resolve-RuntimePackageLayers `
                    -RuntimeRoot $RuntimeRoot `
                    -Manifest $Manifest `
                    -Profile $profile
            )
        }
    } else {
        $requiredRelativePaths += @(
            "python-main\python.exe",
            "python-cosyvoice\python.exe",
            "python-feynobg\python.exe"
        )
    }
    $missing = @(
        foreach ($relativePath in $requiredRelativePaths) {
            $candidate = Join-Path $RuntimeRoot $relativePath
            if (
                -not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
                (Get-Item -LiteralPath $candidate).Length -le 0
            ) {
                ([string]$relativePath).Replace("/", "\")
            }
        }
        foreach ($directory in $requiredDirectories) {
            $directoryText = [string]$directory
            $isRooted = [System.IO.Path]::IsPathRooted($directoryText)
            $candidate = if ($isRooted) {
                $directoryText
            } else {
                Join-Path $RuntimeRoot $directoryText
            }
            if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
                if ($isRooted) {
                    $directoryText.Substring($RuntimeRoot.TrimEnd("\", "/").Length).TrimStart("\", "/")
                } else {
                    $directoryText.Replace("/", "\")
                }
            }
        }
    )
    if ($missing.Count -gt 0) {
        throw "Full Tauri Runtime payload is incomplete; missing: $($missing -join ', ')"
    }
}

function Assert-VersionedRuntimeHelpers {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

    foreach ($relativeHelperPath in $versionedHelpers) {
        $sourceHelper = Join-Path $root $relativeHelperPath
        $stagedHelper = Join-Path $RuntimeRoot $relativeHelperPath
        Assert-FileExists -Path $sourceHelper -Description "Versioned TensorRT helper"
        Assert-FileExists -Path $stagedHelper -Description "Staged TensorRT helper"
        $sourceHash = (Get-FileHash -LiteralPath $sourceHelper -Algorithm SHA256).Hash
        $stagedHash = (Get-FileHash -LiteralPath $stagedHelper -Algorithm SHA256).Hash
        if ($sourceHash -ne $stagedHash) {
            throw "Staged Runtime helper differs from this source tree: $relativeHelperPath"
        }
        if ([System.IO.Path]::GetExtension($stagedHelper) -ieq ".ps1") {
            $tokens = $null
            $parseErrors = $null
            [System.Management.Automation.Language.Parser]::ParseFile(
                $stagedHelper,
                [ref]$tokens,
                [ref]$parseErrors
            ) | Out-Null
            if ($parseErrors.Count -gt 0) {
                throw "Staged Runtime PowerShell helper is invalid: $stagedHelper"
            }
        }
    }
}

function Assert-RuntimeInspection {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest
    )

    $python = Resolve-RuntimePythonPath -RuntimeRoot $RuntimeRoot -Manifest $Manifest
    $inspector = Join-Path $RuntimeRoot "scripts\desktop\inspect_runtime.py"
    $previousDontWriteBytecode = [Environment]::GetEnvironmentVariable(
        "PYTHONDONTWRITEBYTECODE",
        "Process"
    )
    try {
        [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", "Process")
        $inspectionLines = @(& $python "-B" $inspector "--runtime-root" $RuntimeRoot "--json")
        if ($LASTEXITCODE -ne 0) {
            throw "Portable Runtime inspection failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "PYTHONDONTWRITEBYTECODE",
            $previousDontWriteBytecode,
            "Process"
        )
    }
    try {
        $inspection = ($inspectionLines -join [Environment]::NewLine) | ConvertFrom-Json
    }
    catch {
        throw "Portable Runtime inspector returned invalid JSON: $($_.Exception.Message)"
    }
    if (
        $inspection.schema_version -ne 1 -or
        $inspection.platform -ne "windows" -or
        $inspection.artifacts_ready -ne $true -or
        $inspection.models_ready -ne $true
    ) {
        throw "Portable Runtime inspector rejected the Full staged Runtime."
    }
    if (@($inspection.engine_files).Count -gt 0) {
        throw "Full delivery must not contain machine-specific TensorRT engines."
    }
}

function Assert-RuntimeDependencyImports {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]$Manifest
    )

    if ($Manifest.layout -eq "portable-v2") {
        $runtimePython = Resolve-RuntimePythonPath -RuntimeRoot $RuntimeRoot -Manifest $Manifest
        $probes = @(
            [ordered]@{
                name = "main"
                environmentName = "AVTR1_MAIN_PYTHONPATH"
                python = $runtimePython
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,tensorrt,nvvfx,fastapi,uvicorn,aiortc,cv2,avtr1_renderer,avaturn_live_streamer; import avaturn_live_streamer.local_stream_cli; import avaturn_live_streamer.memory.admin; import avaturn_live_streamer.memory.api"
                paths = @(Resolve-RuntimePackageLayers -RuntimeRoot $RuntimeRoot -Manifest $Manifest -Profile "main")
            },
            [ordered]@{
                name = "cosyvoice"
                environmentName = "AVTR1_COSYVOICE_PYTHONPATH"
                python = $runtimePython
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,torchaudio,onnxruntime,transformers,matcha; from cosyvoice.cli.cosyvoice import CosyVoice3; import avaturn_live_streamer.integrations.cosyvoice_server"
                paths = @(Resolve-RuntimePackageLayers -RuntimeRoot $RuntimeRoot -Manifest $Manifest -Profile "cosyvoice")
            },
            [ordered]@{
                name = "feynobg"
                environmentName = "AVTR1_FEYNOBG_PYTHONPATH"
                python = $runtimePython
                code = "import sys; assert 'sitecustomize' in sys.modules; import torch,torchvision,nobg,avaturn_live_streamer.integrations.feynobg_server"
                paths = @(Resolve-RuntimePackageLayers -RuntimeRoot $RuntimeRoot -Manifest $Manifest -Profile "feynobg")
            }
        )
    } else {
        $probes = @(
            [ordered]@{
                name = "python-main"
                environmentName = $null
                python = Resolve-RuntimeManifestPath -RuntimeRoot $RuntimeRoot -RelativePath ([string]$Manifest.paths.mainPython) -Description "Manifest main Python path"
                code = "import torch,tensorrt,nvvfx,fastapi,uvicorn,aiortc,cv2,avtr1_renderer,avaturn_live_streamer; import avaturn_live_streamer.local_stream_cli; import avaturn_live_streamer.memory.admin; import avaturn_live_streamer.memory.api"
                paths = @((Join-Path $RuntimeRoot "src"))
            },
            [ordered]@{
                name = "python-cosyvoice"
                environmentName = $null
                python = Resolve-RuntimeManifestPath -RuntimeRoot $RuntimeRoot -RelativePath ([string]$Manifest.paths.cosyvoicePython) -Description "Manifest CosyVoice Python path"
                code = "import torch,torchaudio,onnxruntime,transformers,matcha; from cosyvoice.cli.cosyvoice import CosyVoice3; import avaturn_live_streamer.integrations.cosyvoice_server"
                paths = @(
                    (Join-Path $RuntimeRoot "src"),
                    (Join-Path $RuntimeRoot "third_party\CosyVoice"),
                    (Join-Path $RuntimeRoot "third_party\CosyVoice\third_party\Matcha-TTS")
                )
            },
            [ordered]@{
                name = "python-feynobg"
                environmentName = $null
                python = Resolve-RuntimeManifestPath -RuntimeRoot $RuntimeRoot -RelativePath ([string]$Manifest.paths.feynobgPython) -Description "Manifest FeyNoBg Python path"
                code = "import torch,torchvision,nobg,avaturn_live_streamer.integrations.feynobg_server"
                paths = @((Join-Path $RuntimeRoot "src"))
            }
        )
    }
    foreach ($probe in $probes) {
        $previousPythonPath = $null
        $previousRoutedPythonPath = $null
        $previousNoUserSite = [Environment]::GetEnvironmentVariable("PYTHONNOUSERSITE", "Process")
        $previousDontWriteBytecode = [Environment]::GetEnvironmentVariable(
            "PYTHONDONTWRITEBYTECODE",
            "Process"
        )
        if ($probe.environmentName) {
            $previousRoutedPythonPath = [Environment]::GetEnvironmentVariable($probe.environmentName, "Process")
            [Environment]::SetEnvironmentVariable(
                $probe.environmentName,
                [string]::Join([System.IO.Path]::PathSeparator, @($probe.paths)),
                "Process"
            )
        }
        try {
            $previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
            [Environment]::SetEnvironmentVariable("PYTHONPATH", [string]::Join([System.IO.Path]::PathSeparator, @($probe.paths)), "Process")
            [Environment]::SetEnvironmentVariable("PYTHONNOUSERSITE", "1", "Process")
            [Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", "Process")
            & $probe.python "-B" "-c" $probe.code
            if ($LASTEXITCODE -ne 0) {
                throw "Full Tauri Runtime dependency import failed for $($probe.name)."
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
            if ($probe.environmentName) {
                [Environment]::SetEnvironmentVariable(
                    $probe.environmentName,
                    $previousRoutedPythonPath,
                    "Process"
                )
            }
        }
    }
}

function Assert-FullRuntime {
    param([Parameter(Mandatory = $true)][string]$RuntimeRoot)

    $manifestPath = Join-Path $RuntimeRoot "runtime-manifest.json"
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "Portable Runtime manifest is invalid JSON: $manifestPath"
    }
    Assert-SupportedRuntimeManifest -Manifest $manifest -ManifestPath $manifestPath
    Assert-RequiredRuntimePayload -RuntimeRoot $RuntimeRoot -Manifest $manifest
    Assert-MemoryPackageEntrypoint -RuntimeRoot $RuntimeRoot
    Assert-CompleteRuntimeComponents -Manifest $manifest -ManifestPath $manifestPath
    if (
        $manifest.privacy.userAssetsIncluded -ne $false -or
        $manifest.privacy.localVoiceCacheIncluded -ne $false -or
        $manifest.privacy.machineSpecificEnginesIncluded -ne $false
    ) {
        throw "Portable Runtime manifest permits private or machine-specific payloads."
    }
    Assert-NoForbiddenRuntimePayload -RuntimeRoot $RuntimeRoot
    Assert-VersionedRuntimeHelpers -RuntimeRoot $RuntimeRoot
    Assert-RuntimeInspection -RuntimeRoot $RuntimeRoot -Manifest $manifest
    Assert-RuntimeDependencyImports -RuntimeRoot $RuntimeRoot -Manifest $manifest
}

function Resolve-TauriExecutable {
    if ($TauriExecutable) {
        $candidate = [System.IO.Path]::GetFullPath($TauriExecutable)
        Assert-FileExists -Path $candidate -Description "Tauri executable"
        return $candidate
    }
    $localCli = Join-Path $root "node_modules\.bin\tauri.cmd"
    Assert-FileExists -Path $localCli -Description "Pinned Tauri CLI"
    return $localCli
}

function Resolve-BuiltShell {
    $cargoManifestPath = Join-Path $root "src-tauri\Cargo.toml"
    $cargoManifest = Get-Content -LiteralPath $cargoManifestPath -Raw -Encoding UTF8
    $packageMatch = [regex]::Match(
        $cargoManifest,
        '(?ms)^\[package\].*?^name\s*=\s*"([^"]+)"'
    )
    if (-not $packageMatch.Success) {
        throw "Unable to resolve the Tauri package name from: $cargoManifestPath"
    }
    $candidate = Join-Path $root ("src-tauri\target\release\{0}.exe" -f $packageMatch.Groups[1].Value)
    Assert-FileExists -Path $candidate -Description "Built Tauri shell"
    return $candidate
}

function Copy-DirectoryWithRobocopy {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $robocopy = (Get-Command "robocopy.exe" -ErrorAction Stop).Source
    & $robocopy $Source $Destination "/E" "/COPY:DAT" "/DCOPY:DAT" "/R:2" "/W:1" "/XJ" "/NFL" "/NDL" "/NJH" "/NJS"
    if ($LASTEXITCODE -ge 8) {
        throw "Runtime staging failed with robocopy exit code $LASTEXITCODE."
    }
}

foreach ($relativeLicense in $requiredLicenses) {
    Assert-FileExists -Path (Join-Path $root $relativeLicense) -Description "Required license"
}
Assert-FileExists -Path (Join-Path $root "package.json") -Description "Desktop package manifest"
Assert-FileExists -Path (Join-Path $root "src-tauri\Cargo.toml") -Description "Tauri Cargo manifest"
Assert-FileExists -Path $baseConfigPath -Description "Base Tauri configuration"
if ($Edition -eq "Full") {
    Assert-FileExists -Path $fullConfigPath -Description "Full Tauri configuration"
    Assert-FileExists -Path $rootTensorRtLauncherSource -Description "Root TensorRT setup launcher source"
}

if ($Edition -eq "Full" -and -not $SkipRuntimeBuild -and ($SkipRuntimeDependencies -or $SkipModels)) {
    throw "A Full distribution cannot skip Runtime dependencies or models."
}

if ($Edition -eq "Full") {
    $runtimePrefix = $runtimeRoot + [System.IO.Path]::DirectorySeparatorChar
    $deliveryPrefix = $fullUnpackedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (
        $fullUnpackedRoot.StartsWith($runtimePrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        $runtimeRoot.StartsWith($deliveryPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        $runtimeRoot.Equals($fullUnpackedRoot, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "RuntimeDestination and the Full delivery directory must not contain one another."
    }
}

if ($Edition -eq "Full") {
    if (-not $SkipRuntimeBuild) {
        Assert-FileExists -Path $runtimeBuilder -Description "Portable Runtime builder"
        $runtimeArguments = @{
            SourceRoot = $root
            Destination = $runtimeRoot
            Layout = "PortableV2"
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
    Assert-FullRuntime -RuntimeRoot $runtimeRoot
}

$verifiedWebView2Installer = $null
if ($Edition -eq "Standard" -and $resolvedTarget -eq "Installer") {
    $verifiedWebView2Installer = Resolve-TrustedOfflineWebView2Installer
    Write-StandardOfflineWebView2Overlay -InstallerPath $verifiedWebView2Installer
}

$tauriCli = Resolve-TauriExecutable
Push-Location $root
try {
    & $tauriCli @tauriArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

if ($Edition -eq "Standard" -and $resolvedTarget -eq "Installer") {
    $builtNsisInstaller = Resolve-BuiltNsisInstaller
    Assert-OfflineWebView2Embedded `
        -InstallerPath $builtNsisInstaller `
        -VerifiedPayloadPath $verifiedWebView2Installer
    Write-Host "Verified offline WebView2 payload in: $builtNsisInstaller" -ForegroundColor Green
}

if ($Edition -eq "Full") {
    $builtShell = Resolve-BuiltShell
    if (Test-Path -LiteralPath $fullUnpackedRoot) {
        Remove-Item -LiteralPath $fullUnpackedRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $fullUnpackedRoot -Force | Out-Null
    Copy-Item -LiteralPath $builtShell -Destination (Join-Path $fullUnpackedRoot $builtShell.Name)
    Copy-DirectoryWithRobocopy -Source $runtimeRoot -Destination (Join-Path $fullUnpackedRoot "avtr-runtime")
    $rootTensorRtLauncherTarget = Join-Path $fullUnpackedRoot $rootTensorRtLauncherName
    Copy-Item -LiteralPath $rootTensorRtLauncherSource -Destination $rootTensorRtLauncherTarget

    $licenseRoot = Join-Path $fullUnpackedRoot "licenses"
    New-Item -ItemType Directory -Path $licenseRoot -Force | Out-Null
    foreach ($relativeLicense in $requiredLicenses) {
        Copy-Item -LiteralPath (Join-Path $root $relativeLicense) -Destination (Join-Path $licenseRoot $relativeLicense)
    }

    Assert-NoForbiddenRuntimePayload -RuntimeRoot (Join-Path $fullUnpackedRoot "avtr-runtime")
    Assert-VersionedRuntimeHelpers -RuntimeRoot (Join-Path $fullUnpackedRoot "avtr-runtime")
    Assert-FileExists -Path $rootTensorRtLauncherTarget -Description "Root TensorRT setup launcher"
    $rootTensorRtLauncherSourceHash = (Get-FileHash -LiteralPath $rootTensorRtLauncherSource -Algorithm SHA256).Hash
    $rootTensorRtLauncherTargetHash = (Get-FileHash -LiteralPath $rootTensorRtLauncherTarget -Algorithm SHA256).Hash
    if ($rootTensorRtLauncherTargetHash -ne $rootTensorRtLauncherSourceHash) {
        throw "Root TensorRT setup launcher verification failed: $rootTensorRtLauncherTarget"
    }

    if ($resolvedTarget -eq "Archive") {
        $tar = (Get-Command "tar.exe" -ErrorAction Stop).Source
        if (Test-Path -LiteralPath $fullArchivePath) {
            Remove-Item -LiteralPath $fullArchivePath -Force
        }
        # Windows bsdtar writes ZIP64 records automatically when the staged payload requires them.
        & $tar "-a" "-c" "-f" $fullArchivePath "-C" $deliveryRoot (Split-Path -Leaf $fullUnpackedRoot)
        if ($LASTEXITCODE -ne 0) {
            throw "Full ZIP64 archive creation failed with exit code $LASTEXITCODE."
        }
        Write-Host "Full portable Tauri archive created: $fullArchivePath" -ForegroundColor Green
    } else {
        Write-Host "Full unpacked Tauri delivery created: $fullUnpackedRoot" -ForegroundColor Green
    }
} else {
    Write-Host "Standard Tauri shell build completed ($resolvedTarget)." -ForegroundColor Green
}
