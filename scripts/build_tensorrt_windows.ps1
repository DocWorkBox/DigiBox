[CmdletBinding()]
param(
    [switch]$IncludeWarp,
    [string]$RuntimeRoot = "",
    [string]$RepoRoot = "",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "dev_runtime_windows.ps1")

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot ".."
}
$runtime = Resolve-DigiBoxSourceRuntime -RepoRoot $RepoRoot -RuntimeRoot $RuntimeRoot
$python = [string]$runtime.Python
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$steps = @(
    "diagnostics",
    "download-artifacts",
    "build-avtr1",
    "build-hubert"
)
if ($IncludeWarp) {
    $steps += "build-warp-plugin"
}
$steps += "build-renderer"

$plan = [pscustomobject]@{
    Python = $python
    RuntimeRoot = [string]$runtime.RuntimeRoot
    Environment = $environment
    Steps = @($steps)
}
if ($PlanOnly) {
    return $plan
}

$snapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime
$locationPushed = $false
try {
    Push-Location $runtime.RepoRoot
    $locationPushed = $true
    & $python scripts\windows_diagnostics.py --check-hf-access --require-tensorrt
    if ($LASTEXITCODE -ne 0) { throw "TensorRT or Hugging Face diagnostics failed" }

    & $python scripts\download_artifacts.py
    if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

    & $python scripts\build_avtr1_engines.py
    if ($LASTEXITCODE -ne 0) { throw "AVTR-1 TensorRT build failed" }

    & $python scripts\build_hubert_engine.py
    if ($LASTEXITCODE -ne 0) { throw "HuBERT TensorRT build failed" }

    $rendererTargets = @("decoder", "modnet", "stitch")
    $pluginArgs = @()
    if ($IncludeWarp) {
        if (-not $env:AVTR1_WARP_PLUGIN) {
            & (Join-Path $PSScriptRoot "build_warp_plugin_windows.ps1") `
                -Python $runtime.Python
            if ($LASTEXITCODE -ne 0) {
                throw "Windows GridSample3D plugin build failed."
            }
        }
        if (-not (Test-Path -LiteralPath $env:AVTR1_WARP_PLUGIN)) {
            throw "AVTR1_WARP_PLUGIN does not exist: $env:AVTR1_WARP_PLUGIN"
        }
        $pluginHash = (
            Get-FileHash -LiteralPath $env:AVTR1_WARP_PLUGIN -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        & $python scripts\smoke_warp_plugin_windows.py `
            --plugin $env:AVTR1_WARP_PLUGIN `
            --max-batch 5
        if ($LASTEXITCODE -ne 0) {
            throw "GridSample3D dynamic-batch smoke failed."
        }

        $warpB1Validation = Join-Path `
            $runtime.ArtifactsRoot `
            "build\warp-b1-validation-$pluginHash"
        & $python scripts\build_renderer_engines.py `
            warp `
            --max-batch 1 `
            --warp-plugin $env:AVTR1_WARP_PLUGIN `
            --out-dir $warpB1Validation
        if ($LASTEXITCODE -ne 0) {
            throw "Batch-1 warp TensorRT validation build failed."
        }

        $warpEnginePath = & $python -c (
            "from avtr1_renderer.avtr1_artifact_manager import get_trt_engine_path; " +
            "print(get_trt_engine_path('warp_network'))"
        )
        if ($LASTEXITCODE -ne 0 -or -not $warpEnginePath) {
            throw "Could not resolve the formal warp TensorRT engine path."
        }
        $installedWarpPlugin = Join-Path `
            (Split-Path -Parent ($warpEnginePath | Select-Object -Last 1)) `
            "grid_sample_3d_plugin.dll"
        if (Test-Path -LiteralPath $installedWarpPlugin) {
            $installedPluginHash = (
                Get-FileHash -LiteralPath $installedWarpPlugin -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            if ($installedPluginHash -ne $pluginHash) {
                $backupTimestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
                $backupBase = "$installedWarpPlugin.backup-$backupTimestamp-$installedPluginHash"
                $backupPath = $backupBase
                $backupIndex = 0
                while (Test-Path -LiteralPath $backupPath) {
                    $backupIndex += 1
                    $backupPath = "$backupBase-$backupIndex"
                }
                Move-Item -LiteralPath $installedWarpPlugin -Destination $backupPath
                Write-Host "Backed up existing warp plugin: $backupPath"
            }
        }

        $rendererTargets += "warp"
        $pluginArgs = @("--warp-plugin", $env:AVTR1_WARP_PLUGIN)
    }

    & $python scripts\build_renderer_engines.py @rendererTargets @pluginArgs
    if ($LASTEXITCODE -ne 0) { throw "Renderer TensorRT build failed" }

    Write-Host "Windows TensorRT engines built successfully."
    if (-not $IncludeWarp) {
        Write-Host "Warp remains on ONNX Runtime CUDA; launch with -Backend tensorrt."
    }
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $snapshot
}
