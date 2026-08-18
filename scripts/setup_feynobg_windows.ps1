[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [switch]$SkipModelDownload,
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
$runtime.Environment["AVTR1_FEYNOBG_DEVICE"] = $Device
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$environment["PYTHONPATH"] = [string]$runtime.PythonPath.FeyNoBg

$steps = @(
    [pscustomobject]@{
        Name = "feynobg-import"
        Arguments = @("-B", "-c", "import nobg, torch, torchvision")
    }
)
if (-not $SkipModelDownload) {
    $steps += [pscustomobject]@{
        Name = "feynobg-model"
        Arguments = @((Join-Path $runtime.RepoRoot "scripts\download_feynobg.py"))
    }
}

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
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
    [Environment]::SetEnvironmentVariable(
        "PYTHONPATH",
        [string]$runtime.PythonPath.FeyNoBg,
        [EnvironmentVariableTarget]::Process
    )
    Push-Location $runtime.RepoRoot
    $locationPushed = $true
    foreach ($step in $steps) {
        & $runtime.Python @($step.Arguments)
        if ($LASTEXITCODE -ne 0) {
            throw "$($step.Name) failed with exit code $LASTEXITCODE."
        }
    }
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $snapshot
}

Write-Host "FeyNoBg source worker uses Full Python: $($runtime.Python)"
