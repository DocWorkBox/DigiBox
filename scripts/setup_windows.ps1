[CmdletBinding()]
param(
    [switch]$CheckHuggingFaceAccess,
    [switch]$DownloadModels,
    [switch]$EnableTensorRT,
    [switch]$EnableNvidiaVfx,
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

$diagnosticArguments = @((Join-Path $runtime.RepoRoot "scripts\windows_diagnostics.py"))
if ($CheckHuggingFaceAccess -or $DownloadModels) {
    $diagnosticArguments += "--check-hf-access"
}
if ($EnableTensorRT) {
    $diagnosticArguments += "--require-tensorrt"
}

$steps = @(
    [pscustomobject]@{
        Name = "diagnostics"
        PythonPath = [string]$runtime.PythonPath.Main
        Arguments = @($diagnosticArguments)
    }
)
if ($EnableNvidiaVfx) {
    $steps += [pscustomobject]@{
        Name = "nvidia-vfx"
        PythonPath = [string]$runtime.PythonPath.Main
        Arguments = @("-B", "-c", "import nvvfx")
    }
}
if ($DownloadModels) {
    $steps += [pscustomobject]@{
        Name = "download-models"
        PythonPath = [string]$runtime.PythonPath.Main
        Arguments = @((Join-Path $runtime.RepoRoot "scripts\download_artifacts.py"))
    }
}

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
    RuntimeRoot = [string]$runtime.RuntimeRoot
    Environment = $runtime.Environment
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
    foreach ($step in $steps) {
        [Environment]::SetEnvironmentVariable(
            "PYTHONPATH",
            [string]$step.PythonPath,
            [EnvironmentVariableTarget]::Process
        )
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

Write-Host "DigiBox source development uses Full Runtime: $($runtime.RuntimeRoot)"
Write-Host "Python: $($runtime.Python)"
