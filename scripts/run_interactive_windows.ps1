[CmdletBinding()]
param(
    [ValidateSet("auto", "torchscript", "tensorrt")]
    [string]$Backend = "auto",
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
$runtime.Environment["AVTR1_BACKEND"] = $Backend
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$arguments = @((Join-Path $runtime.RepoRoot "scripts\run_local_stream.py"))

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
    Arguments = @($arguments)
    Environment = $environment
}
if ($PlanOnly) {
    return $plan
}

Assert-DigiBoxDevelopmentPortsAvailable
$snapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime
$exitCode = 1
$locationPushed = $false
try {
    Push-Location $runtime.RepoRoot
    $locationPushed = $true
    & $runtime.Python @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $snapshot
}
exit $exitCode
