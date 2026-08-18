[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [int]$Port = 8767,
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
$runtime.Environment["PYTHONPATH"] = [string]$runtime.PythonPath.FeyNoBg
$runtime.Environment["AVTR1_FEYNOBG_DEVICE"] = $Device
$runtime.Environment["AVTR1_FEYNOBG_PORT"] = [string]$Port
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$arguments = @(
    "-m",
    "avaturn_live_streamer.integrations.feynobg_server"
)

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
    Arguments = @($arguments)
    Environment = $environment
}
if ($PlanOnly) {
    return $plan
}

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
