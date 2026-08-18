[CmdletBinding()]
param(
    [int]$Port = 8768,
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
$modelDir = Join-Path $runtime.ModelsRoot "Fun-CosyVoice3-0.5B-2512"
if (-not (Test-Path -LiteralPath (Join-Path $modelDir "cosyvoice3.yaml"))) {
    throw "CosyVoice model is missing from the selected Full Runtime: $modelDir"
}

$runtime.Environment["PYTHONPATH"] = [string]$runtime.PythonPath.CosyVoice
$runtime.Environment["AVTR_COSYVOICE_MODEL_DIR"] = $modelDir
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$arguments = @(
    "-m",
    "uvicorn",
    "avaturn_live_streamer.integrations.cosyvoice_server:app",
    "--host",
    "127.0.0.1",
    "--port",
    [string]$Port
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
