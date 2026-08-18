[CmdletBinding()]
param(
    [string]$RuntimeRoot = "",
    [string]$RepoRoot = "",
    [switch]$PlanOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @()
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "dev_runtime_windows.ps1")

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot ".."
}
$runtime = Resolve-DigiBoxSourceRuntime -RepoRoot $RepoRoot -RuntimeRoot $RuntimeRoot
$testDataRoot = Join-Path $runtime.RepoRoot (
    "test-results\source-test-data-$PID-$([Guid]::NewGuid().ToString('N'))"
)
$runtime.Environment["AVTR1_COSYVOICE_SPEAKER_CACHE"] = ""
$runtime.Environment["AVTR_COSYVOICE_MODEL_DIR"] = Join-Path `
    $testDataRoot "cosyvoice\model"
$runtime.Environment["AVTR1_USER_ASSETS_ROOT"] = Join-Path $testDataRoot "user_assets"
$runtime.Environment["AVTR1_MEMORY_ROOT"] = Join-Path $testDataRoot "memory"
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$environment["PYTHONPATH"] = [string]$runtime.PythonPath.Main

[string[]]$arguments = @(
    "-B",
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider"
) + @($PytestArgs)

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
    Arguments = @($arguments)
    Environment = $environment
}
if ($PlanOnly) {
    return $plan
}

$smokeProfiles = @(
    [pscustomobject]@{
        Name = "main"
        PythonPath = [string]$runtime.PythonPath.Main
        Code = @'
import os
from pathlib import Path
import avtr1_renderer
import avaturn_live_streamer
import pytest

source = (Path(os.environ['AVTR1_APP_ROOT']) / 'src').resolve()
for module in (avtr1_renderer, avaturn_live_streamer):
    assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
'@
    },
    [pscustomobject]@{
        Name = "cosyvoice"
        PythonPath = [string]$runtime.PythonPath.CosyVoice
        Code = @'
import importlib.util
import os
from pathlib import Path
import avaturn_live_streamer
import cosyvoice
import fastapi
import uvicorn
import websockets

repo = Path(os.environ['AVTR1_APP_ROOT']).resolve()
assert Path(avaturn_live_streamer.__file__).resolve().is_relative_to(repo / 'src')
assert Path(next(iter(cosyvoice.__path__))).resolve().is_relative_to(repo / 'third_party' / 'CosyVoice')
assert importlib.util.find_spec('torch') is not None
'@
    },
    [pscustomobject]@{
        Name = "feynobg"
        PythonPath = [string]$runtime.PythonPath.FeyNoBg
        Code = @'
import importlib.util
import os
from pathlib import Path
import avaturn_live_streamer
import fastapi
import transformers
import uvicorn

repo_source = (Path(os.environ['AVTR1_APP_ROOT']) / 'src').resolve()
assert Path(avaturn_live_streamer.__file__).resolve().is_relative_to(repo_source)
assert importlib.util.find_spec('nobg') is not None
assert importlib.util.find_spec('torch') is not None
'@
    }
)

$pytestExitCode = 0
$locationPushed = $false
$environmentSnapshot = $null
try {
    $environmentSnapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime

    foreach ($profile in $smokeProfiles) {
        [Environment]::SetEnvironmentVariable(
            "PYTHONPATH",
            [string]$profile.PythonPath,
            [EnvironmentVariableTarget]::Process
        )
        & $runtime.Python -B -c ([string]$profile.Code)
        if ($LASTEXITCODE -ne 0) {
            throw "$($profile.Name) package-layer smoke test failed with exit code $LASTEXITCODE."
        }
    }

    [Environment]::SetEnvironmentVariable(
        "PYTHONPATH",
        [string]$runtime.PythonPath.Main,
        [EnvironmentVariableTarget]::Process
    )
    Push-Location $runtime.RepoRoot
    $locationPushed = $true
    & $runtime.Python @arguments
    $pytestExitCode = $LASTEXITCODE
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    if ($null -ne $environmentSnapshot) {
        Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $environmentSnapshot
    }
}

if ($pytestExitCode -ne 0) {
    throw "pytest failed with exit code $pytestExitCode."
}
