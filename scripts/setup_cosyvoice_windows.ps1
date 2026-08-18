[CmdletBinding()]
param(
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
$modelDir = Join-Path $runtime.ModelsRoot "Fun-CosyVoice3-0.5B-2512"
$runtime.Environment["AVTR_COSYVOICE_MODEL_DIR"] = $modelDir
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$environment["PYTHONPATH"] = [string]$runtime.PythonPath.CosyVoice

$requiredModelFiles = @(
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx"
)
$missingModelFiles = @($requiredModelFiles | Where-Object {
    $path = Join-Path $modelDir $_
    -not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Item -LiteralPath $path).Length -le 0
})

$steps = @()
if ($missingModelFiles.Count -gt 0) {
    if ($SkipModelDownload) {
        throw (
            "CosyVoice model is incomplete in the selected Full Runtime: " +
            ($missingModelFiles -join ", ")
        )
    }
    $steps += [pscustomobject]@{
        Name = "cosyvoice-model"
        Arguments = @((Join-Path $runtime.RepoRoot "scripts\download_cosyvoice.py"))
    }
}
$steps += [pscustomobject]@{
    Name = "cosyvoice-import"
    Arguments = @(
        "-B",
        "-c",
        "import fastapi, huggingface_hub, torch, uvicorn, websockets; from cosyvoice.cli.cosyvoice import AutoModel"
    )
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
        [string]$runtime.PythonPath.CosyVoice,
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

Write-Host "CosyVoice source worker uses Full Python: $($runtime.Python)"
