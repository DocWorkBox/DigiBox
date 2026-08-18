[CmdletBinding()]
param(
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
$manifestPath = Join-Path $runtime.RepoRoot "src-tauri\Cargo.toml"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Tauri Cargo manifest is missing: $manifestPath"
}

$pythonContractScript = Join-Path $runtime.RepoRoot "scripts\test_windows.ps1"
$testDataRoot = Join-Path $runtime.RepoRoot (
    "test-results\pytest-tauri-$PID-$([Guid]::NewGuid().ToString('N'))"
)
[string[]]$pythonContractTests = @(
    "-q",
    "--basetemp=$testDataRoot",
    "tests/desktop/test_tauri_shell_contract.py",
    "tests/desktop/test_tauri_build_contract.py"
)
[string[]]$plannedPythonContractTests = @(
    "-q",
    "--basetemp=<generated-under-test-results>",
    "tests/desktop/test_tauri_shell_contract.py",
    "tests/desktop/test_tauri_build_contract.py"
)
[string[]]$pythonStepArguments = @(
    "-RuntimeRoot",
    [string]$runtime.RuntimeRoot,
    "-RepoRoot",
    [string]$runtime.RepoRoot,
    "-PytestArgs"
) + $plannedPythonContractTests

$cargoCommand = "cargo"
$steps = @(
    [pscustomobject]@{
        Name = "python-tauri-contracts"
        Command = $pythonContractScript
        Arguments = $pythonStepArguments
    },
    [pscustomobject]@{
        Name = "cargo-fmt"
        Command = $cargoCommand
        Arguments = @("fmt", "--manifest-path", $manifestPath, "--", "--check")
    },
    [pscustomobject]@{
        Name = "cargo-check"
        Command = $cargoCommand
        Arguments = @(
            "check", "--manifest-path", $manifestPath, "--all-targets", "--offline"
        )
    },
    [pscustomobject]@{
        Name = "cargo-clippy"
        Command = $cargoCommand
        Arguments = @(
            "clippy", "--manifest-path", $manifestPath, "--all-targets", "--offline",
            "--", "-D", "warnings"
        )
    },
    [pscustomobject]@{
        Name = "cargo-test"
        Command = $cargoCommand
        Arguments = @(
            "test", "--manifest-path", $manifestPath, "--all-targets", "--offline"
        )
    }
)

$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}
$environment["AVTR1_TEST_PYTHON"] = [string]$runtime.Python

$plan = [pscustomobject]@{
    Python = [string]$runtime.Python
    ManifestPath = $manifestPath
    Environment = $environment
    Steps = $steps
}
if ($PlanOnly) {
    return $plan
}

$runtime.Environment["AVTR1_TEST_PYTHON"] = [string]$runtime.Python
$environmentSnapshot = $null
try {
    $environmentSnapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime

    & $pythonContractScript `
        -RuntimeRoot $runtime.RuntimeRoot `
        -RepoRoot $runtime.RepoRoot `
        -PytestArgs $pythonContractTests

    foreach ($step in @($steps | Select-Object -Skip 1)) {
        & $step.Command @($step.Arguments)
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "$($step.Name) failed with exit code $exitCode."
        }
    }
}
finally {
    if ($null -ne $environmentSnapshot) {
        Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $environmentSnapshot
    }
}
