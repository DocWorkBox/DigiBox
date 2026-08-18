# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

[CmdletBinding()]
param(
    [ValidateSet("Tauri", "Electron")]
    [string]$Shell = "Tauri",
    [string]$RuntimeRoot = "",
    [string]$RepoRoot = "",
    [ValidateRange(1, 900)]
    [int]$StartupTimeoutSeconds = 300,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "dev_runtime_windows.ps1")

function Wait-DigiBoxSourceDesktopHealth {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$BackendProcess,
        [Parameter(Mandatory = $true)][string]$HealthUrl,
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($BackendProcess.HasExited) {
            throw "DigiBox source backend exited before becoming ready (exit code $($BackendProcess.ExitCode))."
        }
        try {
            $health = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2
            if (
                $health.service -eq "avtr1-streamer" -and
                $health.status -in @("ready", "degraded") -and
                $health.instance_id -eq $InstanceId
            ) {
                return
            }
        }
        catch {
            # Connection failures are expected while the source backend warms up.
        }
        Start-Sleep -Milliseconds 250
    }
    throw "DigiBox source backend did not become ready within $TimeoutSeconds seconds."
}

function Stop-DigiBoxSourceDesktopBackend {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$BackendProcess,
        [Parameter(Mandatory = $true)][string]$StopFile
    )

    [System.IO.File]::WriteAllText(
        $StopFile,
        "stop $([DateTimeOffset]::UtcNow.ToString('O'))`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    if (-not $BackendProcess.HasExited) {
        $null = $BackendProcess.WaitForExit(20000)
    }
    if (-not $BackendProcess.HasExited) {
        $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
        & $taskkill "/PID" ([string]$backendProcess.Id) "/T" "/F" | Out-Host
        $taskkillExitCode = $LASTEXITCODE
        if (-not $BackendProcess.HasExited) {
            $null = $BackendProcess.WaitForExit(5000)
        }
        if ($taskkillExitCode -ne 0 -or -not $BackendProcess.HasExited) {
            throw "Could not stop owned source backend process $($BackendProcess.Id)."
        }
    }
}

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot ".."
}
$runtime = Resolve-DigiBoxSourceRuntime -RepoRoot $RepoRoot -RuntimeRoot $RuntimeRoot
$healthUrl = "http://127.0.0.1:7860/health"
$instanceId = "source-dev-$PID-$([Guid]::NewGuid().ToString('N'))"
$stopFile = Join-Path (
    [System.IO.Path]::GetTempPath()
) "avtr1-source-desktop-$PID-$([Guid]::NewGuid().ToString('N')).stop"
$runtime.Environment["AVTR1_DESKTOP_INSTANCE_ID"] = $instanceId
$runtime.Environment["AVTR1_DESKTOP_STOP_FILE"] = $stopFile
$runtime.Environment["AVTR1_DESKTOP_RUNTIME"] = [string]$runtime.RuntimeRoot

$commandName = if ($Shell -eq "Tauri") { "tauri.cmd" } else { "electron.cmd" }
$command = Join-Path $runtime.RepoRoot "node_modules\.bin\$commandName"
[string[]]$commandArguments = if ($Shell -eq "Tauri") { @("dev") } else { @(".") }
[string[]]$backendArguments = @(
    (Join-Path $runtime.RepoRoot "scripts\run_local_stream.py")
)
[string[]]$launchArguments = @('"' + $backendArguments[0] + '"')
$environment = [ordered]@{}
foreach ($key in $runtime.Environment.Keys) {
    $environment[[string]$key] = [string]$runtime.Environment[$key]
}

$plan = [pscustomobject]@{
    Shell = $Shell
    Python = [string]$runtime.Python
    BackendArguments = @($backendArguments)
    LaunchArguments = @($launchArguments)
    Command = $command
    CommandArguments = @($commandArguments)
    HealthUrl = $healthUrl
    Environment = $environment
}
if ($PlanOnly) {
    return $plan
}

Assert-DigiBoxDevelopmentPortsAvailable
if (-not (Test-Path -LiteralPath $command -PathType Leaf)) {
    throw "Desktop development command is missing: $command. Run npm install first."
}

Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
$environmentSnapshot = $null
$backendProcess = $null
$locationPushed = $false
$desktopExitCode = 1
try {
    $environmentSnapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime
    $backendProcess = Start-Process `
        -FilePath $runtime.Python `
        -ArgumentList $launchArguments `
        -WorkingDirectory $runtime.RepoRoot `
        -NoNewWindow `
        -PassThru
    Wait-DigiBoxSourceDesktopHealth `
        -BackendProcess $backendProcess `
        -HealthUrl $healthUrl `
        -InstanceId $instanceId `
        -TimeoutSeconds $StartupTimeoutSeconds

    Push-Location $runtime.RepoRoot
    $locationPushed = $true
    & $command @commandArguments
    $desktopExitCode = $LASTEXITCODE
}
finally {
    $cleanupFailure = $null
    if ($locationPushed) {
        try {
            Pop-Location
        }
        catch {
            $cleanupFailure = $_
        }
    }
    if ($null -ne $backendProcess) {
        try {
            Stop-DigiBoxSourceDesktopBackend `
                -BackendProcess $backendProcess `
                -StopFile $stopFile
        }
        catch {
            if ($null -eq $cleanupFailure) {
                $cleanupFailure = $_
            }
            else {
                Write-Warning "Additional backend stop failure: $($_.Exception.Message)"
            }
        }
        try {
            $backendProcess.Dispose()
        }
        catch {
            if ($null -eq $cleanupFailure) {
                $cleanupFailure = $_
            }
            else {
                Write-Warning "Additional process disposal failure: $($_.Exception.Message)"
            }
        }
    }
    Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
    if ($null -ne $environmentSnapshot) {
        try {
            Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $environmentSnapshot
        }
        catch {
            if ($null -eq $cleanupFailure) {
                $cleanupFailure = $_
            }
            else {
                Write-Warning "Additional environment restore failure: $($_.Exception.Message)"
            }
        }
    }
    if ($null -ne $cleanupFailure) {
        throw $cleanupFailure
    }
}
exit $desktopExitCode
