from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "desktop" / "build_tensorrt.ps1"


def test_tensorrt_helper_reads_portable_v2_python_and_main_layers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"portable-v2"' in source
    assert '"python\\python.exe"' in source
    assert "packageLayers.main" in source
    assert "AVTR1_MAIN_PYTHONPATH" in source
    assert 'Join-Path $root "python-main\\python.exe"' in source
    assert "schemaVersion" in source
    assert "IsPathRooted" in source
    assert "escapes the Runtime root" in source
    assert 'Join-Path $root "runtime-manifest.json"' in source
    assert 'Join-Path $root "python\\python.exe"' in source


@pytest.mark.parametrize(
    ("stub_body", "expect_success", "expected_error"),
    [
        (
            """
            Write-Output ""
            Write-Output "CMake progress is not a result path"
            New-Item -ItemType Directory -Force `
                -Path (Split-Path -Parent $OutputPath) | Out-Null
            [System.IO.File]::WriteAllBytes($OutputPath, [byte[]](1, 2, 3))
            """,
            True,
            "",
        ),
        (
            """
            Write-Output ""
            throw "sentinel native failure"
            """,
            False,
            "sentinel native failure",
        ),
        (
            """
            Write-Output ""
            Write-Output "CMake completed without the requested DLL"
            """,
            False,
            "returned successfully but the expected DLL is missing",
        ),
    ],
)
def test_warp_plugin_build_uses_an_explicit_output_contract(
    tmp_path: Path,
    stub_body: str,
    expect_success: bool,
    expected_error: str,
) -> None:
    plugin_script = tmp_path / "plugin-stub.ps1"
    plugin_script.write_text(
        textwrap.dedent(
            f"""
            param(
                [string]$Python = "",
                [string]$BuildRoot = "",
                [string]$OutputPath = ""
            )
            $ErrorActionPreference = "Stop"
            {stub_body}
            """
        ).strip(),
        encoding="utf-8",
    )
    runner = tmp_path / "invoke-warp-helper.ps1"
    runner.write_text(
        textwrap.dedent(
            r"""
            param(
                [Parameter(Mandatory = $true)][string]$BuildScript,
                [Parameter(Mandatory = $true)][string]$PluginScript,
                [Parameter(Mandatory = $true)][string]$BuildRoot,
                [Parameter(Mandatory = $true)][string]$OutputPath
            )
            $ErrorActionPreference = "Stop"
            $tokens = $null
            $parseErrors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile(
                $BuildScript,
                [ref]$tokens,
                [ref]$parseErrors
            )
            if ($parseErrors.Count -gt 0) {
                throw "Build script failed to parse: $($parseErrors -join ', ')"
            }
            $functionAst = $ast.Find(
                {
                    param($node)
                    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq "Invoke-WarpPluginBuild"
                },
                $true
            )
            if ($null -eq $functionAst) {
                throw "Invoke-WarpPluginBuild is missing"
            }
            Invoke-Expression $functionAst.Extent.Text

            $errorMessage = ""
            try {
                Invoke-WarpPluginBuild `
                    -ScriptPath $PluginScript `
                    -Python "python.exe" `
                    -BuildRoot $BuildRoot `
                    -OutputPath $OutputPath
            }
            catch {
                $errorMessage = $_.Exception.Message
            }
            [ordered]@{
                error = $errorMessage
                outputExists = Test-Path -LiteralPath $OutputPath -PathType Leaf
            } | ConvertTo-Json -Compress
            """
        ).strip(),
        encoding="utf-8",
    )
    build_root = tmp_path / "stage with spaces" / "warp-plugin"
    output_path = tmp_path / "stage with spaces" / "renderer" / "grid_sample_3d_plugin.dll"

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-BuildScript",
            str(SCRIPT),
            "-PluginScript",
            str(plugin_script),
            "-BuildRoot",
            str(build_root),
            "-OutputPath",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(
        next(line for line in reversed(result.stdout.splitlines()) if line.startswith("{"))
    )
    if expect_success:
        assert payload["outputExists"] is True
        assert payload["error"] == ""
    else:
        assert payload["outputExists"] is False
        assert expected_error in payload["error"]


def test_full_install_accepts_an_empty_retire_target_collection(tmp_path: Path) -> None:
    source = tmp_path / "staging" / "engine.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new-engine")
    target = tmp_path / "active with spaces" / "engine.bin"
    runner = tmp_path / "invoke-install-helper.ps1"
    runner.write_text(
        textwrap.dedent(
            r"""
            param(
                [Parameter(Mandatory = $true)][string]$BuildScript,
                [Parameter(Mandatory = $true)][string]$Source,
                [Parameter(Mandatory = $true)][string]$Target
            )
            $ErrorActionPreference = "Stop"
            $tokens = $null
            $parseErrors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile(
                $BuildScript,
                [ref]$tokens,
                [ref]$parseErrors
            )
            if ($parseErrors.Count -gt 0) {
                throw "Build script failed to parse: $($parseErrors -join ', ')"
            }
            $functionAst = $ast.Find(
                {
                    param($node)
                    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq "Install-ActiveArtifacts"
                },
                $true
            )
            if ($null -eq $functionAst) {
                throw "Install-ActiveArtifacts is missing"
            }
            Invoke-Expression $functionAst.Extent.Text
            $entries = @(
                [pscustomobject]@{
                    Source = $Source
                    Target = $Target
                }
            )
            Install-ActiveArtifacts `
                -InstallEntries $entries `
                -RetireTargets @() `
                -Timestamp "empty-retire-contract"
            """
        ).strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-BuildScript",
            str(SCRIPT),
            "-Source",
            str(source),
            "-Target",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert target.read_bytes() == b"new-engine"
    assert not list(target.parent.glob("*.incoming.*"))


@pytest.mark.parametrize("rollback_fails", [False, True])
@pytest.mark.parametrize("warning_preference_stops", [False, True])
def test_install_transaction_preserves_install_and_rollback_failures(
    tmp_path: Path,
    rollback_fails: bool,
    warning_preference_stops: bool,
) -> None:
    runner = tmp_path / "invoke-install-transaction.ps1"
    runner.write_text(
        textwrap.dedent(
            r"""
            param(
                [Parameter(Mandatory = $true)][string]$BuildScript,
                [Parameter(Mandatory = $true)][int]$RollbackFails,
                [Parameter(Mandatory = $true)][int]$WarningPreferenceStops
            )
            $ErrorActionPreference = "Stop"
            $tokens = $null
            $parseErrors = $null
            $ast = [System.Management.Automation.Language.Parser]::ParseFile(
                $BuildScript,
                [ref]$tokens,
                [ref]$parseErrors
            )
            if ($parseErrors.Count -gt 0) {
                throw "Build script failed to parse: $($parseErrors -join ', ')"
            }
            $functionAst = $ast.Find(
                {
                    param($node)
                    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $node.Name -eq "Invoke-ArtifactInstallTransaction"
                },
                $true
            )
            if ($null -eq $functionAst) {
                throw "Invoke-ArtifactInstallTransaction is missing"
            }
            Invoke-Expression $functionAst.Extent.Text
            $script:rollbackCalled = $false
            function Restore-ActiveArtifacts {
                param([Parameter(Mandatory = $true)][object]$BackupState)
                $script:rollbackCalled = $true
                if ($RollbackFails -ne 0) {
                    throw "rollback-sentinel"
                }
            }

            if ($WarningPreferenceStops -ne 0) {
                $WarningPreference = "Stop"
            }

            $errorMessage = ""
            $errorType = ""
            $innerMessages = @()
            try {
                Invoke-ArtifactInstallTransaction `
                    -BackupState ([pscustomobject]@{}) `
                    -Action { throw "install-sentinel" }
            }
            catch {
                $errorMessage = $_.Exception.Message
                $errorType = $_.Exception.GetType().FullName
                if ($_.Exception -is [System.AggregateException]) {
                    $innerMessages = @(
                        $_.Exception.InnerExceptions | ForEach-Object { $_.Message }
                    )
                }
            }
            [ordered]@{
                error = $errorMessage
                errorType = $errorType
                innerMessages = $innerMessages
                rollbackCalled = $script:rollbackCalled
            } | ConvertTo-Json -Compress
            """
        ).strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            "-BuildScript",
            str(SCRIPT),
            "-RollbackFails",
            str(int(rollback_fails)),
            "-WarningPreferenceStops",
            str(int(warning_preference_stops)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(
        next(line for line in reversed(result.stdout.splitlines()) if line.startswith("{"))
    )
    assert payload["rollbackCalled"] is True
    assert "install-sentinel" in payload["error"]
    if rollback_fails:
        assert "rollback-sentinel" in payload["error"]
        assert payload["errorType"] == "System.AggregateException"
        assert payload["innerMessages"] == ["install-sentinel", "rollback-sentinel"]
        assert "Rollback completed" not in result.stdout
    else:
        assert "rollback-sentinel" not in payload["error"]
        assert payload["errorType"] != "System.AggregateException"
        assert payload["innerMessages"] == []
        assert "Rollback completed" in result.stdout
