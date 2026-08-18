from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from tests.scripts.test_single_python_dev_runtime import (
    REPO_ROOT,
    _quote,
    _run_powershell,
    _write_portable_v2,
)

WRAPPER = REPO_ROOT / "scripts" / "desktop_dev_windows.ps1"


def _plan(shell: str, repo: Path, runtime: Path) -> dict[str, object]:
    command = (
        f"& {_quote(WRAPPER)} -Shell {shell} "
        f"-RepoRoot {_quote(repo)} -RuntimeRoot {_quote(runtime)} -PlanOnly "
        "| ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("shell", "executable", "arguments"),
    [
        ("Tauri", "tauri.cmd", ["dev"]),
        ("Electron", "electron.cmd", ["."]),
    ],
)
def test_desktop_plan_starts_the_shell_over_the_source_full_backend(
    tmp_path: Path,
    shell: str,
    executable: str,
    arguments: list[str],
) -> None:
    repo = tmp_path / "repo with spaces"
    runtime = tmp_path / "runtime with spaces"
    _write_portable_v2(repo, runtime)

    plan = _plan(shell, repo, runtime)

    python = str(runtime / "python/python.exe")
    assert plan["Shell"] == shell
    assert plan["Python"] == python
    assert plan["BackendArguments"] == [str(repo / "scripts/run_local_stream.py")]
    assert plan["LaunchArguments"] == [f'"{repo / "scripts/run_local_stream.py"}"']
    assert Path(plan["Command"]) == repo / "node_modules/.bin" / executable
    assert plan["CommandArguments"] == arguments
    assert plan["HealthUrl"] == "http://127.0.0.1:7860/health"
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    assert plan["Environment"]["AVTR1_COSYVOICE_PYTHON"] == python
    assert plan["Environment"]["AVTR1_FEYNOBG_PYTHON"] == python
    assert plan["Environment"]["AVTR1_DESKTOP_RUNTIME"] == str(runtime)
    assert plan["Environment"]["AVTR1_DESKTOP_STOP_FILE"].endswith(".stop")


@pytest.mark.parametrize("port", [7860, 8000, 8767, 8768])
def test_desktop_dev_refuses_to_attach_to_an_existing_backend_port(
    tmp_path: Path, port: int
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            pytest.skip(f"port {port} was already unavailable: {error}")
        listener.listen()

        command = (
            f"& {_quote(WRAPPER)} -Shell Tauri "
            f"-RepoRoot {_quote(repo)} -RuntimeRoot {_quote(runtime)}"
        )
        completed = _run_powershell(command)
    finally:
        listener.close()

    assert completed.returncode != 0
    message = completed.stdout + completed.stderr
    assert str(port) in message
    assert "already" in message.lower()


def test_desktop_wrapper_has_owned_cooperative_and_forced_cleanup() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert '"AVTR1_DESKTOP_STOP_FILE"' in source
    assert "Start-Process" in source
    assert "WaitForExit" in source
    assert '"/PID"' in source
    assert '"/T"' in source
    assert '"/F"' in source
    assert "$backendProcess.Id" in source
    assert "Get-NetTCPConnection" not in source

    cleanup = source[source.rindex("\nfinally {") :]
    assert "$cleanupFailure = $null" in cleanup
    assert "$cleanupFailure = $_" in cleanup
    assert cleanup.index("Stop-DigiBoxSourceDesktopBackend") < cleanup.index(
        "$backendProcess.Dispose()"
    )
    assert cleanup.index("$backendProcess.Dispose()") < cleanup.index(
        "Restore-DigiBoxSourceRuntimeEnvironment"
    )
    assert cleanup.index("Restore-DigiBoxSourceRuntimeEnvironment") < cleanup.index(
        "throw $cleanupFailure"
    )


def test_package_desktop_commands_use_the_source_bridge_and_full_test_runtime() -> None:
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    scripts = package["scripts"]

    assert "desktop_dev_windows.ps1 -Shell Tauri" in scripts["tauri:dev"]
    assert "desktop_dev_windows.ps1 -Shell Electron" in scripts["desktop:dev"]
    assert "test_tauri_windows.ps1" in scripts["test:tauri"]
    assert "test_tauri_shell_contract.py" not in scripts["test:tauri"]
    assert "test_tauri_build_contract.py" not in scripts["test:tauri"]
    assert ".venv" not in scripts["test:tauri"]
