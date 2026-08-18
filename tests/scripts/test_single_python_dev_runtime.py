from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_HELPER = REPO_ROOT / "scripts" / "dev_runtime_windows.ps1"
TEST_ENTRYPOINT = REPO_ROOT / "scripts" / "test_windows.ps1"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    assert executable, "Windows PowerShell is required"
    return executable


def _quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_powershell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


def _write_portable_v2(repo: Path, runtime: Path) -> None:
    for relative in (
        "src",
        "third_party/CosyVoice/cosyvoice",
        "third_party/CosyVoice/third_party/Matcha-TTS",
    ):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "python",
        "packages/main",
        "packages/cosyvoice",
        "packages/feynobg",
        "packages/shared",
        "src",
        "artifacts/main",
        "models/Fun-CosyVoice3-0.5B-2512",
    ):
        (runtime / relative).mkdir(parents=True, exist_ok=True)
    (runtime / "python/python.exe").touch()
    (runtime / "models/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml").touch()
    manifest = {
        "schemaVersion": 2,
        "layout": "portable-v2",
        "runtimeId": "test-runtime",
        "paths": {
            "python": "python/python.exe",
            "orchestrator": "scripts/run_local_stream.py",
            "source": "src",
            "artifacts": "artifacts/main",
            "models": "models",
        },
        "python": {
            "version": "3.12.9",
            "packageLayers": {
                "main": ["packages/main", "packages/shared", "src"],
                "cosyvoice": [
                    "packages/cosyvoice",
                    "packages/shared",
                    "third_party/CosyVoice",
                    "third_party/CosyVoice/third_party/Matcha-TTS",
                    "src",
                ],
                "feynobg": ["packages/feynobg", "packages/shared", "src"],
            },
        },
        "components": {
            "dependenciesIncluded": True,
            "modelsIncluded": True,
            "frontendVendorIncluded": True,
            "tensorRtBuildInputsIncluded": True,
        },
        "tensorrt": {"engines": []},
    }
    (runtime / "runtime-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _resolve(repo: Path, runtime: Path) -> dict[str, object]:
    command = (
        f". {_quote(RUNTIME_HELPER)}; "
        f"Resolve-DigiBoxSourceRuntime -RepoRoot {_quote(repo)} "
        f"-RuntimeRoot {_quote(runtime)} | ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_source_runtime_uses_one_full_python_and_repo_sources_first(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo with spaces"
    runtime = tmp_path / "runtime with spaces"
    _write_portable_v2(repo, runtime)

    resolved = _resolve(repo, runtime)

    assert Path(resolved["Python"]) == runtime / "python/python.exe"
    paths = resolved["PythonPath"]
    main = paths["Main"].split(";")
    cosyvoice = paths["CosyVoice"].split(";")
    feynobg = paths["FeyNoBg"].split(";")
    assert main == [
        str(repo / "src"),
        str(runtime / "packages/main"),
        str(runtime / "packages/shared"),
    ]
    assert cosyvoice == [
        str(repo / "src"),
        str(repo / "third_party/CosyVoice"),
        str(repo / "third_party/CosyVoice/third_party/Matcha-TTS"),
        str(runtime / "packages/cosyvoice"),
        str(runtime / "packages/shared"),
    ]
    assert feynobg == [
        str(repo / "src"),
        str(runtime / "packages/feynobg"),
        str(runtime / "packages/shared"),
    ]
    assert str(runtime / "src") not in main + cosyvoice + feynobg


def test_source_runtime_environment_routes_every_worker_to_the_same_python(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    resolved = _resolve(repo, runtime)
    environment = resolved["Environment"]
    python = str(runtime / "python/python.exe")
    assert environment["AVTR1_RUNTIME_ROOT"] == str(runtime)
    assert environment["AVTR1_APP_ROOT"] == str(repo)
    assert environment["AVTR1_LOCAL_STORAGE"] == str(runtime / "artifacts")
    assert environment["AVTR1_MODELS_ROOT"] == str(runtime / "models")
    assert environment["AVTR1_COSYVOICE_PYTHON"] == python
    assert environment["AVTR1_FEYNOBG_PYTHON"] == python
    assert environment["AVTR1_SINGLE_ENV"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPATH"] == resolved["PythonPath"]["Main"]
    assert environment["AVTR1_MAIN_PYTHONPATH"] == resolved["PythonPath"]["Main"]
    assert environment["AVTR1_COSYVOICE_PYTHONPATH"] == resolved["PythonPath"]["CosyVoice"]
    assert environment["AVTR1_FEYNOBG_PYTHONPATH"] == resolved["PythonPath"]["FeyNoBg"]


def test_source_runtime_routes_memory_to_explicit_local_app_data(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    local_app_data = tmp_path / "Local AppData"
    _write_portable_v2(repo, runtime)

    command = (
        f"$env:LOCALAPPDATA = {_quote(local_app_data)}; "
        f". {_quote(RUNTIME_HELPER)}; "
        f"Resolve-DigiBoxSourceRuntime -RepoRoot {_quote(repo)} "
        f"-RuntimeRoot {_quote(runtime)} | ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    environment = json.loads(completed.stdout)["Environment"]
    memory_root = Path(environment["AVTR1_MEMORY_ROOT"])
    assert memory_root == local_app_data / "DigiBox/memory"
    assert not memory_root.is_relative_to(runtime)


def test_source_runtime_disables_memory_for_a_relative_local_app_data_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    command = (
        "$env:LOCALAPPDATA = 'relative-local-data'; "
        "$env:AVTR1_MEMORY_ROOT = 'D:\\stale-memory'; "
        f". {_quote(RUNTIME_HELPER)}; "
        f"$runtime = Resolve-DigiBoxSourceRuntime -RepoRoot {_quote(repo)} "
        f"-RuntimeRoot {_quote(runtime)}; "
        "$snapshot = Set-DigiBoxSourceRuntimeEnvironment -Runtime $runtime; "
        "try { [pscustomobject]@{ "
        "HasPlannedMemory = $runtime.Environment.Contains('AVTR1_MEMORY_ROOT'); "
        "ActiveMemory = [Environment]::GetEnvironmentVariable('AVTR1_MEMORY_ROOT', 'Process') "
        "} | ConvertTo-Json -Compress } finally { "
        "Restore-DigiBoxSourceRuntimeEnvironment -Snapshot $snapshot }"
    )
    completed = _run_powershell(command)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result == {"HasPlannedMemory": False, "ActiveMemory": None}


def test_windows_test_entrypoint_isolates_memory_under_test_results(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    command = (
        f"& {_quote(TEST_ENTRYPOINT)} -RepoRoot {_quote(repo)} "
        f"-RuntimeRoot {_quote(runtime)} -PlanOnly "
        "| ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    environment = json.loads(completed.stdout)["Environment"]
    memory_root = Path(environment["AVTR1_MEMORY_ROOT"])
    assert memory_root.name == "memory"
    assert memory_root.is_relative_to(repo / "test-results")
    assert not memory_root.is_relative_to(runtime)


def test_source_runtime_discovers_the_newest_full_build_without_a_parameter(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    older = repo / "desktop/builds/20260101-old/dist/DigiBox-Full-win64/avtr-runtime"
    newer = repo / "desktop/builds/20260102-new/dist/DigiBox-Full-win64/avtr-runtime"
    _write_portable_v2(repo, older)
    _write_portable_v2(repo, newer)

    command = (
        f". {_quote(RUNTIME_HELPER)}; "
        f"Resolve-DigiBoxSourceRuntime -RepoRoot {_quote(repo)} "
        "| ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert Path(json.loads(completed.stdout)["RuntimeRoot"]) == newer


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda manifest: manifest.update(layout="portable-v1"), "portable-v2"),
        (
            lambda manifest: manifest["python"]["packageLayers"]["main"].append("../escape"),
            "escapes",
        ),
    ],
)
def test_source_runtime_rejects_incompatible_or_escaping_manifests(
    tmp_path: Path, mutate, expected: str
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)
    manifest_path = runtime / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    command = (
        f". {_quote(RUNTIME_HELPER)}; "
        f"Resolve-DigiBoxSourceRuntime -RepoRoot {_quote(repo)} "
        f"-RuntimeRoot {_quote(runtime)}"
    )
    completed = _run_powershell(command)

    assert completed.returncode != 0
    assert expected.lower() in (completed.stderr + completed.stdout).lower()


def test_source_runtime_rejects_an_occupied_owned_port() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        command = (
            f". {_quote(RUNTIME_HELPER)}; Assert-DigiBoxDevelopmentPortsAvailable -Ports @({port})"
        )
        completed = _run_powershell(command)

    assert completed.returncode != 0
    assert str(port) in (completed.stderr + completed.stdout)
