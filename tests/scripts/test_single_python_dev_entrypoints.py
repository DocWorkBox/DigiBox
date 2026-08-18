from __future__ import annotations

import json
from pathlib import Path

from tests.scripts.test_single_python_dev_runtime import (
    REPO_ROOT,
    _quote,
    _run_powershell,
    _write_portable_v2,
)


def _plan(script: str, repo: Path, runtime: Path, arguments: str = "") -> dict:
    command = (
        f"& {_quote(REPO_ROOT / 'scripts' / script)} "
        f"-RuntimeRoot {_quote(runtime)} -RepoRoot {_quote(repo)} "
        f"-PlanOnly {arguments} | ConvertTo-Json -Depth 8 -Compress"
    )
    completed = _run_powershell(command)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_interactive_entrypoint_uses_full_python_and_routes_all_workers(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan("run_interactive_windows.ps1", repo, runtime)

    assert plan["Python"] == str(runtime / "python/python.exe")
    assert plan["Arguments"] == [str(repo / "scripts/run_local_stream.py")]
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    assert plan["Environment"]["AVTR1_COSYVOICE_PYTHON"] == plan["Python"]
    assert plan["Environment"]["AVTR1_FEYNOBG_PYTHON"] == plan["Python"]


def test_direct_worker_entrypoints_share_python_but_keep_isolated_layers(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    cosy = _plan("run_cosyvoice_windows.ps1", repo, runtime)
    fey = _plan("run_feynobg_windows.ps1", repo, runtime)

    python = str(runtime / "python/python.exe")
    assert cosy["Python"] == fey["Python"] == python
    assert cosy["Environment"]["PYTHONPATH"].split(";")[:3] == [
        str(repo / "src"),
        str(repo / "third_party/CosyVoice"),
        str(repo / "third_party/CosyVoice/third_party/Matcha-TTS"),
    ]
    assert str(runtime / "packages/main") not in cosy["Environment"]["PYTHONPATH"]
    assert fey["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    assert str(runtime / "packages/main") not in fey["Environment"]["PYTHONPATH"]


def test_test_entrypoint_uses_full_pytest_and_forwards_arguments(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan(
        "test_windows.ps1",
        repo,
        runtime,
        "-PytestArgs @('-q','tests/scripts')",
    )

    assert plan["Python"] == str(runtime / "python/python.exe")
    assert plan["Arguments"] == [
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "tests/scripts",
    ]
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    # Keep the service's default-cache contract intact for tests that construct
    # CosyVoice3Service(tmp_path), while ensuring the module-level default can
    # never fall back to the selected Full Runtime's immutable model directory.
    assert plan["Environment"]["AVTR1_COSYVOICE_SPEAKER_CACHE"] == ""
    test_model_dir = plan["Environment"]["AVTR_COSYVOICE_MODEL_DIR"]
    assert str(repo / "test-results") in test_model_dir
    assert str(runtime / "models") not in test_model_dir


def test_tauri_test_entrypoint_uses_full_python_for_python_and_rust_contracts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo with spaces"
    runtime = tmp_path / "runtime with spaces"
    _write_portable_v2(repo, runtime)
    (repo / "src-tauri").mkdir()
    (repo / "src-tauri/Cargo.toml").touch()

    plan = _plan("test_tauri_windows.ps1", repo, runtime)

    python = str(runtime / "python/python.exe")
    manifest = str(repo / "src-tauri/Cargo.toml")
    assert plan["Python"] == python
    assert plan["Environment"]["AVTR1_TEST_PYTHON"] == python
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    assert [step["Name"] for step in plan["Steps"]] == [
        "python-tauri-contracts",
        "cargo-fmt",
        "cargo-check",
        "cargo-clippy",
        "cargo-test",
    ]
    python_step = plan["Steps"][0]
    assert Path(python_step["Command"]) == repo / "scripts/test_windows.ps1"
    assert python_step["Arguments"] == [
        "-RuntimeRoot",
        str(runtime),
        "-RepoRoot",
        str(repo),
        "-PytestArgs",
        "-q",
        "--basetemp=<generated-under-test-results>",
        "tests/desktop/test_tauri_shell_contract.py",
        "tests/desktop/test_tauri_build_contract.py",
    ]
    assert plan["Steps"][1]["Arguments"] == [
        "fmt",
        "--manifest-path",
        manifest,
        "--",
        "--check",
    ]
    assert plan["Steps"][2]["Arguments"] == [
        "check",
        "--manifest-path",
        manifest,
        "--all-targets",
        "--offline",
    ]
    assert plan["Steps"][3]["Arguments"] == [
        "clippy",
        "--manifest-path",
        manifest,
        "--all-targets",
        "--offline",
        "--",
        "-D",
        "warnings",
    ]
    assert plan["Steps"][4]["Arguments"] == [
        "test",
        "--manifest-path",
        manifest,
        "--all-targets",
        "--offline",
    ]


def test_tauri_test_entrypoint_restores_the_full_runtime_environment() -> None:
    source = (REPO_ROOT / "scripts/test_tauri_windows.ps1").read_text(encoding="utf-8")

    assert ".venv" not in source
    assert '"AVTR1_TEST_PYTHON"' in source
    assert "Set-DigiBoxSourceRuntimeEnvironment" in source
    assert "Restore-DigiBoxSourceRuntimeEnvironment" in source
    assert source.index("try {") < source.index("finally {")
    assert source.index("finally {") < source.index("Restore-DigiBoxSourceRuntimeEnvironment")


def test_package_tauri_tests_use_the_unified_full_runtime_entrypoint() -> None:
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    command = package["scripts"]["test:tauri"]

    assert "test_tauri_windows.ps1" in command
    assert "test_tauri_shell_contract.py" not in command
    assert "test_tauri_build_contract.py" not in command
    assert ".venv" not in command


def test_offline_source_entrypoint_uses_the_full_main_profile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan("run_offline_windows.ps1", repo, runtime, "-Backend tensorrt")

    assert plan["Python"] == str(runtime / "python/python.exe")
    assert plan["Arguments"][:3] == [
        str(repo / "scripts/generate_offline.py"),
        "--backend",
        "tensorrt",
    ]
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")


def test_active_source_entrypoints_do_not_reference_legacy_venvs() -> None:
    for name in (
        "run_interactive_windows.ps1",
        "run_cosyvoice_windows.ps1",
        "run_feynobg_windows.ps1",
        "run_offline_windows.ps1",
        "test_windows.ps1",
    ):
        source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert ".venv" not in source


def test_setup_entrypoints_validate_the_full_runtime_without_recreating_venvs() -> None:
    for name in (
        "setup_windows.ps1",
        "setup_cosyvoice_windows.ps1",
        "setup_feynobg_windows.ps1",
    ):
        source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "dev_runtime_windows.ps1" in source
        assert ".venv" not in source
        assert " uv venv " not in f" {source.lower()} "
        assert "pip install" not in source.lower()


def test_cosyvoice_setup_repairs_the_selected_full_model_directory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan("setup_cosyvoice_windows.ps1", repo, runtime)

    model_dir = runtime / "models/Fun-CosyVoice3-0.5B-2512"
    assert plan["Environment"]["AVTR_COSYVOICE_MODEL_DIR"] == str(model_dir)
    assert str(repo / "models") not in json.dumps(plan)


def test_hugging_face_login_uses_the_full_main_profile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan("login_huggingface_windows.ps1", repo, runtime)

    assert plan["Python"] == str(runtime / "python/python.exe")
    assert plan["Arguments"] == [str(repo / "scripts/login_huggingface_windows.py")]
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")

    cmd = (REPO_ROOT / "login_huggingface_windows.cmd").read_text(encoding="utf-8")
    assert ".venv" not in cmd
    assert "login_huggingface_windows.ps1" in cmd


def test_tensorrt_source_builder_uses_full_python_and_repo_scripts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    _write_portable_v2(repo, runtime)

    plan = _plan("build_tensorrt_windows.ps1", repo, runtime)

    assert plan["Python"] == str(runtime / "python/python.exe")
    assert plan["Environment"]["PYTHONPATH"].split(";")[0] == str(repo / "src")
    assert plan["Steps"][:4] == [
        "diagnostics",
        "download-artifacts",
        "build-avtr1",
        "build-hubert",
    ]
    source = (REPO_ROOT / "scripts/build_tensorrt_windows.ps1").read_text(encoding="utf-8")
    assert ".venv" not in source
    assert "-Python $runtime.Python" in source
    warp_source = (REPO_ROOT / "scripts/build_warp_plugin_windows.ps1").read_text(encoding="utf-8")
    assert ".venv" not in warp_source
    assert "Pass -Python" in warp_source
