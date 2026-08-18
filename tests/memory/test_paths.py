from __future__ import annotations

from pathlib import Path

from avaturn_live_streamer.memory.paths import MemoryPaths, resolve_memory_paths


def test_explicit_absolute_root_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "safe-memory"

    result = resolve_memory_paths(
        {
            "AVTR1_MEMORY_ROOT": str(explicit),
            "LOCALAPPDATA": str(tmp_path / "ignored-local-app-data"),
        },
        cwd=tmp_path / "source",
    )

    assert result == MemoryPaths(
        root=explicit.resolve(),
        database=(explicit / "memory.sqlite3").resolve(),
        backups=(explicit / "backups").resolve(),
    )


def test_local_app_data_fallback_uses_digibox_memory(tmp_path: Path) -> None:
    local_app_data = tmp_path / "Local"

    result = resolve_memory_paths(
        {"LOCALAPPDATA": str(local_app_data)},
        cwd=tmp_path / "source",
    )

    assert result is not None
    assert result.root == (local_app_data / "DigiBox" / "memory").resolve()
    assert result.database == result.root / "memory.sqlite3"
    assert result.backups == result.root / "backups"


def test_relative_explicit_root_disables_memory(tmp_path: Path) -> None:
    result = resolve_memory_paths(
        {
            "AVTR1_MEMORY_ROOT": "relative-memory",
            "LOCALAPPDATA": str(tmp_path / "Local"),
        },
        cwd=tmp_path,
    )

    assert result is None


def test_root_inside_runtime_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"

    result = resolve_memory_paths(
        {
            "AVTR1_MEMORY_ROOT": str(runtime / "memory"),
            "AVTR1_RUNTIME_ROOT": str(runtime),
        },
        cwd=tmp_path / "source",
    )

    assert result is None


def test_root_inside_current_working_directory_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"

    result = resolve_memory_paths(
        {"AVTR1_MEMORY_ROOT": str(source / ".memory")},
        cwd=source,
    )

    assert result is None


def test_roaming_app_data_is_not_a_fallback(tmp_path: Path) -> None:
    result = resolve_memory_paths(
        {"APPDATA": str(tmp_path / "Roaming")},
        cwd=tmp_path / "source",
    )

    assert result is None
