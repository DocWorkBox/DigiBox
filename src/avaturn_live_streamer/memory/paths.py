from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryPaths:
    root: Path
    database: Path
    backups: Path


def _absolute_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        return None
    return path.resolve()


def _is_within(candidate: Path, parent: Path) -> bool:
    return candidate == parent or candidate.is_relative_to(parent)


def resolve_memory_paths(
    environment: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> MemoryPaths | None:
    """Resolve a persistent LocalAppData root or disable memory safely.

    An explicit but unsafe ``AVTR1_MEMORY_ROOT`` is treated as a configuration
    error; it never falls through to a different location silently.
    """

    env = os.environ if environment is None else environment
    explicit = env.get("AVTR1_MEMORY_ROOT")
    if explicit is not None and explicit.strip():
        root = _absolute_path(explicit)
        if root is None:
            return None
    else:
        local_app_data = _absolute_path(env.get("LOCALAPPDATA"))
        if local_app_data is None:
            return None
        root = (local_app_data / "DigiBox" / "memory").resolve()

    unsafe_roots = [(cwd or Path.cwd()).resolve()]
    for key in ("AVTR1_RUNTIME_ROOT", "AVTR1_APP_ROOT"):
        unsafe = _absolute_path(env.get(key))
        if unsafe is not None:
            unsafe_roots.append(unsafe)
    if any(_is_within(root, unsafe) for unsafe in unsafe_roots):
        return None

    return MemoryPaths(
        root=root,
        database=root / "memory.sqlite3",
        backups=root / "backups",
    )
