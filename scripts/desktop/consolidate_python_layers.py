#!/usr/bin/env python3
"""Conservatively move provably identical profile packages into a shared layer."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path, PurePosixPath

PROFILES = ("main", "cosyvoice", "feynobg")
INVENTORY_NAME = "python-layer-inventory.json"
SCRIPT_ROOTS = frozenset(("bin", "scripts"))
SITE_BOOTSTRAP = '''"""Activate DigiBox portable package layers loaded through PYTHONPATH."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_AVTR1_DLL_HANDLES = []
_PACKAGES_ROOT = Path(__file__).resolve().parent.parent
_LAYER_NAMES = frozenset(("main", "cosyvoice", "feynobg", "shared"))
_LAYER_DIRS = []
_seen_layers = set()
for _raw_path in tuple(sys.path):
    try:
        _candidate = Path(_raw_path).resolve()
    except (OSError, RuntimeError):
        continue
    if (
        os.path.normcase(str(_candidate.parent)) == os.path.normcase(str(_PACKAGES_ROOT))
        and _candidate.name in _LAYER_NAMES
        and _candidate.is_dir()
    ):
        _key = os.path.normcase(str(_candidate))
        if _key not in _seen_layers:
            _seen_layers.add(_key)
            _LAYER_DIRS.append(_candidate)

_shared = _PACKAGES_ROOT / "shared"
if _shared.is_dir() and os.path.normcase(str(_shared)) not in _seen_layers:
    _LAYER_DIRS.append(_shared)

_seen_dll_dirs = set()
for _layer in _LAYER_DIRS:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _dll_candidates = [
            _layer / "torch" / "lib",
            _layer / "onnxruntime" / "capi",
            _layer / "tensorrt_libs",
        ]
        _dll_candidates.extend(sorted(_layer.glob("*.libs")))
        _dll_candidates.extend(sorted(_layer.glob("nvidia/*/bin")))
        for _dll_dir in _dll_candidates:
            if not _dll_dir.is_dir():
                continue
            _dll_key = os.path.normcase(str(_dll_dir.resolve()))
            if _dll_key in _seen_dll_dirs:
                continue
            _seen_dll_dirs.add(_dll_key)
            try:
                _AVTR1_DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
            except OSError:
                pass
    # PYTHONPATH entries do not process .pth files by themselves. DLL
    # directories are activated first in case a .pth import loads native code.
    site.addsitedir(str(_layer))
'''


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _relative_record_path(raw: str) -> str | None:
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or not candidate.parts:
        return None
    if any(part in ("", ".", "..") for part in candidate.parts):
        return None
    return candidate.as_posix()


@dataclass
class DistributionCopy:
    profile: str
    layer: Path
    name: str
    normalized_name: str
    version: str
    dist_info: Path
    files: tuple[str, ...] = ()
    hashes: tuple[tuple[str, str], ...] = ()
    roots: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def byte_size(self) -> int:
        return sum((self.layer / relative).stat().st_size for relative in self.files)


def _metadata_identity(dist_info: Path) -> tuple[str, str] | None:
    metadata_path = dist_info / "METADATA"
    if not metadata_path.is_file():
        return None
    metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    name = (metadata.get("Name") or "").strip()
    version = (metadata.get("Version") or "").strip()
    if not name or not version:
        return None
    return name, version


def _import_root(value: str) -> str | None:
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or len(candidate.parts) != 1:
        return None
    root = candidate.parts[0].strip().lower()
    if not root or root in (".", ".."):
        return None
    if root.endswith(".py"):
        root = root[:-3]
    elif root.endswith(".pyd"):
        root = root.split(".", maxsplit=1)[0]
    return root or None


def _top_level_hints(copy: DistributionCopy) -> tuple[str, ...]:
    top_level = copy.dist_info / "top_level.txt"
    if not top_level.is_file():
        return ()
    try:
        hints = {_import_root(line.strip()) for line in top_level.read_text("utf-8").splitlines()}
    except (OSError, UnicodeError):
        return ()
    return tuple(sorted(hint for hint in hints if hint is not None))


def _read_record(copy: DistributionCopy) -> None:
    copy.roots = _top_level_hints(copy)
    record = copy.dist_info / "RECORD"
    if not record.is_file():
        copy.reason = "missing-record"
        return
    relative_files: list[str] = []
    try:
        with record.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                relative = _relative_record_path(row[0])
                if relative is None:
                    copy.reason = "unsafe-record-path"
                    return
                target = copy.layer / Path(*PurePosixPath(relative).parts)
                try:
                    target.resolve(strict=True).relative_to(copy.layer.resolve(strict=True))
                except (FileNotFoundError, ValueError):
                    copy.reason = "missing-record-file"
                    return
                if not target.is_file():
                    copy.reason = "missing-record-file"
                    return
                relative_files.append(relative)
    except (csv.Error, OSError, UnicodeError):
        copy.reason = "invalid-record"
        return
    if not relative_files or len(set(relative_files)) != len(relative_files):
        copy.reason = "invalid-record"
        return

    owned = set(relative_files)
    dist_info_relative = copy.dist_info.relative_to(copy.layer).as_posix()
    top_level_entries = {
        PurePosixPath(relative).parts[0]
        for relative in relative_files
        if PurePosixPath(relative).parts[0] != dist_info_relative
        and ".data" not in PurePosixPath(relative).parts[0]
    }
    if not top_level_entries:
        copy.reason = "no-import-payload"
        return
    import_entries = {
        entry for entry in top_level_entries if entry.casefold() not in SCRIPT_ROOTS
    }
    actual_roots = {_import_root(entry) for entry in import_entries}
    copy.roots = tuple(
        sorted(set(copy.roots) | {root for root in actual_roots if root is not None})
    )
    copy.files = tuple(sorted(owned))
    copy.hashes = tuple((relative, _file_digest(copy.layer / relative)) for relative in copy.files)
    for entry in sorted(import_entries):
        target = copy.layer / entry
        if target.is_dir():
            init_file = f"{entry}/__init__.py"
            if init_file not in owned or not (target / "__init__.py").is_file():
                copy.reason = "namespace-package"
                return
            actual_files = {
                item.relative_to(copy.layer).as_posix()
                for item in target.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
            }
            if not actual_files.issubset(owned):
                copy.reason = "unowned-package-file"
                return
        elif not target.is_file() or entry not in owned:
            copy.reason = "unowned-package-file"
            return
    actual_metadata_files = {
        item.relative_to(copy.layer).as_posix()
        for item in copy.dist_info.rglob("*")
        if item.is_file()
    }
    if not actual_metadata_files.issubset(owned):
        copy.reason = "unowned-package-file"
        return


def _discover(packages_root: Path) -> list[DistributionCopy]:
    copies: list[DistributionCopy] = []
    for profile in PROFILES:
        layer = packages_root / profile
        layer.mkdir(parents=True, exist_ok=True)
        for dist_info in sorted(layer.glob("*.dist-info"), key=lambda path: path.name.lower()):
            identity = _metadata_identity(dist_info)
            if identity is None:
                continue
            name, version = identity
            copy = DistributionCopy(
                profile=profile,
                layer=layer,
                name=name,
                normalized_name=_normalize_name(name),
                version=version,
                dist_info=dist_info,
            )
            _read_record(copy)
            copies.append(copy)
    return copies


def _group_reasons(copies: list[DistributionCopy]) -> dict[str, str | None]:
    by_name: dict[str, list[DistributionCopy]] = {}
    for copy in copies:
        by_name.setdefault(copy.normalized_name, []).append(copy)

    owners: dict[tuple[str, str], set[str]] = {}
    root_owners: dict[tuple[str, str], set[str]] = {}
    for copy in copies:
        for relative in copy.files:
            owners.setdefault((copy.profile, relative.lower()), set()).add(copy.normalized_name)
        for root in copy.roots:
            root_owners.setdefault((copy.profile, root), set()).add(copy.normalized_name)

    reasons: dict[str, str | None] = {}
    for normalized_name, group in sorted(by_name.items()):
        per_profile = {
            profile: [copy for copy in group if copy.profile == profile] for profile in PROFILES
        }
        if any(len(per_profile[profile]) != 1 for profile in PROFILES):
            reasons[normalized_name] = "not-in-all-profiles"
            continue
        ordered = [per_profile[profile][0] for profile in PROFILES]
        existing_reason = next((copy.reason for copy in ordered if copy.reason is not None), None)
        if existing_reason is not None:
            reasons[normalized_name] = existing_reason
            continue
        if len({copy.version for copy in ordered}) != 1:
            reasons[normalized_name] = "version-mismatch"
            continue
        if any(
            len(owners.get((copy.profile, relative.lower()), set())) != 1
            for copy in ordered
            for relative in copy.files
        ) or any(
            len(root_owners.get((copy.profile, root), set())) != 1
            for copy in ordered
            for root in copy.roots
        ):
            reasons[normalized_name] = "path-conflict"
            continue
        if any(
            copy.files != ordered[0].files or copy.hashes != ordered[0].hashes
            for copy in ordered[1:]
        ):
            reasons[normalized_name] = "content-mismatch"
            continue
        reasons[normalized_name] = None
    return reasons


def _copy_to_shared(copy: DistributionCopy, shared: Path) -> None:
    for relative in copy.files:
        source = copy.layer / relative
        target = shared / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"shared layer path collision: {relative}")
        shutil.copy2(source, target)


def _remove_copy(copy: DistributionCopy) -> None:
    layer = copy.layer.resolve()
    parents: set[Path] = set()
    for relative in sorted(copy.files, key=lambda value: (value.count("/"), value), reverse=True):
        target = copy.layer / relative
        parents.update(target.parents)
        target.unlink()
    for directory in sorted(parents, key=lambda path: len(path.parts), reverse=True):
        if directory == layer or layer not in directory.parents:
            continue
        with contextlib.suppress(OSError):
            directory.rmdir()


def _write_site_bootstrap(packages_root: Path) -> None:
    for profile in PROFILES:
        collision = packages_root / profile / "sitecustomize.py"
        if collision.exists():
            raise RuntimeError(
                f"profile layer owns reserved bootstrap path: {profile}/sitecustomize.py"
            )
    target = packages_root / "shared" / "sitecustomize.py"
    if target.exists() and target.read_text("utf-8") != SITE_BOOTSTRAP:
        raise RuntimeError("shared layer owns reserved bootstrap path: shared/sitecustomize.py")
    target.write_text(SITE_BOOTSTRAP, encoding="utf-8", newline="\n")


def consolidate(runtime_root: Path) -> dict[str, object]:
    packages_root = runtime_root.resolve() / "packages"
    shared = packages_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        (packages_root / profile).mkdir(parents=True, exist_ok=True)
    _write_site_bootstrap(packages_root)

    inventory_path = packages_root / INVENTORY_NAME
    if inventory_path.exists():
        inventory_path.unlink()
    before_bytes = sum(_tree_size(packages_root / name) for name in (*PROFILES, "shared"))
    copies = _discover(packages_root)
    reasons = _group_reasons(copies)
    by_name: dict[str, list[DistributionCopy]] = {}
    for copy in copies:
        by_name.setdefault(copy.normalized_name, []).append(copy)

    distributions: list[dict[str, object]] = []
    shared_count = 0
    for normalized_name, group in sorted(by_name.items()):
        reason = reasons[normalized_name]
        if reason is None:
            ordered = sorted(group, key=lambda copy: PROFILES.index(copy.profile))
            canonical = ordered[0]
            canonical_bytes = canonical.byte_size
            _copy_to_shared(canonical, shared)
            for copy in ordered:
                _remove_copy(copy)
            distributions.append(
                {
                    "bytes": canonical_bytes,
                    "fileCount": len(canonical.files),
                    "layer": "shared",
                    "name": normalized_name,
                    "profiles": list(PROFILES),
                    "version": canonical.version,
                }
            )
            shared_count += 1
            continue
        for copy in sorted(group, key=lambda item: (item.profile, item.version)):
            distributions.append(
                {
                    "bytes": copy.byte_size if copy.files else _tree_size(copy.dist_info),
                    "fileCount": len(copy.files),
                    "layer": copy.profile,
                    "name": normalized_name,
                    "profile": copy.profile,
                    "reason": reason,
                    "version": copy.version,
                }
            )

    distributions.sort(
        key=lambda item: (str(item["name"]), str(item["layer"]), str(item.get("profile", "")))
    )
    after_bytes = sum(_tree_size(packages_root / name) for name in (*PROFILES, "shared"))
    inventory: dict[str, object] = {
        "distributions": distributions,
        "layers": {
            name: {
                "bytes": _tree_size(packages_root / name),
                "path": f"packages/{name}",
            }
            for name in (*PROFILES, "shared")
        },
        "profiles": list(PROFILES),
        "schemaVersion": 1,
        "summary": {
            "afterBytes": after_bytes,
            "beforeBytes": before_bytes,
            "retainedDistributionCopies": len(distributions) - shared_count,
            "savedBytes": before_bytes - after_bytes,
            "sharedDistributions": shared_count,
        },
    }
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    inventory = consolidate(args.runtime_root)
    print(json.dumps(inventory["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
