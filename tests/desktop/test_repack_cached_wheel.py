from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "desktop" / "repack_cached_wheel.py"


def _record_digest(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _create_unpacked_wheel(root: Path) -> Path:
    source = root / "unpacked"
    dist_info = source / "demo_pkg-1.2.3+cuda.dist-info"
    package = source / "demo_pkg"
    package.mkdir(parents=True)
    dist_info.mkdir()
    files = {
        "demo_pkg/__init__.py": b"VALUE = 42\n",
        "demo_pkg-1.2.3+cuda.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.2.3+cuda\n\n"
        ),
        "demo_pkg-1.2.3+cuda.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-win_amd64\n\n"
        ),
    }
    for relative, data in files.items():
        target = source / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record_path = dist_info / "RECORD"
    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for relative, data in files.items():
            writer.writerow((relative, _record_digest(data), len(data)))
        writer.writerow(("demo_pkg-1.2.3+cuda.dist-info/RECORD", "", ""))
    return source


def _run(source: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_repacks_verified_archive_as_stored_zip64_wheel_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = _create_unpacked_wheel(tmp_path)
    before = {
        path.relative_to(source).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "wheelhouse"

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    wheel = output / "demo_pkg-1.2.3+cuda-py3-none-win_amd64.whl"
    assert Path(payload["wheel"]) == wheel.resolve()
    assert payload["compression"] == "stored"
    with zipfile.ZipFile(wheel) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert archive.read("demo_pkg/__init__.py") == b"VALUE = 42\n"
        assert archive.read("demo_pkg-1.2.3+cuda.dist-info/RECORD")
    after = {
        path.relative_to(source).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_normalizes_windows_cache_record_paths_in_the_output_wheel(tmp_path: Path) -> None:
    source = _create_unpacked_wheel(tmp_path)
    record = source / "demo_pkg-1.2.3+cuda.dist-info" / "RECORD"
    rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row[0] = row[0].replace("/", "\\")
    with record.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    output = tmp_path / "wheelhouse"

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    wheel = output / "demo_pkg-1.2.3+cuda-py3-none-win_amd64.whl"
    with zipfile.ZipFile(wheel) as archive:
        record_text = archive.read("demo_pkg-1.2.3+cuda.dist-info/RECORD").decode("utf-8")
        assert "\\" not in record_text
        assert "demo_pkg/__init__.py," in record_text
        assert "demo_pkg-1.2.3+cuda.dist-info/RECORD,," in record_text


@pytest.mark.parametrize("failure", ["hash", "missing", "escape", "unrecorded"])
def test_rejects_incomplete_or_unsafe_cached_archives_before_writing(
    tmp_path: Path,
    failure: str,
) -> None:
    source = _create_unpacked_wheel(tmp_path)
    record = source / "demo_pkg-1.2.3+cuda.dist-info" / "RECORD"
    rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
    if failure == "hash":
        rows[0][1] = "sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    elif failure == "missing":
        (source / "demo_pkg" / "__init__.py").unlink()
    elif failure == "escape":
        rows.insert(0, ("../escape.py", _record_digest(b"outside"), "7"))
        (tmp_path / "escape.py").write_bytes(b"outside")
    else:
        (source / "demo_pkg" / "unrecorded.py").write_text("unsafe\n", encoding="utf-8")
    if failure in {"hash", "escape"}:
        with record.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
    output = tmp_path / "wheelhouse"

    result = _run(source, output)

    assert result.returncode != 0
    assert not output.exists(), "validation must finish before the output directory is created"


def test_refuses_to_write_the_repacked_wheel_inside_the_source_tree(tmp_path: Path) -> None:
    source = _create_unpacked_wheel(tmp_path)
    result = _run(source, source / "wheelhouse")
    assert result.returncode != 0
    assert not (source / "wheelhouse").exists()
