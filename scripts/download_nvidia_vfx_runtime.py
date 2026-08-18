"""Reliably download and verify NVIDIA's large Windows VFX runtime wheel."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = (
    "https://pypi.nvidia.com/nvidia-vfx/"
    "nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl"
)
FILENAME = "nvidia_vfx-0.1.0.1-cp312-abi3-win_amd64.whl"
EXPECTED_SIZE = 490_396_952
EXPECTED_SHA256 = "b6cfaff5f435ad18329a1e1c1ac3ceb36f2aa6cfb0774d271c0bcc3aeaf31c53"
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024
DEFAULT_WORKERS = 8
_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_part(
    *,
    index: int,
    start: int,
    end: int,
    part_count: int,
    parts_dir: Path,
) -> Path:
    expected = end - start + 1
    final_path = parts_dir / f"part-{index:04d}.bin"
    partial_path = parts_dir / f"part-{index:04d}.partial"
    if final_path.is_file() and final_path.stat().st_size == expected:
        return final_path

    for attempt in range(1, 13):
        current = partial_path.stat().st_size if partial_path.exists() else 0
        if current > expected:
            partial_path.unlink()
            current = 0
        if current == expected:
            os.replace(partial_path, final_path)
            return final_path

        request_start = start + current
        request = urllib.request.Request(
            URL,
            headers={"Range": f"bytes={request_start}-{end}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {request_start}-{end}/"
                if response.status != 206 or not content_range.startswith(expected_prefix):
                    raise RuntimeError(
                        f"unexpected range response {response.status}: {content_range!r}"
                    )
                with partial_path.open("ab") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if partial_path.stat().st_size != expected:
                raise RuntimeError(
                    f"part {index} has {partial_path.stat().st_size} bytes; "
                    f"expected {expected}"
                )
            os.replace(partial_path, final_path)
            _log(f"completed part {index + 1}/{part_count}")
            return final_path
        except Exception as exc:
            if attempt == 12:
                raise RuntimeError(
                    f"part {index} failed after {attempt} attempts: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def download(destination: Path, *, workers: int, chunk_size: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == EXPECTED_SIZE
        and _sha256(destination) == EXPECTED_SHA256
    ):
        _log(f"already verified {destination}")
        return destination

    parts_dir = destination.parent / "nvidia-vfx-verified-parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(EXPECTED_SIZE - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, EXPECTED_SIZE, chunk_size))
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _download_part,
                index=index,
                start=start,
                end=end,
                part_count=len(ranges),
                parts_dir=parts_dir,
            )
            for index, start, end in ranges
        ]
        for future in as_completed(futures):
            future.result()

    temporary = destination.with_suffix(destination.suffix + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index, _start, _end in ranges:
            part = parts_dir / f"part-{index:04d}.bin"
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    actual_sha256 = digest.hexdigest()
    if temporary.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(
            f"assembled size {temporary.stat().st_size} != {EXPECTED_SIZE}"
        )
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"SHA256 mismatch: {actual_sha256}")
    os.replace(temporary, destination)
    shutil.rmtree(parts_dir)
    _log(
        f"verified {destination} "
        f"({EXPECTED_SIZE} bytes, sha256={actual_sha256})"
    )
    return destination


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=root / ".downloads" / FILENAME,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        parser.error("workers and chunk-size must be positive")
    download(args.destination.resolve(), workers=args.workers, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
