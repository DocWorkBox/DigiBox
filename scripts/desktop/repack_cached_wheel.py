"""Repack one verified, unpacked wheel cache entry without recompressing binaries."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class WheelRepackError(RuntimeError):
    """The unpacked archive is not a complete, safe wheel payload."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_record_path(value: str, root: Path) -> tuple[str, Path]:
    if not value or "\0" in value:
        raise WheelRepackError(f"unsafe RECORD path: {value!r}")
    normalized = value.replace("\\", "/")
    raw_parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", ".."} or ":" in part for part in raw_parts)
    ):
        raise WheelRepackError(f"unsafe RECORD path: {value!r}")
    relative = PurePosixPath(normalized)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise WheelRepackError(f"RECORD file is missing: {value}") from error
    if not _inside(resolved, root) or not candidate.is_file():
        raise WheelRepackError(f"RECORD path escaped the archive: {value}")
    return relative.as_posix(), candidate


def _hash_file(path: Path, algorithm: str) -> tuple[bytes, int]:
    digest = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.digest(), size


def _decode_record_digest(value: str) -> tuple[str, bytes]:
    try:
        algorithm, encoded = value.split("=", 1)
    except ValueError as error:
        raise WheelRepackError(f"invalid RECORD digest: {value!r}") from error
    algorithm = algorithm.lower()
    if algorithm not in {"sha256", "sha384", "sha512"} or not encoded:
        raise WheelRepackError(f"unsupported RECORD digest: {value!r}")
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError) as error:
        raise WheelRepackError(f"invalid RECORD digest: {value!r}") from error
    return algorithm, decoded


def _read_identity(dist_info: Path) -> tuple[str, str, str]:
    metadata_path = dist_info / "METADATA"
    wheel_path = dist_info / "WHEEL"
    record_path = dist_info / "RECORD"
    for required in (metadata_path, wheel_path, record_path):
        if not required.is_file() or required.is_symlink():
            raise WheelRepackError(f"wheel metadata file is missing or unsafe: {required.name}")

    metadata = BytesParser(policy=policy.default).parsebytes(metadata_path.read_bytes())
    name = str(metadata.get("Name", "")).strip()
    version = str(metadata.get("Version", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise WheelRepackError(f"unsafe METADATA Name: {name!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
        raise WheelRepackError(f"unsafe METADATA Version: {version!r}")

    wheel = BytesParser(policy=policy.default).parsebytes(wheel_path.read_bytes())
    tags = [str(value).strip() for value in wheel.get_all("Tag", [])]
    if len(tags) != 1 or not re.fullmatch(
        r"[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+", tags[0] if tags else ""
    ):
        raise WheelRepackError("WHEEL must contain exactly one safe Tag header")
    normalized_name = re.sub(r"[-_.]+", "_", name)
    normalized_version = version.replace("-", "_")
    return normalized_name, normalized_version, tags[0]


def _verify_archive(source: Path) -> tuple[list[tuple[str, Path]], str, bytes, str]:
    if source.is_symlink() or not source.is_dir():
        raise WheelRepackError(f"source must be one real unpacked wheel directory: {source}")
    root = source.resolve(strict=True)
    dist_infos = [path for path in root.glob("*.dist-info") if path.is_dir()]
    if len(dist_infos) != 1 or dist_infos[0].is_symlink():
        raise WheelRepackError("source must contain exactly one top-level .dist-info directory")
    name, version, tag = _read_identity(dist_infos[0])
    record_path = dist_infos[0] / "RECORD"
    record_relative = record_path.relative_to(root).as_posix()

    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WheelRepackError(f"wheel source contains a symbolic link: {path}")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path

    recorded: dict[str, Path] = {}
    verified_rows: dict[str, tuple[str, str]] = {}
    with record_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 3:
                raise WheelRepackError(f"RECORD row {row_number} does not have three fields")
            relative, candidate = _safe_record_path(row[0], root)
            if relative in recorded:
                raise WheelRepackError(f"duplicate RECORD path: {relative}")
            recorded[relative] = candidate
            digest_text, size_text = row[1], row[2]
            if relative == record_relative:
                if digest_text or size_text:
                    raise WheelRepackError("the RECORD self-entry must have empty hash and size")
                verified_rows[relative] = ("", "")
                continue
            if not digest_text or not size_text:
                raise WheelRepackError(f"RECORD hash or size is missing: {relative}")
            algorithm, expected_digest = _decode_record_digest(digest_text)
            actual_digest, actual_size = _hash_file(candidate, algorithm)
            try:
                expected_size = int(size_text)
            except ValueError as error:
                raise WheelRepackError(
                    f"invalid RECORD size for {relative}: {size_text!r}"
                ) from error
            if expected_size != actual_size or not hmac.compare_digest(
                expected_digest, actual_digest
            ):
                raise WheelRepackError(f"RECORD verification failed: {relative}")
            verified_rows[relative] = (digest_text, str(expected_size))

    if set(recorded) != set(actual):
        missing = sorted(set(actual) - set(recorded))
        extra = sorted(set(recorded) - set(actual))
        raise WheelRepackError(
            f"RECORD does not exactly describe the archive; unrecorded={missing}, missing={extra}"
        )
    wheel_name = f"{name}-{version}-{tag}.whl"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for relative in sorted(verified_rows):
        if relative != record_relative:
            writer.writerow((relative, *verified_rows[relative]))
    writer.writerow((record_relative, "", ""))
    return (
        sorted(recorded.items()),
        wheel_name,
        record_buffer.getvalue().encode("utf-8"),
        record_relative,
    )


def repack_cached_wheel(source: Path, output_directory: Path) -> dict[str, object]:
    source = source.absolute()
    output_directory = output_directory.absolute()
    source_root = source.resolve(strict=True)
    output_resolved = output_directory.resolve(strict=False)
    if _inside(output_resolved, source_root):
        raise WheelRepackError("output directory must be outside the unpacked source tree")

    # Validation deliberately completes before creating or writing any output.
    files, wheel_name, generated_record, record_relative = _verify_archive(source_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise WheelRepackError(f"output must be one real directory: {output_directory}")
    final_path = output_directory / wheel_name
    if final_path.exists():
        raise WheelRepackError(f"refusing to overwrite an existing wheel: {final_path}")
    temporary_path = output_directory / f".{wheel_name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for relative, path in files:
                info = zipfile.ZipInfo.from_file(path, arcname=relative)
                info.compress_type = zipfile.ZIP_STORED
                if relative == record_relative:
                    archive.writestr(info, generated_record, compress_type=zipfile.ZIP_STORED)
                    continue
                with (
                    path.open("rb") as source_handle,
                    archive.open(info, mode="w", force_zip64=True) as target_handle,
                ):
                    shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
        os.rename(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "wheel": str(final_path.resolve()),
        "files": len(files),
        "bytes": final_path.stat().st_size,
        "compression": "stored",
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and repack one unpacked wheel cache entry using ZIP_STORED."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = repack_cached_wheel(args.source, args.output_directory)
    except (OSError, WheelRepackError) as error:
        print(f"repack_cached_wheel: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
