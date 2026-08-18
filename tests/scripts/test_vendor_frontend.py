from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SCRIPT = REPOSITORY_ROOT / "scripts" / "desktop" / "vendor_frontend.mjs"


def _write_fake_package(
    root: Path,
    package: str,
    version: str,
    files: dict[str, bytes],
) -> None:
    package_root = root / "node_modules" / package
    package_root.mkdir(parents=True)
    (package_root / "package.json").write_text(
        json.dumps({"name": package, "version": version}),
        encoding="utf-8",
    )
    for relative_path, payload in files.items():
        destination = package_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def test_desktop_package_pins_frontend_vendor_versions_and_command() -> None:
    package = json.loads((REPOSITORY_ROOT / "package.json").read_text(encoding="utf-8"))

    assert package["devDependencies"]["preact"] == "10.22.0"
    assert package["devDependencies"]["htm"] == "3.1.1"
    assert package["scripts"]["vendor:frontend"] == (
        "node scripts/desktop/vendor_frontend.mjs"
    )


def test_vendor_script_copies_only_the_pinned_browser_modules(tmp_path: Path) -> None:
    _write_fake_package(
        tmp_path,
        "preact",
        "10.22.0",
        {
            "dist/preact.module.js": b"preact-module",
            "hooks/dist/hooks.module.js": b"preact-hooks-module",
        },
    )
    _write_fake_package(
        tmp_path,
        "htm",
        "3.1.1",
        {"dist/htm.module.js": b"htm-module"},
    )

    completed = subprocess.run(
        ["node", str(VENDOR_SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    vendor = tmp_path / "src" / "avaturn_live_streamer" / "vendor"
    assert (vendor / "preact.module.js").read_bytes() == b"preact-module"
    assert (vendor / "preact-hooks.module.js").read_bytes() == b"preact-hooks-module"
    assert (vendor / "htm.module.js").read_bytes() == b"htm-module"


def test_vendor_script_rejects_an_installed_version_drift(tmp_path: Path) -> None:
    _write_fake_package(
        tmp_path,
        "preact",
        "10.23.0",
        {
            "dist/preact.module.js": b"wrong-preact",
            "hooks/dist/hooks.module.js": b"wrong-hooks",
        },
    )
    _write_fake_package(
        tmp_path,
        "htm",
        "3.1.1",
        {"dist/htm.module.js": b"htm-module"},
    )

    completed = subprocess.run(
        ["node", str(VENDOR_SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode != 0
    assert "preact must be exactly 10.22.0" in completed.stderr
    assert not (
        tmp_path / "src" / "avaturn_live_streamer" / "vendor"
    ).exists()
