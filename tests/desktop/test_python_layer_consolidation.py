from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATOR = ROOT / "scripts" / "desktop" / "consolidate_python_layers.py"
PROFILES = ("main", "cosyvoice", "feynobg")


def _write_distribution(
    layer: Path,
    *,
    name: str,
    version: str,
    files: dict[str, bytes],
    record: bool = True,
    top_level: tuple[str, ...] | None = None,
) -> None:
    layer.mkdir(parents=True, exist_ok=True)
    normalized = name.replace("-", "_")
    dist_info = layer / f"{normalized}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    (dist_info / "METADATA").write_text(metadata, encoding="utf-8")
    if top_level is not None:
        (dist_info / "top_level.txt").write_text("\n".join(top_level) + "\n", encoding="utf-8")
    for relative, payload in files.items():
        target = layer / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if record:
        rows = sorted(files)
        rows.append(f"{dist_info.name}/METADATA")
        if top_level is not None:
            rows.append(f"{dist_info.name}/top_level.txt")
        rows.append(f"{dist_info.name}/RECORD")
        with (dist_info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for relative in rows:
                writer.writerow((relative, "", ""))


def _fixture(root: Path) -> None:
    packages = root / "packages"
    for profile in PROFILES:
        layer = packages / profile
        _write_distribution(
            layer,
            name="identical-lib",
            version="1.2.3",
            files={"identical_lib/__init__.py": b"VALUE = 1\n"},
            top_level=("identical_lib",),
        )
        _write_distribution(
            layer,
            name="console-script-lib",
            version="1.0.0",
            files={
                "console_script_lib/__init__.py": b"VALUE = 1\n",
                "bin/torchrun.exe": b"launcher",
            },
            top_level=("console_script_lib",),
        )
        _write_distribution(
            layer,
            name="namespace-lib",
            version="2.0.0",
            files={"namespace_lib/plugin.py": b"PLUGIN = True\n"},
            top_level=("namespace_lib",),
        )
        _write_distribution(
            layer,
            name="recordless-lib",
            version="3.0.0",
            files={"recordless_lib.py": b"VALUE = 3\n"},
            record=False,
            top_level=("recordless_lib",),
        )
        _write_distribution(
            layer,
            name="conflicted-lib",
            version="4.0.0",
            files={"conflicted_lib/__init__.py": b"VALUE = 4\n"},
            top_level=("conflicted_lib",),
        )
        _write_distribution(
            layer,
            name="hinted-valid-lib",
            version="4.1.0",
            files={"hinted_pkg/__init__.py": b"VALUE = 41\n"},
            top_level=("hinted_pkg",),
        )
        # No RECORD means ownership cannot be proven. Its top_level hint must
        # still prevent another distribution from moving the same import root.
        _write_distribution(
            layer,
            name="hinted-recordless-lib",
            version="4.1.0",
            files={"hinted_pkg/__init__.py": b"VALUE = 41\n"},
            record=False,
            top_level=("hinted_pkg",),
        )
    # Same version, different bytes: never share it.
    for index, profile in enumerate(PROFILES):
        _write_distribution(
            packages / profile,
            name="divergent-lib",
            version="5.0.0",
            files={"divergent_lib/__init__.py": f"VALUE = {index}\n".encode()},
            top_level=("divergent_lib",),
        )
    # Same name, mismatched version: never share it.
    for index, profile in enumerate(PROFILES):
        _write_distribution(
            packages / profile,
            name="versioned-lib",
            version="6.0.0" if index < 2 else "6.0.1",
            files={"versioned_lib/__init__.py": b"VALUE = 6\n"},
            top_level=("versioned_lib",),
        )
    # A second distribution claiming the same path is an ownership conflict.
    _write_distribution(
        packages / "main",
        name="path-collider",
        version="1.0.0",
        files={"conflicted_lib/__init__.py": b"VALUE = 4\n"},
        top_level=("conflicted_lib",),
    )


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(CONSOLIDATOR), "--runtime-root", str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_only_record_owned_byte_identical_non_conflicting_distributions_are_shared(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    _fixture(runtime)

    result = _run(runtime)

    assert result.returncode == 0, result.stdout + result.stderr
    shared = runtime / "packages" / "shared"
    assert (shared / "identical_lib" / "__init__.py").read_bytes() == b"VALUE = 1\n"
    assert (shared / "identical_lib-1.2.3.dist-info" / "METADATA").is_file()
    assert (shared / "console_script_lib" / "__init__.py").read_bytes() == b"VALUE = 1\n"
    assert (shared / "bin" / "torchrun.exe").read_bytes() == b"launcher"
    for profile in PROFILES:
        layer = runtime / "packages" / profile
        assert not (layer / "identical_lib").exists()
        assert not (layer / "identical_lib-1.2.3.dist-info").exists()
        assert not (layer / "console_script_lib").exists()
        assert not (layer / "bin" / "torchrun.exe").exists()
        assert (layer / "namespace_lib" / "plugin.py").is_file()
        assert (layer / "recordless_lib.py").is_file()
        assert (layer / "divergent_lib" / "__init__.py").is_file()
        assert (layer / "versioned_lib" / "__init__.py").is_file()
        assert (layer / "conflicted_lib" / "__init__.py").is_file()
        assert (layer / "hinted_pkg" / "__init__.py").is_file()

    inventory = json.loads(
        (runtime / "packages" / "python-layer-inventory.json").read_text("utf-8")
    )
    assert inventory["schemaVersion"] == 1
    assert inventory["profiles"] == list(PROFILES)
    assert inventory["summary"]["sharedDistributions"] == 2
    assert inventory["summary"]["savedBytes"] > 0
    identical = next(item for item in inventory["distributions"] if item["name"] == "identical-lib")
    assert identical["layer"] == "shared"
    assert identical["profiles"] == list(PROFILES)
    retained_reasons = {
        item["name"]: item["reason"]
        for item in inventory["distributions"]
        if item["layer"] != "shared" and item["profile"] == "main"
    }
    assert retained_reasons["namespace-lib"] == "namespace-package"
    assert retained_reasons["recordless-lib"] == "missing-record"
    assert retained_reasons["divergent-lib"] == "content-mismatch"
    assert retained_reasons["versioned-lib"] == "version-mismatch"
    assert retained_reasons["conflicted-lib"] == "path-conflict"
    assert retained_reasons["hinted-valid-lib"] == "path-conflict"


def test_inventory_is_byte_deterministic_and_contains_no_absolute_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _fixture(first)
    _fixture(second)

    first_result = _run(first)
    second_result = _run(second)

    assert first_result.returncode == second_result.returncode == 0
    first_inventory = (first / "packages" / "python-layer-inventory.json").read_bytes()
    second_inventory = (second / "packages" / "python-layer-inventory.json").read_bytes()
    assert first_inventory == second_inventory
    assert str(first).encode() not in first_inventory
    payload = json.loads(first_inventory)
    assert payload["summary"]["beforeBytes"] >= payload["summary"]["afterBytes"]
    assert (
        payload["summary"]["beforeBytes"] - payload["summary"]["afterBytes"]
        == payload["summary"]["savedBytes"]
    )


def test_shared_sitecustomize_processes_layer_pth_and_keeps_windows_dll_handles(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    main = runtime / "packages" / "main"
    shared = runtime / "packages" / "shared"
    probe_root = main / "pth_payload"
    probe_root.mkdir(parents=True)
    (probe_root / "portable_layer_probe.py").write_text("VALUE = 73\n", encoding="utf-8")
    (main / "dll_order_probe.py").write_text(
        (
            "import os, sitecustomize\n"
            "if os.name == 'nt':\n"
            "    assert sitecustomize._AVTR1_DLL_HANDLES\n"
        ),
        encoding="utf-8",
    )
    (main / "portable-layer-probe.pth").write_text(
        "pth_payload\nimport dll_order_probe\n", encoding="utf-8"
    )
    (main / "torch" / "lib").mkdir(parents=True)

    result = _run(runtime)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (shared / "sitecustomize.py").is_file()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(main), str(shared)))
    env["PYTHONNOUSERSITE"] = "1"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import portable_layer_probe, sitecustomize; "
                "print(portable_layer_probe.VALUE); "
                "print(len(sitecustomize._AVTR1_DLL_HANDLES))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stderr == ""
    output = probe.stdout.splitlines()
    assert output[0] == "73"
    if os.name == "nt":
        assert int(output[1]) >= 1
