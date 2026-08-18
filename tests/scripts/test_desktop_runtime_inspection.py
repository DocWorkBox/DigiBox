from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "desktop" / "inspect_runtime.py"


def _load_inspector():
    spec = importlib.util.spec_from_file_location("desktop_runtime_inspector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_engine_paths_are_inspected_without_using_active_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inspector = _load_inspector()
    staged_engine = tmp_path / "staging" / "decoder.engine"
    staged_engine.parent.mkdir(parents=True)
    staged_engine.write_bytes(b"staged-plan")
    probed: list[Path] = []

    monkeypatch.setattr(inspector, "_version", lambda _name: ("test", None))
    monkeypatch.setattr(inspector, "_gpu_status", lambda: {"available": True})

    def fake_probe(paths: list[Path]):
        probed.extend(paths)
        return {str(path): {"ok": True} for path in paths}

    monkeypatch.setattr(inspector, "_probe_engines", fake_probe)

    report = inspector.inspect(
        tmp_path,
        probe_engines=True,
        engine_paths=[staged_engine],
    )

    assert probed == [staged_engine.resolve()]
    assert report["engine_files"] == [str(staged_engine.resolve())]
    assert report["engine_probe"][str(staged_engine.resolve())]["ok"] is True


def test_explicit_engine_paths_reject_missing_files(tmp_path: Path) -> None:
    inspector = _load_inspector()
    missing = tmp_path / "missing.engine"

    try:
        inspector.inspect(
            tmp_path,
            probe_engines=True,
            engine_paths=[missing],
        )
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing explicit engine path was accepted")


def test_inspection_reports_the_portable_manifest_layout(tmp_path: Path, monkeypatch) -> None:
    inspector = _load_inspector()
    (tmp_path / "runtime-manifest.json").write_text(
        '{"schemaVersion":2,"layout":"portable-v2"}', encoding="utf-8"
    )
    monkeypatch.setattr(inspector, "_version", lambda _name: ("test", None))
    monkeypatch.setattr(inspector, "_gpu_status", lambda: {"available": False})

    report = inspector.inspect(tmp_path, probe_engines=False)

    assert report["runtime_manifest"] == {
        "schema_version": 2,
        "layout": "portable-v2",
    }
