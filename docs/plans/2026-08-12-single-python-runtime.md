# DigiBox Single Python Runtime Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` and implement each behavior with test-driven development.

**Goal:** Replace the three duplicated standalone CPython installations with one portable Python 3.12 base plus isolated Main, CosyVoice, and FeyNoBg dependency layers, while preserving the existing multi-process architecture and `portable-v1` rollback compatibility.

**Architecture:** The Full runtime uses `python/python.exe` as its only CPython distribution. Packages are installed into three profile directories, then byte-identical distributions are moved into a shared layer. Each process receives an explicit, ordered `PYTHONPATH`; profile-specific packages shadow the shared layer, so incompatible dependency versions never share one flat `site-packages`. Consumers accept both legacy `portable-v1` and the new `portable-v2` manifest during migration.

**Tech Stack:** PowerShell 5.1, uv, CPython 3.12, Python package metadata/RECORD files, Node.js contract tests, pytest, Rust/Tauri v2.

---

### Task 1: Define the portable-v2 layout and compatibility contract

**Files:**
- Modify: `desktop/runtime-manifest.example.json`
- Modify: `desktop/lib/runtime-manifest.cjs`
- Modify: `src-tauri/src/runtime.rs`
- Test: `desktop/test/runtime-manifest.test.cjs`
- Test: `src-tauri/tests/runtime_contract.rs`

1. Add failing tests for `portable-v2`, its single Python entrypoint, ordered package layers, and continued `portable-v1` acceptance.
2. Run the Node and Rust tests and verify the new assertions fail because only `portable-v1` is understood.
3. Implement strict v2 parsing and path validation without relaxing v1 checks.
4. Run the focused suites to green.

### Task 2: Route per-process package layers

**Files:**
- Modify: `scripts/run_local_stream.py`
- Modify: `desktop/main.cjs`
- Modify: `src-tauri/src/supervisor.rs`
- Test: `tests/scripts/test_run_local_stream_windows.py`
- Test: `desktop/test/backend-supervisor.test.cjs`
- Test: `src-tauri/tests/supervisor_contract.rs`

1. Add failing tests that Main, CosyVoice, and FeyNoBg all use the same executable while receiving different ordered `PYTHONPATH` values.
2. Add `AVTR1_COSYVOICE_PYTHONPATH` and `AVTR1_FEYNOBG_PYTHONPATH` handling to worker launches.
3. Make desktop supervisors derive the main and worker paths from the v2 manifest; keep v1 directory inference.
4. Verify focused Python, Node, and Rust tests.

### Task 3: Build one Python base and isolated package profiles

**Files:**
- Modify: `scripts/desktop/build_portable_runtime.ps1`
- Create: `scripts/desktop/consolidate_python_layers.py`
- Test: `desktop/test/portable-runtime-build.test.cjs`
- Create: `tests/desktop/test_python_layer_consolidation.py`

1. Add RED contracts for one Python installation, three target package layers, v2 manifest output, and deterministic consolidation.
2. Install each dependency set into its own target directory using the same Python 3.12 base.
3. Consolidate only distributions whose normalized name, version, RECORD ownership, and file bytes are identical. Leave namespace/file collisions in their profiles.
4. Write size and package inventories into the manifest and verify no absolute staging paths leak into it.

### Task 4: Make CosyVoice's Windows runtime Python 3.12 compatible

**Files:**
- Modify: `requirements-windows-cosyvoice.txt`
- Test: `tests/integrations/test_cosyvoice_server_contract.py`
- Test: `desktop/test/portable-runtime-build.test.cjs`

1. Add RED assertions for Python 3.12 and a Windows cp312-compatible PyWORLD pin.
2. Change `pyworld==0.3.4` to `pyworld==0.3.5`; retain the other pins initially.
3. Use uv dry-run to verify the complete Windows dependency set resolves on CPython 3.12.
4. Run syntax/import contracts for the actual CosyVoice inference adapter.

### Task 5: Update Standard/Full validation and documentation

**Files:**
- Modify: `scripts/build_tauri_windows.ps1`
- Modify: `scripts/build_desktop_windows.ps1`
- Modify: `tests/desktop/test_tauri_build_contract.py`
- Modify: `docs/windows-tauri-desktop-distribution.md`
- Modify: `docs/windows-desktop-distribution.md`

1. Add failing validation tests for both layouts and v2 single-runtime paths.
2. Teach Full packaging validation and privacy scans about `python/` and `packages/`.
3. Preserve v1 acceptance for existing external runtimes and document the rollback path.
4. Run all desktop build contracts.

### Task 6: End-to-end verification

1. Run all focused Python, Node, and Rust suites.
2. Build a small plan-only Full layout and inspect its manifest.
3. Build a disposable Python 3.12 CosyVoice profile and verify imports, model loading, cloned-voice lookup, streaming startup/cancel, and release behavior where local assets permit.
4. Compare v1 and v2 unpacked sizes and report actual savings; do not replace or delete existing Runtime or installer artifacts.
