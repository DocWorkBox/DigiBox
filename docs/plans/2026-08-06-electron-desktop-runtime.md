# AVTR-1 Electron Desktop Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Package the existing native Windows AVTR-1 web UI and Python/TensorRT service stack as a secure Electron desktop application, then add a relocatable Python runtime staging path and a first-run TensorRT compatibility/build assistant.

**Architecture:** Electron owns exactly one Python orchestrator and renders the existing localhost UI in a locked-down Chromium window. The application may attach to an already healthy external AVTR service during development, but only stops a backend process that it started. Large models and machine-specific TensorRT engines stay outside ASAR. Distribution uses a portable application runtime assembled from clean standalone CPython installations, not copied virtual environments; first run validates GPU/runtime/engine compatibility and offers an explicit TensorRT build action when required.

**Tech Stack:** Electron 43.3.0, Node.js 22+, electron-builder 26.15.3, CommonJS, built-in `node:test`, PowerShell 7/Windows PowerShell, CPython 3.12 and 3.10 runtimes, FastAPI/Uvicorn, TensorRT 10.11/CUDA 12.8.

---

## Constraints and decisions

- Work in the current `windows-native` checkout because the desktop app must integrate the user's extensive uncommitted Windows implementation; creating a clean worktree would omit the product being packaged.
- Preserve every pre-existing file and local model. No cleanup, reset, model replacement, commit, or push is part of this execution.
- Do not copy or rename `.venv`, `.venv-cosyvoice`, or `.venv-feynobg` into a release. Their `pyvenv.cfg` files contain absolute paths and are not relocatable.
- Keep models, user avatars, local voices, logs, and TensorRT engines outside ASAR and outside the program install directory when possible.
- A packaged engine is only accepted after an on-device compatibility/deserialization probe. Otherwise the UI offers a build using the bundled runtime and portable ONNX sources.
- The first deliverable is runnable on this machine and contains the complete bootstrap/staging mechanism. Producing a redistributable full offline runtime is conditional on downloading clean standalone Python archives and all redistributable third-party assets.

## Task 1: Establish RED desktop unit tests

**Files:**

- Create: `desktop/test/runtime-paths.test.cjs`
- Create: `desktop/test/health.test.cjs`
- Create: `desktop/test/navigation.test.cjs`
- Create: `desktop/test/backend-supervisor.test.cjs`
- Create: `desktop/test/runtime-manifest.test.cjs`

**Tests:**

- Resolve development, packaged, environment-variable, and explicit runtime roots.
- Report missing required runtime components without mutating the machine.
- Poll health until success, timeout, process exit, or cancellation.
- Allow only exact AVTR localhost navigation and microphone permission; reject unknown origins.
- Attach to healthy external services without taking ownership; spawn a single orchestrator when absent; only stop owned processes.
- Validate a portable-runtime manifest and classify TensorRT engines as ready, rebuild-required, or unavailable.

**RED command:** `node --test desktop/test/*.test.cjs`

## Task 2: Implement pure desktop runtime libraries

**Files:**

- Create: `desktop/lib/runtime-paths.cjs`
- Create: `desktop/lib/health.cjs`
- Create: `desktop/lib/navigation.cjs`
- Create: `desktop/lib/backend-supervisor.cjs`
- Create: `desktop/lib/runtime-manifest.cjs`

**Implementation:** Keep Electron out of these modules and inject filesystem, fetch, spawn, clock, and process-tree termination dependencies so tests stay deterministic and offline.

**GREEN command:** `node --test desktop/test/*.test.cjs`

## Task 3: Add cooperative Python health and shutdown

**Files:**

- Modify: `scripts/run_local_stream.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Create or modify: `tests/test_run_local_stream.py`
- Modify: `tests/test_local_stream_cli.py`

**Tests:**

- A desktop stop-file or `shutdown` stdin command interrupts renderer warm-up and the main monitor loop.
- `/health` returns an AVTR-specific JSON identity only after the streamer app is constructed.
- Optional FeyNoBg/CosyVoice exits are surfaced as degraded state without corrupting required service ownership.

**Verification:** Run focused pytest with a unique `--basetemp` under `test-results`.

## Task 4: Implement the secure Electron shell

**Files:**

- Create: `package.json`
- Create: `desktop/main.cjs`
- Create: `desktop/preload.cjs`
- Create: `desktop/splash.html`
- Create: `desktop/splash.css`
- Create: `desktop/splash.js`
- Modify: `.gitignore`

**Implementation:**

- Show a local startup/diagnostic page immediately, then load `http://127.0.0.1:7860` after identity health checks pass.
- Use one BrowserWindow with `nodeIntegration:false`, `contextIsolation:true`, `sandbox:true`, `webSecurity:true`, `backgroundThrottling:false` and a minimal IPC API.
- Enforce a single instance, exact-origin navigation, denied popups, allowlisted external HTTPS links, and exact-origin media permission.
- Capture backend logs under Electron `userData`; retry startup and reveal logs/runtime folder from the splash screen.
- On quit, request cooperative shutdown, wait, and force-stop only the owned process tree as a bounded fallback.

## Task 5: Build the relocatable Python runtime and first-run TensorRT assistant

**Files:**

- Create: `scripts/desktop/build_portable_runtime.ps1`
- Create: `scripts/desktop/install_runtime_dependencies.ps1`
- Create: `scripts/desktop/inspect_runtime.py`
- Create: `scripts/desktop/build_tensorrt.ps1`
- Create: `desktop/runtime-manifest.example.json`
- Create: `docs/windows-desktop-distribution.md`

**Implementation:**

- Stage clean standalone CPython 3.12 and 3.10 distributions into `runtime/python-*`; install packages with `pip --target`/prefix-based scripts so no `pyvenv.cfg` or host interpreter path is retained.
- Install AVTR source as a regular wheel or add its staged source directory through a runtime-owned `PYTHONPATH`; never use an editable install in the release.
- Generate a manifest with Python/package/CUDA/TensorRT/model versions and file hashes.
- Inspect NVIDIA driver, GPU name/compute capability, TensorRT importability, required ONNX/model inputs, engine files, and an actual deserialize/smoke result.
- Offer an explicit build command with progress/logging and safe output paths; never overwrite a different engine silently.
- Support `runtimeMode` values `managed`, `development`, and `external` so the same desktop app can run the staged package or the current checkout.

## Task 6: Package Windows deliverables

**Files:**

- Create: `electron-builder.yml`
- Generate: `package-lock.json`
- Create: `scripts/build_desktop_windows.ps1`

**Implementation:**

- ASAR-pack desktop JS/HTML only; include licenses and notices.
- Produce `win-unpacked` for real-machine QA and an assisted per-user NSIS installer.
- Keep the optional multi-gigabyte runtime/model payload external or in a separate offline bundle so shell upgrades do not reinstall all assets.
- Installer/uninstaller must retain user avatars, voices, API settings, models, logs, and engines unless the user explicitly removes them.

## Task 7: Verify end to end

1. Run all desktop Node tests.
2. Run focused Python shutdown/health/runtime tests.
3. Build `win-unpacked`, launch the real Electron executable, and verify splash-to-AVTR transition.
4. Confirm microphone permission, WebRTC page rendering, themes/avatar display, and persisted settings.
5. Quit the desktop app and verify ports 7860, 8000, 8767, and 8768 are released when the app owns them.
6. Relaunch and verify single-instance behavior and clean service ownership.
7. Run the existing Python test suite with an isolated basetemp, then `git diff --check` on touched files.
