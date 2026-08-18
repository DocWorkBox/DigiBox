# DigiBox Tauri v2 Desktop Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver DigiBox as a production-ready Tauri v2 Windows shell while preserving the current native AVTR-1 Python/CUDA runtime, local web UI, TensorRT assistant, user data, and Electron fallback until migration acceptance.

**Architecture:** Tauri owns one Python orchestrator process and loads the existing AVTR loopback UI only after an identity health check succeeds. The Rust core resolves either the development checkout, an explicitly selected Runtime, or a packaged `avtr-runtime` resource; it owns only processes it starts, requests cooperative shutdown through `AVTR1_DESKTOP_STOP_FILE`, and uses a Windows Job Object plus a bounded fallback for child-tree cleanup. The loopback application receives no Tauri IPC capability; only the bundled splash can call tightly scoped Rust commands.

**Tech Stack:** Tauri v2, Rust 1.93+, WebView2, Tokio, Reqwest, Windows Job Objects, standalone CPython 3.12/3.10, FastAPI/Uvicorn, WebRTC, PowerShell, NSIS, TensorRT 10.11/CUDA 12.8.

---

## Constraints and decisions

- Work in the current `windows-native` checkout because the product and existing Electron/runtime implementation contain extensive uncommitted work that a clean worktree would omit.
- Preserve the existing Electron files until Tauri passes real Windows microphone/WebRTC/startup/shutdown acceptance. Do not delete, reset, commit, push, or clean user files.
- Reuse `scripts/desktop/build_portable_runtime.ps1`, `inspect_runtime.py`, `build_tensorrt.ps1`, the runtime manifest, model layout, privacy exclusions, and licenses.
- Do not package CUDA/Python with PyInstaller. Ship the three standalone Python runtimes as resources and let Rust launch `python-main/python.exe scripts/run_local_stream.py`.
- Do not expose shell execution or filesystem capabilities to `http://127.0.0.1:7860`. Rust owns backend and helper process execution.
- Keep mutable avatars, voices, API settings, logs, and TensorRT engines outside a per-machine Program Files installation. Standard uses an external Runtime; Full remains portable Zip64/unpacked unless a separate large-payload installer is introduced.
- A TensorRT plan remains target-machine specific and is accepted only after the existing compatibility/deserialization probe.
- No Git commit or push is part of this migration unless the user explicitly asks.

### Task 1: Establish the Tauri project and RED contracts

**Files:**

- Create: `src-tauri/Cargo.toml`
- Create: `src-tauri/build.rs`
- Create: `src-tauri/src/lib.rs`
- Create: `src-tauri/tests/runtime_contract.rs`
- Create: `src-tauri/tests/navigation_contract.rs`
- Create: `src-tauri/tests/supervisor_contract.rs`

**Steps:**

1. Add only the minimum Cargo/config scaffold required to collect tests.
2. Write failing integration tests for Runtime candidate precedence, managed/development layouts, exact AVTR health identity, navigation allowlisting, external-service ownership, owned-process shutdown ordering, and stop-file environment construction.
3. Run `cargo test --manifest-path src-tauri/Cargo.toml --tests` and record failures caused by missing production modules.
4. Implement only enough pure Rust modules to satisfy each contract, rerunning tests after every behavior.

### Task 2: Implement safe Runtime resolution and health polling

**Files:**

- Create: `src-tauri/src/runtime.rs`
- Create: `src-tauri/src/health.rs`
- Create: `src-tauri/src/navigation.rs`
- Extend: `src-tauri/tests/runtime_contract.rs`
- Extend: `src-tauri/tests/navigation_contract.rs`

**Behaviors:**

- Resolve explicit CLI, `AVTR1_DESKTOP_RUNTIME`, persisted, packaged resource, executable sibling, and development roots in deterministic order.
- Prefer `python-main/python.exe`; permit `.venv/Scripts/python.exe` only in development mode.
- Require `scripts/run_local_stream.py`, `src`, and `artifacts/main`; validate `runtime-manifest.json` when present.
- Probe only `http://127.0.0.1:7860/health`, require AVTR identity, and support cancellation, timeout, and early child exit.
- Permit in-app navigation only for the bundled splash and exact `127.0.0.1:7860`; open allowlisted HTTPS links externally and deny all other schemes/origins.

### Task 3: Implement the Windows backend supervisor

**Files:**

- Create: `src-tauri/src/supervisor.rs`
- Create: `src-tauri/src/windows_job.rs`
- Extend: `src-tauri/tests/supervisor_contract.rs`

**Behaviors:**

- Attach to an already healthy external service without taking ownership.
- Otherwise spawn exactly one orchestrator with the existing desktop environment variables and independent CosyVoice/FeyNoBg interpreters.
- Capture stdout/stderr into the Tauri user log directory and emit bounded startup progress.
- Assign the owned process to a kill-on-close Windows Job Object before considering startup successful.
- On exit: write the stop file, wait up to 20 seconds, close/terminate the Job Object, then use an exact-PID `taskkill /T /F` fallback only if necessary.
- Never stop a backend not started by this desktop process.

### Task 4: Build the secure Tauri application shell

**Files:**

- Create: `src-tauri/src/app.rs`
- Create: `src-tauri/src/main.rs`
- Create: `src-tauri/tauri.conf.json`
- Create: `src-tauri/capabilities/default.json`
- Create: `desktop/tauri/index.html`
- Create: `desktop/tauri/style.css`
- Create: `desktop/tauri/app.js`

**Steps:**

1. Write Rust/JavaScript contract tests for command scope and splash state transitions before application code.
2. Display the bundled splash immediately and emit resolving/starting/ready/error states.
3. Expose only `get_desktop_state`, `retry_startup`, `select_runtime`, and `open_logs` to the bundled splash.
4. After readiness, navigate the main WebView to `http://127.0.0.1:7860/`; do not grant remote-domain IPC access.
5. Enforce a single instance, exact navigation rules, external-link handling, no background throttling where WebView2 permits it, and clean process ownership on all exit paths.

### Task 5: Integrate Standard and Full Windows builds

**Files:**

- Create: `src-tauri/tauri.full.conf.json`
- Create: `scripts/build_tauri_windows.ps1`
- Create: `tests/desktop/test_tauri_build_contract.py`
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `.gitignore`

**Steps:**

1. Write RED tests that require Standard NSIS, Full portable/unpacked resource mapping, license inclusion, Runtime validation, privacy exclusions, and TensorRT helper preservation.
2. Add `tauri:dev`, `tauri:build`, and desktop-test scripts without removing Electron scripts.
3. Standard builds only the shell and selects an external Runtime.
4. Full first calls the existing portable Runtime builder and validation gates, then maps the staged Runtime into Tauri resources.
5. Refuse a single-file Full NSIS/MSI path when the payload exceeds the safe distribution model; produce unpacked/Zip64 delivery instead.

### Task 6: Preserve settings and mutable data boundaries

**Files:**

- Modify: `src-tauri/src/runtime.rs`
- Modify: `src-tauri/src/app.rs`
- Modify: `scripts/run_local_stream.py` only if a new explicit writable-root variable is required
- Test: `src-tauri/tests/runtime_contract.rs`
- Test: relevant Python runtime tests

**Behaviors:**

- Persist selected Runtime under the Tauri application config directory.
- Keep WebView2 local storage stable across restarts.
- Route logs and mutable Runtime data to user-writable directories when installed; never overwrite packaged public model inputs.
- Preserve existing avatars, cloned voices, references, API keys, engine caches, and settings across shell updates/uninstall unless the user explicitly removes them.

### Task 7: Documentation and migration compatibility

**Files:**

- Create: `docs/windows-tauri-desktop-distribution.md`
- Modify: `README.md`
- Preserve: `docs/windows-desktop-distribution.md`

**Content:**

- Document Standard vs Full, WebView2 offline strategy, Runtime selection, logs, TensorRT build assistant, update separation, code signing, and license boundaries.
- Mark Electron as a temporary fallback, not the new default, until Task 8 acceptance passes.
- Include exact PowerShell build and launch commands.

### Task 8: Fresh verification and real Windows acceptance

1. Run `cargo fmt --manifest-path src-tauri/Cargo.toml -- --check`.
2. Run `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`.
3. Run `cargo test --manifest-path src-tauri/Cargo.toml --all-targets`.
4. Run the Tauri build-contract and existing desktop Electron tests.
5. Build the Standard NSIS/unpacked Tauri app.
6. Launch the real executable against the current development Runtime; verify splash to AVTR UI.
7. Verify microphone permission, device enumeration, WebRTC connection, audio playback, themes, avatar display, window resizing, and persisted settings.
8. Quit the app and verify owned listeners on 7860/8000/8767/8768 are released while an externally owned stack is left untouched.
9. Relaunch and verify single-instance behavior and clean recovery from missing Runtime/backend failure.
10. Run focused Python shutdown/health tests and `git diff --check` on all touched files.

Electron may be retired only after all Task 8 acceptance checks pass. Until then both shells coexist and share the same portable Runtime builder.
