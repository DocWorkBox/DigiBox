# AVTR-1 Portable Runtime and Desktop Distribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a relocatable native-Windows AVTR-1 runtime with three isolated standalone CPython installations, and package it with the Electron shell plus a post-install TensorRT build assistant.

**Architecture:** `build_portable_runtime.ps1` stages an application-owned runtime from Astral/uv CPython distributions instead of copying path-bound virtual environments. It copies only runtime code, model inputs and required third-party source while enforcing privacy and machine-specific artifact exclusions. `build_desktop_windows.ps1` then selects the existing shell-only builder or a full builder whose `extraResources` installs the staged runtime under Electron's `resources/avtr-runtime` directory.

**Tech Stack:** PowerShell 5.1+, uv managed CPython/python-build-standalone, Node.js built-in test runner, Electron 43, electron-builder 26, NSIS for Standard, Zip64/unpacked Full distribution, TensorRT 10.11.

---

### Task 1: Define the distribution contracts in failing tests

**Files:**
- Create: `desktop/test/portable-runtime-build.test.cjs`

**Step 1: Write failing static and PowerShell-plan tests**

Assert three standalone runtime names and versions, copy-mode dependency installation, guarded `-Clean`, forbidden private/machine-specific files, full builder resource mapping, and distribution documentation sections.

**Step 2: Run the test to verify RED**

Run: `node --test desktop/test/portable-runtime-build.test.cjs`

Expected: FAIL because the scripts, full builder config, and distribution guide do not exist.

### Task 2: Build the portable runtime safely

**Files:**
- Create: `scripts/desktop/build_portable_runtime.ps1`
- Test: `desktop/test/portable-runtime-build.test.cjs`

**Step 1: Add read-only plan output and exact target validation**

Resolve source and destination paths, refuse non-empty destinations by default, and allow `-Clean` only for a leaf named `avtr-runtime` carrying the script's marker. `-PlanOnly` must perform validation but never create or remove files.

**Step 2: Stage three application-owned CPython runtimes**

Use `uv python install --install-dir` for main 3.12, CosyVoice 3.10, and FeyNoBg 3.12, then copy each standalone distribution into `python-main`, `python-cosyvoice`, and `python-feynobg`. Install packages with uv link mode `copy`; never create or copy a venv.

**Step 3: Copy and validate the payload**

Copy `src`, `scripts`, the required CosyVoice/Matcha-TTS source, root license files, non-engine artifact inputs, and complete public model files. Exclude all caches, user assets, TensorRT engines, warp DLLs, and `spk2info.pt`; scan the completed payload and fail if any forbidden path remains.

**Step 4: Write the runtime manifest**

Record the portable layout, exact Python versions, relative paths, exclusions, and an empty TensorRT engine inventory.

### Task 3: Orchestrate standard and full desktop builds

**Files:**
- Create: `scripts/build_desktop_windows.ps1`
- Create: `electron-builder-full.yml`
- Test: `desktop/test/portable-runtime-build.test.cjs`

**Step 1: Add a deterministic build plan**

`-Edition Standard` selects the existing shell-only config. `-Edition Full` builds or validates the staged portable runtime and selects `electron-builder-full.yml`. `-PlanOnly` prints JSON without invoking npm, uv, or deleting anything.

**Step 2: Add the full electron-builder config**

Include the staged runtime at `resources/avtr-runtime`, explicitly include the TensorRT setup helper, and repeat private/engine exclusions as packaging defense in depth.

**Step 3: Invoke electron-builder without shell interpolation**

Resolve npm explicitly, pass argument arrays, support Standard NSIS plus Full Zip64/unpacked-directory targets, and fail on non-zero native exit codes.

### Task 4: Document redistribution and hardware boundaries

**Files:**
- Create: `docs/windows-desktop-distribution.md`
- Test: `desktop/test/portable-runtime-build.test.cjs`

**Step 1: Document the two desktop editions**

Explain that Standard is a small shell requiring an existing runtime, while Full embeds three standalone runtimes and public model/build inputs but intentionally excludes private and GPU-specific state.

**Step 2: Document TensorRT Standard and Full helper modes**

List the exact NVIDIA/runtime prerequisites for standard engine creation and the additional Visual Studio, CMake/Ninja, CUDA Toolkit, and matching TensorRT SDK requirements for the custom warp plugin.

**Step 3: Document license and privacy obligations**

State the Renderer and Streamer noncommercial limitation, the model revenue threshold and redistribution obligations, third-party restrictions, and the requirement to obtain separate commercial licenses where applicable.

### Task 5: Verify the implementation

**Files:**
- Verify: all files above

**Step 1: Run PowerShell parser checks**

Parse both scripts through `System.Management.Automation.Language.Parser` and require zero syntax errors.

**Step 2: Run focused desktop tests**

Run: `node --test desktop/test/portable-runtime-build.test.cjs`

Expected: all tests pass.

**Step 3: Run the complete desktop suite**

Run: `npm run test:desktop`

Expected: all tests pass.

**Step 4: Check whitespace and scope**

Run `git diff --check` for the assigned scripts, config, documentation, and tests. No commit is created because the user did not request one.
