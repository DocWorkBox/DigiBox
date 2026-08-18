# AVTR-1 Native Windows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run AVTR-1 directly on Windows 11 with the installed NVIDIA GPU, without WSL2, Docker, or a Linux virtual machine.

**Architecture:** Keep the released TorchScript AVTR-1 model on CUDA and add an in-process PyTorch backend for its encode/decode stages. Run HuBERT and the LivePortrait renderer through ONNX Runtime CUDA after caching batch-dynamic Windows-compatible graphs for HuBERT, warp, stitch, and MODNet. Make TensorRT an optional lazy dependency and support a hybrid Windows TensorRT mode that accelerates every stage except the custom-plugin warp, then validate with a real gated-model download and short offline render.

**Tech Stack:** Python 3.12, PyTorch CUDA 12.8, ONNX Runtime GPU, TorchScript, FastAPI, imageio-ffmpeg, PowerShell, pytest.

---

### Task 1: Make TensorRT optional

**Files:**
- Create: `tests/runtime/test_optional_tensorrt.py`
- Modify: `src/avtr1_renderer/runtime/__init__.py`
- Modify: `src/avtr1_renderer/runtime/loader.py`

1. Write a subprocess test that blocks importing `tensorrt` and verifies `avtr1_renderer.runtime` plus the ONNX loader can still import.
2. Run the test and verify it fails because `runtime.__init__` imports `TRTEngine` eagerly.
3. Move the TensorRT import into the `.engine` branch and expose `TRTEngine` lazily for compatibility.
4. Run the targeted test and the existing test suite.

### Task 2: Add a TorchScript AVTR-1 engine pair

**Files:**
- Create: `tests/runtime/test_torchscript_avtr1.py`
- Create: `src/avtr1_renderer/runtime/torchscript_avtr1.py`

1. Write tests around a small fake AVTR ScriptModule contract: encode output mapping, reusable `out=` buffers, decode keyword mapping, and output allocation shapes.
2. Run the tests and verify the module is missing.
3. Implement a shared-module backend exposing `encode` and `decode` objects matching the existing `InferenceEngine` protocol.
4. Keep tensors on the caller's CUDA device in production while permitting CPU fake modules in unit tests.
5. Run the targeted tests and refactor only after they pass.

### Task 3: Load normalizer data directly from TorchScript

**Files:**
- Create: `tests/test_normalizer_torchscript.py`
- Modify: `src/avtr1_renderer/avtr1_motion_generator.py`

1. Write a failing test with all released normalizer buffers on a fake scripted module.
2. Add `Normalizer.from_scripted` and validate missing buffers with an actionable error.
3. Verify the returned tensors preserve device, dtype, and the 39-coordinate lip-sync tail.

### Task 4: Select a portable backend in Pipeline

**Files:**
- Create: `tests/test_pipeline_backend.py`
- Modify: `src/avtr1_renderer/pipeline.py`
- Modify: `scripts/generate_offline.py`

1. Write failing pure selection tests: `auto` chooses TorchScript on Windows, explicit `tensorrt` requires both engines, and explicit `torchscript` does not.
2. Add `avtr1_backend=auto|torchscript|tensorrt` to `Pipeline.from_artifacts` and `--backend` to the offline CLI.
3. Load the released `avtr1.scripted.pt` once for the portable backend and derive the normalizer from that same module.
4. When using the portable backend, force renderer/HuBERT models to their ONNX paths so a copied Linux engine or plugin is never loaded on Windows.
5. Run selection and pipeline unit tests.

### Task 5: Harden ONNX Runtime CUDA startup on Windows

**Files:**
- Create: `tests/runtime/test_onnx_windows.py`
- Modify: `src/avtr1_renderer/runtime/onnxrt.py`

1. Write a platform-isolated test that expects ONNX Runtime DLL preloading before session creation on Windows.
2. Add guarded `onnxruntime.preload_dlls()` support when available, keeping Linux behavior unchanged.
3. Validate that `CUDAExecutionProvider` is present and raise a diagnostic listing available providers when it is not.

### Task 6: Add reproducible Windows bootstrap and diagnostics

**Files:**
- Create: `requirements-windows.txt`
- Create: `scripts/setup_windows.ps1`
- Create: `scripts/windows_diagnostics.py`
- Create: `scripts/run_offline_windows.ps1`
- Create: `scripts/run_interactive_windows.ps1`
- Create: `scripts/build_tensorrt_windows.ps1`
- Modify: `README.md`

1. Add pinned/compatible Windows dependencies without TensorRT.
2. Create an idempotent setup script using local `.venv` and the installed `uv`, including the PyTorch cu128 index.
3. Add diagnostics for Windows version, GPU, CUDA availability, compute capability, ORT CUDA provider, FFmpeg, Hugging Face authentication, and artifact paths.
4. Add a launcher that activates `.venv`, selects `torchscript`, and forwards offline CLI arguments safely.
5. Document native Windows setup, model-gate acceptance, limitations, and rollback/removal paths.

### Task 7: Install and verify on the current machine

**Files:**
- Runtime output only: `.venv/`, `artifacts/`, `output/windows-smoke.mp4`

1. Run the Windows bootstrap script.
2. Run unit tests and static import checks.
3. Run diagnostics and verify PyTorch sees the RTX 5070 Ti with CUDA and ONNX Runtime exposes `CUDAExecutionProvider`.
4. Check Hugging Face authentication without printing the token; if access is absent, stop only at the gated-download boundary and request the user login/acceptance action.
5. Download artifacts and run a one-chunk pipeline smoke test.
6. Render a one-second offline MP4 and inspect file metadata/frame count.
7. Record actual latency, peak VRAM, and any quality/performance difference from TensorRT.

### Task 8: Final regression and handoff

**Files:**
- Modify only if verification exposes a tested defect.

1. Run the complete test suite.
2. Run Ruff on changed Python files.
3. Confirm `git diff --check`, review the final diff, and verify no credentials or downloaded model binaries are tracked.
4. Report exactly what works natively, what remains slower than Linux TensorRT, and the commands to reproduce the verified result.
