# AVTR-1 Low-Latency And RTX VSR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce speech-to-speech response latency across all six identified stages and add an optional, real NVIDIA VFX VideoSuperRes output path capped at 1080p.

**Architecture:** Keep the existing single-session WebRTC architecture, but add a process-local turn-latency store shared by the conversation engine and media worklet. Qwen ASR moves from whole-WAV HTTP upload to a per-turn WebSocket stream; LLM production and TTS consumption run concurrently through a bounded queue. RTX VSR runs as an optional post-composition CUDA/DLPack stage through NVIDIA's `nvvfx.VideoSuperRes`, with explicit capability reporting and no fake non-AI fallback.

**Tech Stack:** Python 3.12, asyncio, websockets, FastAPI, Preact/HTM, PyAV/aiortc, PyTorch CUDA, NVIDIA Maxine VFX SDK Python bindings, pytest.

---

### Task 1: Turn latency observability

**Files:**
- Create: `src/avaturn_live_streamer/performance_metrics.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/localrtc/worklet.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Test: `tests/test_performance_metrics.py`
- Test: `tests/test_local_stream_ui.py`

**Steps:**
1. Write failing tests for monotonic stage recording, sanitised snapshots, and first avatar frame completion.
2. Run the focused tests and verify failure because the store and UI fields do not exist.
3. Implement `TurnLatencyStore` with stages `vad_endpoint`, `asr_complete`, `llm_first_token`, `first_text_chunk`, `tts_first_pcm`, and `first_avatar_frame`.
4. Feed the current snapshot through `/system-stats` and render it in the optional performance overlay.
5. Run focused tests until green.

### Task 2: Low-latency VAD and early sentence chunking

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Test: `tests/conversation_engines/test_custom_api_contract.py`
- Test: `tests/test_local_stream_ui.py`

**Steps:**
1. Write failing tests for a persisted low-latency toggle, 350 ms hangover, 24-character hard cap, and comma/semicolon soft boundaries after a safe minimum length.
2. Verify the tests fail against the current 550 ms / 60-character behavior.
3. Add `low_latency_mode`, keeping 550 ms as the compatibility value and using 350 ms when enabled.
4. Implement `SentenceChunker(max_chars=24, min_soft_chars=12)` with hard and soft punctuation classes.
5. Run focused tests until green.

### Task 3: Selective Qwen web search

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Test: `tests/conversation_engines/test_custom_api_preflight.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`

**Steps:**
1. Write failing tests showing stable/local questions omit `web_search`, while current/latest/search-intent questions include it.
2. Verify the tests fail because tools are currently attached to every Responses request.
3. Add `web_search_mode = off|auto|always`, migrate the old boolean to `auto`, and persist the UI selection.
4. Route each LLM turn without an extra provider call using a deterministic, testable search-intent classifier.
5. Keep preflight capable of explicitly probing the web-search tool.
6. Run focused tests until green.

### Task 4: Qwen realtime streaming ASR

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/aliyun_bailian_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Test: `tests/conversation_engines/test_aliyun_bailian_client.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`

**Steps:**
1. Write a fake-WebSocket failing test for `session.update`, incremental `input_audio_buffer.append`, `commit`, `session.finish`, final transcript, provider errors, cancellation, and secret redaction.
2. Verify failure because no realtime ASR client exists.
3. Implement a per-turn `BailianQwenRealtimeASRSession` using `qwen3-asr-flash-realtime`, PCM16/16 kHz and manual turn detection.
4. Start the WebSocket at local speech onset, stream chunks while the user speaks, and commit at the 350 ms local VAD endpoint.
5. Retain whole-WAV HTTP ASR as an explicit compatibility fallback when a non-realtime model is selected.
6. Run focused tests until green.

### Task 5: Concurrent LLM producer and TTS consumer

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`

**Steps:**
1. Write a failing ordering test proving LLM delta production continues while the first TTS piece is still being consumed.
2. Verify the current serial implementation fails the ordering assertion.
3. Add a bounded `asyncio.Queue` and run LLM producer/TTS consumer in one `TaskGroup`.
4. Preserve interruption, segment completion, transcript history, and error propagation semantics.
5. Run focused tests until green.

### Task 6: Local CosyVoice first-packet regression

**Files:**
- Modify: `src/avaturn_live_streamer/integrations/cosyvoice_server.py`
- Test: `tests/integrations/test_cosyvoice_server_contract.py`

**Steps:**
1. Write a failing regression test proving each streaming request restores the model's original `token_hop_len`.
2. Verify the current shared model state leaks the doubled value across requests.
3. Guard synthesis with a per-model lock and restore `token_hop_len` in `finally`.
4. Run focused tests until green.

### Task 7: RTX VideoSuperRes capability and dimensions

**Files:**
- Modify: `src/avaturn_live_streamer/integrations/nvidia_video_effects.py`
- Modify: `src/avaturn_live_streamer/types.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Test: `tests/localrtc/test_nvidia_video_effects_detection.py`
- Test: `tests/localrtc/test_output_quality.py`
- Test: `tests/test_local_stream_ui.py`

**Steps:**
1. Write failing tests for capability details, persisted switch, offer propagation, disabled/unavailable errors, and mappings 360p→720p, 540p→1080p, 720p→1080p while preserving aspect ratio/even dimensions.
2. Verify failure because the current detector is not connected to production.
3. Add `rtx_super_resolution: bool` independently from `output_quality`.
4. Expose `/nvidia-video-effects` capability data and prevent session start with a precise installation message if requested but unavailable.
5. Run focused tests until green.

### Task 8: NVIDIA VFX CUDA/DLPack processing stage

**Files:**
- Create: `src/avaturn_live_streamer/integrations/nvidia_vsr.py`
- Modify: `src/avaturn_live_streamer/localrtc/worklet.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Create: `scripts/setup_nvidia_vfx_windows.ps1`
- Test: `tests/localrtc/test_nvidia_vsr.py`
- Test: `tests/scripts/test_setup_nvidia_vfx_windows.py`

**Steps:**
1. Write failing tests around an injected fake `nvvfx.VideoSuperRes` backend and frame size/format conversion.
2. Verify failure because no processing stage exists.
3. Implement lazy `nvvfx` loading, `QualityLevel.HIGH`, CUDA/DLPack input, immediate output clone, lifecycle cleanup, and per-session output-size caching.
4. Apply VSR after aspect crop/downscale and before H.264 encoding; never silently substitute bicubic when RTX mode is requested.
5. Add a Windows setup helper that validates driver/GPU, SDK root, Python bindings and `nvvfxvideosuperres` feature installation without storing NGC credentials.
6. Run focused tests until green.

### Task 9: Regression and live verification

**Files:**
- Modify only if a failing regression demonstrates a real defect.

**Steps:**
1. Run all focused conversation, localrtc, renderer and UI tests.
2. Run the complete pytest suite with a workspace-local temporary base.
3. Start the Windows stack, verify health endpoints, and exercise one short Qwen turn with latency stages visible.
4. Verify 24+ presented FPS with RTX disabled.
5. If NVIDIA VFX SDK is installed, verify all three VSR mappings and record GPU/frame timing; otherwise verify the UI reports the exact missing SDK component and refuses fake VSR.
6. Run `git diff --check` and report modified files, tests, measured gains and the remaining SDK installation boundary.
