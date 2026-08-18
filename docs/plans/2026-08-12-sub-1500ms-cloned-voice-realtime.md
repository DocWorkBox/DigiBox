# Sub-1500ms Cloned-Voice Realtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce warm, non-web-search speech-to-avatar latency toward a p50 of 1.5 seconds while preserving both Qianwen cloud-cloned voices and local CosyVoice clones as the final speech output.

**Architecture:** Keep the existing ASR -> text LLM -> cloned-voice TTS -> AVTR renderer chain. Remove hidden thinking from the fast path, overlap safe work using stable partial ASR without emitting unverified speech, replace per-sentence cloud TTS HTTP calls with a turn-scoped duplex WebSocket, and reduce scheduler/media buffering. Deep-thinking remains an explicit opt-in mode and is outside the low-latency SLO.

**Tech Stack:** Python 3.12, asyncio, httpx, websockets, FastAPI, Preact/HTM, aiortc, pytest.

---

## Non-negotiable compatibility contract

- Never synthesize with a fallback stock voice when a cloned voice ID is selected.
- Pass the selected Qianwen `voice_id` unchanged to every cloud CosyVoice task.
- Preserve local `zero_shot_spk_id` selection, clone inventory, preview, deletion, and persistence.
- Do not play speculative LLM output until the final ASR transcript validates it.
- Web search and deep thinking remain independent controls.
- Do not commit, reset, clean, or overwrite unrelated files in the shared dirty worktree.

### Task 1: Establish latency contracts and configuration schema

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/events.py`
- Modify: `src/avaturn_live_streamer/performance_metrics.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`
- Test: `tests/conversation_engines/test_custom_api_preflight.py`
- Test: `tests/test_performance_metrics.py`

1. Add RED tests for fast/deep thinking payloads, independent web-search behavior, new timing milestones, and backward-compatible defaults.
2. Run the three focused test files and record the expected failures.
3. Add explicit fast/deep thinking mode; fast sends `enable_thinking: false` for supported Qwen Chat Completions, deep preserves reasoning.
4. Add request-sent, first-provider-event, first-reasoning, first-content, ASR-ready/final, and first-audio metrics without storing transcript or secrets.
5. Run focused tests to GREEN.

### Task 2: Shorten VAD, ASR, history, and first text chunk

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/aliyun_bailian_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/local_tts_bridge.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Test: `tests/conversation_engines/test_aliyun_bailian_client.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`
- Test: `tests/conversation_engines/test_local_tts_bridge.py`

1. Add RED tests for 220 ms ultra-low VAD, 6-8 Chinese-character first chunk, 120 ms idle flush, shorter fast-mode history, and ASR completion returning on final transcript rather than session teardown.
2. Add a reusable/prepared realtime-ASR session boundary or background handshake that never blocks microphone ingestion; retain HTTP ASR fallback.
3. Resolve the ASR turn on `transcription.completed` and clean up `session.finished` asynchronously with bounded timeouts.
4. Implement asymmetric first/subsequent chunking and timed flush while preserving word/number boundaries.
5. Run focused tests to GREEN.

### Task 3: Implement safe speculative text generation

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/aliyun_bailian_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Test: `tests/conversation_engines/test_aliyun_bailian_client.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`

1. Add RED tests for stable partial-ASR callbacks, speculative LLM buffering, exact/prefix validation, mismatch cancellation/restart, interruption, and no speculative audio.
2. Start a private LLM task only after a stable partial transcript threshold.
3. On final ASR, release buffered text only if normalized validation succeeds; otherwise cancel it and run the final transcript normally.
4. Keep TTS invocation behind final validation so cloud/local cloned voices never speak unverified text.
5. Run focused tests to GREEN.

### Task 4: Replace cloud CosyVoice per-piece HTTP with duplex WebSocket

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/aliyun_bailian_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Test: `tests/conversation_engines/test_aliyun_bailian_client.py`
- Test: `tests/conversation_engines/test_custom_api_contract.py`
- Test: `tests/conversation_engines/test_custom_api_preflight.py`

1. Add RED protocol tests for `run-task`, `task-started`, incremental `continue-task`, binary PCM, `finish-task`, task failure, cancellation, and key redaction.
2. Add a turn-scoped streaming-text TTS interface with a bounded producer queue and deterministic close/cancel behavior.
3. Implement Qianwen CosyVoice WebSocket using the selected cloned `voice_id` unchanged; retain HTTP only as a controlled compatibility fallback before any audio is emitted.
4. Feed stable LLM text increments into one TTS task rather than opening one network request per sentence.
5. Run focused tests to GREEN.

### Task 5: Preserve and accelerate local cloned voices

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/local_tts_bridge.py`
- Modify: `src/avaturn_live_streamer/integrations/cosyvoice_server.py`
- Test: `tests/conversation_engines/test_local_tts_bridge.py`
- Test: `tests/integrations/test_cosyvoice_server_contract.py`

1. Add RED tests proving the selected `zero_shot_spk_id` survives streaming, cancellation, preview, and consecutive turns.
2. Add a bounded turn-level incremental-text endpoint/bridge while retaining existing clone inventory APIs.
3. Keep the inference lock, hop reset, cancellation cleanup, and voice cache semantics from the current service.
4. Benchmark Torch FP16 and any available TensorRT flow engine; select TensorRT only when an actual compatible engine exists and improves TTFA without hurting renderer FPS.
5. Run focused tests to GREEN.

### Task 6: Reduce scheduler/media startup buffering and expose real first-audible timing

**Files:**
- Modify: `src/avaturn_live_streamer/speech/speech_scheduler.py`
- Modify: `src/avaturn_live_streamer/localrtc/worklet.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Modify: `src/avaturn_live_streamer/performance_metrics.py`
- Test: locate/update scheduler tests under `tests/`
- Test: `tests/test_local_stream_ui.py`
- Test: `tests/test_performance_metrics.py`

1. Add RED tests for first-audio coalescing, maximum startup wait, interruption, browser first-non-silent-audio reporting, and persistence of latency controls.
2. Coalesce the first PCM packets until enough useful audio or a short 80-120 ms deadline, then publish generation start once; avoid converting a tiny first packet into hundreds of milliseconds of left padding.
3. Add browser-side first non-silent remote-audio measurement and a loopback-only milestone endpoint; distinguish server first frame from actual browser first audible audio.
4. Show p50/p95 stage timings in the existing performance overlay and mark deep-thinking turns outside the fast SLO.
5. Run focused tests to GREEN.

### Task 7: Integrated verification and live acceptance

**Files:**
- Test only; modify production code only for defects proven by the integrated run.

1. Run focused suites for all changed modules with a unique Windows `--basetemp`.
2. Run the complete test suite, Ruff on changed Python files, JavaScript module syntax validation, `compileall`, and `git diff --check`.
3. Restart the Windows-native service tree using the project launcher; verify ports 7860, 8000, 8767, and 8768 and health endpoints.
4. Warm the selected ASR/TTS/renderer once, then run at least 10 short non-web-search turns using a cloud-cloned voice and 10 using a local clone.
5. Report end-of-speech to browser-first-audible p50/p95, ASR/LLM/TTS/renderer breakdown, 24 fps stability, clone IDs used (redacted if needed), fallback count, interruption correctness, and any remaining provider/network limit.
6. Claim the 1.5-second target only if the measured warm p50 meets it; otherwise report the exact residual bottleneck and observed value.
