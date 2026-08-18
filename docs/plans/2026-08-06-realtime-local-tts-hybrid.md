# Realtime Local TTS Hybrid Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional hybrid mode in which Codex GPT-Live or OpenAI Realtime handles microphone understanding and text generation, cloud speech is never sent to the avatar, local CosyVoice streams synthesized PCM into the existing avatar pipeline, and interruption stops local speech promptly.

**Architecture:** Reuse the existing `TTSAPIOptions`, `OpenAICompatibleTTS`, and sentence chunking code instead of creating another TTS transport. OpenAI Realtime requests text output when hybrid mode is enabled and converts assistant text deltas into ordered local-TTS segments. Codex requests text output when its local app-server protocol supports it; otherwise the implementation explicitly discards Codex cloud-audio events while consuming transcript deltas as the compatibility source. Both engines share cancellation, segment lifecycle, and avatar-buffer discard semantics. The web UI exposes independent persisted switches for OpenAI and Codex while reusing the installed local CosyVoice URL, model, voice list, refresh, and preview controls.

**Tech Stack:** Python 3.12, asyncio, FastAPI/Pydantic, OpenAI Realtime WebSocket/WebRTC protocol, Codex app-server protocol, OpenAI-compatible local CosyVoice HTTP streaming, vanilla HTML/CSS/JavaScript, pytest.

---

### Task 1: Lock the protocol and regression contract

**Files:**
- Modify: `tests/conversation_engines/test_realtime_api_client.py`
- Modify: `tests/conversation_engines/test_codex_realtime_client.py`
- Modify: `tests/test_local_stream_ui.py`
- Modify: `tests/test_conversation_engine_builders.py`

**Steps:**
1. Verify the current OpenAI Realtime text-output event names against official OpenAI documentation.
2. Inspect the installed Codex app-server schema/help for supported output modalities.
3. Add failing tests for local-TTS configuration, cloud-audio suppression, text-delta synthesis, ordered completion, and interruption cancellation.
4. Add failing UI source-contract tests for both hybrid switches, persisted state, payload construction, local-voice selection, and validation.
5. Run only the new tests and record the expected RED failures.

### Task 2: Extract the reusable local streaming TTS bridge

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Create or modify: `src/avaturn_live_streamer/conversation_engines/local_tts_bridge.py`
- Test: `tests/conversation_engines/test_local_tts_bridge.py`

**Steps:**
1. Extract or wrap the existing sentence chunker and OpenAI-compatible streaming TTS client behind a small per-response bridge.
2. Preserve the existing custom API and MiniMax behavior.
3. Implement ordered text ingestion, final flush, segment close, cancellation, and `DiscardAvatarSpeechBuffer` publication.
4. Run bridge and existing custom-provider tests.

### Task 3: Add OpenAI Realtime hybrid mode

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/realtime_api_client.py`
- Test: `tests/conversation_engines/test_realtime_api_client.py`
- Test: `tests/test_conversation_engine_builders.py`

**Steps:**
1. Add optional `tts_override` configuration to the OpenAI engine model.
2. Mint/configure a text-output Realtime session when the override is present.
3. Consume assistant text deltas and completion events through the local bridge.
4. Ignore any cloud-audio delta defensively in hybrid mode.
5. Cancel local synthesis and discard buffered avatar speech on barge-in.
6. Run focused OpenAI tests.

### Task 4: Add Codex GPT-Live hybrid mode

**Files:**
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/codex_realtime_client.py`
- Test: `tests/conversation_engines/test_codex_realtime_client.py`
- Test: `tests/test_conversation_engine_builders.py`

**Steps:**
1. Add optional `tts_override` configuration to the Codex engine model.
2. Select Codex text output when supported by the installed app-server contract.
3. Feed assistant transcript deltas into the local bridge and suppress direct cloud-audio publication.
4. Preserve microphone upload, transcript display, session continuity, and reconnection behavior.
5. Cancel local synthesis and discard buffered avatar speech on user interruption.
6. Run focused Codex tests.

### Task 5: Add UI configuration and persistence

**Files:**
- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Modify: `tests/test_local_stream_ui.py`

**Steps:**
1. Add independent “本地 CosyVoice 混合模式” switches to the OpenAI and Codex panels.
2. Reuse the local TTS URL, key, model, installed-voice dropdown, refresh, and preview controls.
3. Persist both switches and all local TTS fields without losing provider-specific API keys.
4. Include `tts_override` only when the selected engine's hybrid switch is enabled.
5. Validate URL/model/voice and show a clear connection summary before session start.
6. Run UI tests.

### Task 6: Verify the complete Windows flow

**Files:**
- Modify if needed: `scripts/run_local_stream_windows.ps1`
- Test: all affected tests, then the full suite

**Steps:**
1. Run formatting/static checks on changed Python files.
2. Run all focused conversation-engine, UI, and launcher tests.
3. Run the complete pytest suite with a repo-local base temp directory.
4. Restart only the AVTR local services needed for the change.
5. Use the live web UI to connect in OpenAI and Codex hybrid modes where credentials/session availability permit; otherwise verify configuration and local synthesis endpoints separately and report the exact remaining external dependency.

**Note:** This is the user's active shared Windows deployment with existing uncommitted work and running services. The implementation stays in the current worktree and does not create commits unless the user asks.
