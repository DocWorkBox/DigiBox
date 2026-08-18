# DigiBox Local Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one-owner, local-only long-term memory for people, relationships, and events with silent saves, bounded recall, safe merge import/export, and low-confidence-only automatic cleanup.

**Architecture:** A standard-library SQLite core lives behind an async `MemoryService`. A non-blocking event worklet extracts only final user transcripts, while engine builders receive a bounded confirmed-memory prompt and Custom API gets a 25ms per-turn recall. Loopback-only FastAPI routes and a settings card provide management, export, preview, safe merge import, and clear-with-backup.

**Tech Stack:** Python 3.12 (`sqlite3`, `asyncio`, `json`, `hashlib`), FastAPI/Pydantic already bundled, React-style in-browser UI already vendored, Node desktop contracts, Rust/Tauri environment contracts, pytest/node:test/cargo test.

---

## Task 1: Freeze storage paths and schema invariants

**Files:**

- Create: `src/avaturn_live_streamer/memory/__init__.py`
- Create: `src/avaturn_live_streamer/memory/models.py`
- Create: `src/avaturn_live_streamer/memory/paths.py`
- Create: `src/avaturn_live_streamer/memory/schema.py`
- Create: `tests/memory/test_paths.py`
- Create: `tests/memory/test_schema.py`

1. Write RED tests for explicit absolute root, LocalAppData fallback, unsafe cwd/runtime rejection, idempotent schema creation, confirmed-no-expiry constraints, temporary-candidate constraints, foreign keys, and newer-schema refusal.
2. Run: `python -B -m pytest -q -p no:cacheprovider tests/memory/test_paths.py tests/memory/test_schema.py --basetemp F:\AVTR-1\test-results\memory-schema-red`
3. Implement the smallest path resolver, immutable DTOs, schema v1, and connection PRAGMAs.
4. Re-run the same tests with a new basetemp and make them GREEN.
5. Commit: `feat(memory): define local storage schema`

## Task 2: Implement transactional store, retention, and recall

**Files:**

- Create: `src/avaturn_live_streamer/memory/sqlite_store.py`
- Create: `tests/memory/test_sqlite_store.py`
- Create: `tests/memory/test_retention.py`
- Create: `tests/memory/test_recall.py`

1. Write RED tests for atomic person/relationship/event ingest, source replay idempotency, normalization without same-name auto-merge, optimistic revisions, one-shot follow-up claim, and clear-all atomicity.
2. Add RED retention tests proving only expired low-confidence candidates are purged and confirmed/high-confidence/referenced memories survive.
3. Add RED recall tests for alias/event relevance, confirmed-only results, five-item cap, bounded session profile, and non-mutating due-follow-up discovery.
4. Implement transactions and deterministic scoring; do not add a vector database or GPU dependency.
5. Run the three focused test files and commit: `feat(memory): add transactional local store`

## Task 3: Implement local extraction and asynchronous service

**Files:**

- Create: `src/avaturn_live_streamer/memory/extractor.py`
- Create: `src/avaturn_live_streamer/memory/service.py`
- Create: `src/avaturn_live_streamer/memory/worklet.py`
- Create: `tests/memory/test_extractor.py`
- Create: `tests/memory/test_service.py`
- Create: `tests/memory/test_worklet.py`

1. Write RED extraction tests for owner name, two relationship word orders, explicit remember cue, dated planned/completed events, uncertainty downgrade, and no extraction from assistant text.
2. Write RED service tests for `to_thread`, 25ms timeout fail-open, immediate `try_submit`, bounded queue, mutation serialization, graceful drain, and corrupted/locked DB degradation.
3. Write RED worklet tests proving only `InputTranscript` is submitted, duplicate user transcripts are idempotent, queue full never blocks EventBus, and Shutdown clears pending state.
4. Implement heuristic extraction and background writer with no cloud call and no engine/render dependency.
5. Run focused tests and commit: `feat(memory): save user memories asynchronously`

## Task 4: Add safe export/import and management operations

**Files:**

- Create: `src/avaturn_live_streamer/memory/transfer.py`
- Create: `tests/memory/test_transfer.py`

1. Write RED tests for deterministic atomic export, round trip, preview-no-write, identical skip, ID/fingerprint conflicts, dangling references, size/version rejection, pre-import backup, stale plan, rollback, and never update/delete local data.
2. Implement canonical JSON, SHA-256, strict limits, preview tokens, backup, and a single `BEGIN IMMEDIATE` insert-only transaction.
3. Run: `python -B -m pytest -q -p no:cacheprovider tests/memory/test_transfer.py --basetemp F:\AVTR-1\test-results\memory-transfer`
4. Commit: `feat(memory): add safe local transfer`

## Task 5: Route the one persistent local data root

**Files:**

- Modify: `src-tauri/src/app.rs`
- Modify: `src-tauri/tests/supervisor_contract.rs`
- Modify: `desktop/main.cjs`
- Modify: `desktop/lib/backend-supervisor.cjs`
- Modify: `desktop/test/backend-supervisor.test.cjs`
- Modify: `desktop/test/main-contract.test.cjs`
- Modify: `scripts/dev_runtime_windows.ps1`
- Modify: `scripts/test_windows.ps1`
- Modify: `tests/scripts/test_single_python_dev_runtime.py`

1. Add RED contracts for `%LOCALAPPDATA%\DigiBox\memory`, unique case-insensitive `AVTR1_MEMORY_ROOT`, no Runtime/Roaming fallback, and isolated test root.
2. Implement Tauri, Electron, source-dev and test env injection.
3. Run focused Node, Rust and Python contract tests.
4. Commit: `feat(memory): persist data outside runtime`

## Task 6: Inject bounded confirmed memory and register the worklet

**Files:**

- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/builders.py`
- Modify: `src/avaturn_live_streamer/conversation_engines/custom_api_client.py`
- Modify: `tests/scripts/test_run_local_stream_windows.py`
- Modify: `tests/conversation_engines/test_openai_hybrid_tts.py`
- Modify: `tests/conversation_engines/test_codex_builder.py`
- Modify: `tests/conversation_engines/test_custom_api_contract.py`
- Modify: `tests/conversation_engines/test_minimax_realtime_client.py`

1. Add RED tests that session profile enters OpenAI/Codex/MiniMax/Custom prompts as a delimited untrusted-data block and candidates never enter prompts.
2. Add RED Custom API tests for at-most-five per-turn items, correct system-message placement, 25ms timeout, degraded-empty fallback, and no history pollution.
3. Add RED integration tests proving `MemoryWorklet` is registered once per session and memory initialization/recall failure does not break `/engine-connections` or `/offer`.
4. Implement session-profile augmentation and Custom per-turn callback while retaining existing speculative fast path behavior.
5. Run all conversation-engine and local-stream focused tests.
6. Commit: `feat(memory): recall context without blocking dialogue`

## Task 7: Add loopback management API

**Files:**

- Create: `src/avaturn_live_streamer/memory/api.py`
- Modify: `src/avaturn_live_streamer/local_stream_cli.py`
- Create: `tests/localrtc/test_memory_api.py`

1. Write RED tests for remote 403, stats degradation, pagination/filter validation, delete revisions, clear confirmation, export headers, import preview/apply, stale token/revision, size/schema errors, and unavailable-service 503.
2. Implement a router using the same `MemoryService` instance as dialogue; routes never access SQLite directly.
3. Run focused API tests and commit: `feat(memory): expose loopback management API`

## Task 8: Add settings UI

**Files:**

- Modify: `src/avaturn_live_streamer/local_stream_ui.html`
- Modify: `tests/test_local_stream_ui.py`

1. Add RED source/behavior contracts for the 本地记忆 card, stats, manager filters, candidate expiry text, delete revision, clear phrase, export Blob cleanup, preview-before-import, conflict skip wording, and no memory localStorage writes.
2. Implement main/manager/import settings pages with busy/error states and silent normal saves.
3. Run UI tests and commit: `feat(memory): add local memory controls`

## Task 9: Harden packaging and privacy

**Files:**

- Modify: `scripts/desktop/build_portable_runtime.ps1`
- Modify: `scripts/build_desktop_windows.ps1`
- Modify: `scripts/build_tauri_windows.ps1`
- Modify: `desktop/test/portable-runtime-build.test.cjs`
- Modify: `tests/desktop/test_electron_portable_v2_build.py`
- Modify: `tests/desktop/test_tauri_build_contract.py`

1. Add RED contracts requiring the memory source package and rejecting `memory.sqlite3`, `memory.sqlite3-wal`, `memory.sqlite3-shm`, backups and `.digibox-memory` exports from Runtime/archive.
2. Implement required-payload and forbidden-payload gates.
3. Run desktop packaging contract suites and PowerShell parser checks.
4. Commit: `build(memory): keep local memories out of packages`

## Task 10: Full verification and safe handoff

1. Run all memory tests under Full Runtime Python.
2. Run `scripts/test_windows.ps1` against the verified portable-v2 Runtime for the full Python suite.
3. Run `npm run test:desktop`.
4. Run `npm run test:tauri` or the equivalent offline Rust fmt/check/clippy/test sequence with `AVTR1_TEST_PYTHON` set to the Full Runtime Python.
5. Run `git diff --check`, Python AST/ruff checks, Node syntax checks, and PowerShell 5.1 parser checks for modified scripts.
6. Measure 100 warm recalls and assert p95 below 25ms; prove EventBus submission remains non-blocking under a full queue.
7. Build a synthetic package fixture and prove no memory DB/export is included.
8. Review all diffs, preserve unrelated user changes, then synchronize only memory-feature files back to the primary workspace.
