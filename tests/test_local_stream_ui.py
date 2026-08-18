from __future__ import annotations

import re
from pathlib import Path

UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "avaturn_live_streamer"
    / "local_stream_ui.html"
)


def _ui_source() -> str:
    return UI_PATH.read_text(encoding="utf-8")


def test_frontend_modules_are_pinned_and_loaded_from_local_vendor_routes() -> None:
    source = _ui_source()

    assert "https://esm.sh" not in source
    assert '<script type="importmap" nonce="__AVTR_CSP_NONCE__">' in source
    assert '"preact": "/vendor/preact.module.js"' in source
    assert '"preact/hooks": "/vendor/preact-hooks.module.js"' in source
    assert '"htm": "/vendor/htm.module.js"' in source
    assert '<script type="module" nonce="__AVTR_CSP_NONCE__">' in source
    assert 'from "preact"' in source
    assert 'from "preact/hooks"' in source
    assert 'from "htm"' in source


def test_engine_selector_includes_experimental_codex_gpt_live() -> None:
    source = _ui_source()

    assert '{ id: "codex", label: "Codex GPT-Live（实验功能）" },' in source


def test_cartesia_is_replaced_by_custom_api() -> None:
    source = _ui_source()

    assert '{ id: "custom_api", label: "自定义 API" },' in source
    assert '{ id: "cartesia", label: "Cartesia" },' not in source
    assert 'engine === "cartesia"' not in source
    for field_id in (
        "custom-llm-url",
        "custom-llm-model",
        "custom-asr-url",
        "custom-asr-model",
        "custom-tts-url",
        "custom-tts-model",
        "custom-tts-voice",
    ):
        assert f'id="{field_id}"' in source


def test_engine_connection_is_explicit_and_precedes_session_start() -> None:
    source = _ui_source()

    assert 'id="engine-connect"' in source
    assert 'id="engine-disconnect"' in source
    assert 'fetch("/engine-connections"' in source
    assert 'connection_id: engineConnectionId' in source
    assert "请先连接并测试" in source
    assert "连接并测试" in source


def test_session_end_preserves_connection_intent_for_both_release_modes() -> None:
    source = _ui_source()

    assert "const reconnectAfterSessionRef = useRef(false);" in source
    assert "const resumeSessionAfterConnectRef = useRef(false);" in source
    assert "const recoveryGenerationRef = useRef(0);" in source
    assert "reconnectAfterSessionRef.current = true;" in source
    assert re.search(
        r'if \(active \|\| connectionStatus !== "disconnected" \|\| '
        r'!reconnectAfterSessionRef\.current\) return;\s*'
        r'reconnectAfterSessionRef\.current = false;\s*'
        r'connectEngine\(\);',
        source,
    )

    disconnect_block = re.search(
        r"const disconnectEngine\s*=\s*useCallback\(async\s*\(\)\s*=>\s*\{"
        r"(?P<body>.*?)\n\s*\},\s*\[",
        source,
        re.DOTALL,
    )
    assert disconnect_block is not None
    assert "reconnectAfterSessionRef.current = false;" in disconnect_block.group("body")
    assert "resumeSessionAfterConnectRef.current = false;" in disconnect_block.group("body")
    assert "recoveryGenerationRef.current += 1;" in disconnect_block.group("body")


def test_codex_voice_can_be_previewed_after_preconnection() -> None:
    source = _ui_source()

    assert 'id="codex-voice-preview"' in source
    assert "/preview" in source
    assert "试听音色" in source


def test_codex_voice_change_uses_the_shared_automatic_reconnection() -> None:
    source = _ui_source()

    assert "const reconnectAfterVoiceChangeRef = useRef(false);" in source
    assert "const changeVoiceWithReconnect = useCallback" in source
    change_block = re.search(
        r"const changeCodexVoice\s*=\s*useCallback\(\(nextVoice\)\s*=>\s*\{"
        r"(?P<body>.*?)\n\s*\},\s*\[",
        source,
        re.DOTALL,
    )
    assert change_block is not None
    assert "changeVoiceWithReconnect(setCodexVoice, nextVoice, codexVoice);" in change_block.group("body")
    assert 'codexStatus === "connecting"' in source
    assert re.search(
        r'!selectedConnectionStale.*?'
        r'!reconnectAfterVoiceChangeRef\.current\s*\) return;\s*'
        r'reconnectAfterVoiceChangeRef\.current = false;\s*'
        r'connectEngine\(\);',
        source,
        re.DOTALL,
    )
    assert "codexVoice=${codexVoice} setCodexVoice=${changeCodexVoice}" in source
    assert (
        'disabled=${sessionActive || codexLocalTtsEnabled || connectionBusy || displayConnectionStatus === "stale" || codexStatus === "connecting"}'
        in source
    )


def test_codex_engine_config_sends_selected_voice_without_an_openai_key() -> None:
    source = _ui_source()

    codex_config = re.search(
        r'if \(engine === "codex"\)\s*\{\s*return \{(?P<body>.*?)\};\s*\}',
        source,
        re.DOTALL,
    )

    assert codex_config is not None, (
        "Codex mode must send its selected built-in voice without requiring "
        "or sending an OpenAI API key"
    )
    codex_body = codex_config.group("body")
    assert 'type: "codex"' in codex_body
    assert "voice: codexVoice || DEFAULT_CODEX_VOICE" in codex_body
    assert 'prompt: codexPrompt || ""' in codex_body
    assert "api_key" not in codex_body
    assert 'useLocalStorage("codex_voice", DEFAULT_CODEX_VOICE)' in source
    assert 'id="codex-voice"' in source
    assert 'useLocalStorage("codex_prompt", DEFAULT_CODEX_PROMPT)' in source
    assert 'id="codex-prompt"' in source
    assert "人格 / 系统提示词" in source


def test_output_aspect_ui_is_removed_and_offer_is_fixed_to_16_9() -> None:
    source = _ui_source()

    assert 'useLocalStorage("output_aspect", "16:9")' not in source
    assert 'id="output-aspect"' not in source
    assert "output_aspect: outputAspect" not in source
    assert 'output_aspect: "16:9"' in source


def test_output_quality_has_three_real_presets_and_is_sent_with_offer() -> None:
    source = _ui_source()

    assert 'useLocalStorage("output_quality", "ultra")' in source
    assert 'id="output-quality"' in source
    for quality in ("smooth", "balanced", "ultra"):
        assert f'value="{quality}"' in source
    assert "output_quality: outputQuality" in source
    for label in ("流畅", "高清", "极清"):
        assert label in source


def test_rtx_super_resolution_switch_is_persisted_and_sent_with_offer() -> None:
    source = _ui_source()

    assert 'id="rtx-super-resolution"' in source
    assert 'useLocalStorage("rtx_super_resolution", "false")' in source
    assert 'const rtxSuperResolution = rtxSuperResolutionRaw === "true";' in source
    assert "rtx_super_resolution: rtxSuperResolution" in source
    assert 'fetch("/nvidia-video-effects")' in source


def test_rtx_super_resolution_switch_keeps_checkbox_and_text_inline() -> None:
    source = _ui_source()

    assert (
        '<label class="setting-toggle-row" for="rtx-super-resolution">'
        in source
    )
    row_rule = re.search(r"\.setting-toggle-row\s*\{(?P<body>[^}]*)\}", source)
    assert row_rule is not None
    assert "display: inline-flex" in row_rule.group("body")
    assert "align-items: center" in row_rule.group("body")

    checkbox_rule = re.search(
        r'\.setting-toggle-row input\[type="checkbox"\]\s*\{(?P<body>[^}]*)\}',
        source,
    )
    assert checkbox_rule is not None
    assert "width: auto" in checkbox_rule.group("body")
    assert "padding: 0" in checkbox_rule.group("body")


def test_codex_gpt_live_connection_status_is_visible() -> None:
    source = _ui_source()

    assert 'const [codexStatus, setCodexStatus] = useState("idle");' in source
    assert "codexPc.onconnectionstatechange" in source
    assert "setCodexStatus" in source
    assert "GPT-Live 连接状态" in source
    for label in ("未连接", "连接中", "已连接", "连接失败", "已断开"):
        assert label in source


def test_hardware_monitor_is_opt_in_and_stops_polling_when_disabled() -> None:
    source = _ui_source()

    assert (
        'const [showHardwareStatsRaw, setShowHardwareStatsRaw] = '
        'useLocalStorage("hardware_stats_enabled", "false");'
        in source
    )
    assert 'const showHardwareStats = showHardwareStatsRaw === "true";' in source
    assert 'id="hardware-stats-toggle"' in source
    assert 'fetch("/system-stats")' in source
    assert "setInterval(refresh, 2000)" in source
    assert "clearInterval(timer)" in source
    assert "if (inFlight) return;" in source
    assert "Number.isFinite" in source
    for label in ("显示硬件占用", "CPU", "系统内存", "GPU", "显存", "温度"):
        assert label in source


def test_every_user_preference_is_restored_from_persistent_storage() -> None:
    source = _ui_source()

    expected_state_keys = {
        "themeId": "ui_theme",
        "engine": "engine",
        "avatarId": "avatar_id",
        "outputQuality": "output_quality",
        "rtxSuperResolutionRaw": "rtx_super_resolution",
        "openaiKey": "openai_api_key",
        "openaiModel": "openai_model",
        "openaiVoice": "openai_voice",
        "openaiPrompt": "openai_prompt",
        "openaiLocalTtsEnabledRaw": "openai_local_tts_enabled",
        "codexVoice": "codex_voice",
        "codexPrompt": "codex_prompt",
        "codexLocalTtsEnabledRaw": "codex_local_tts_enabled",
        "customLlmUrl": "custom_llm_url",
        "customLlmKey": "custom_llm_key",
        "customLlmModel": "custom_llm_model",
        "customAsrUrl": "custom_asr_url",
        "customAsrKey": "custom_asr_key",
        "customAsrModel": "custom_asr_model",
        "customTtsUrl": "custom_tts_url",
        "customTtsKey": "custom_tts_key",
        "customTtsModel": "custom_tts_model",
        "customTtsVoice": "custom_tts_voice",
        "cloudLocalTtsEnabledRaw": "cloud_local_tts_enabled",
        "customPrompt": "custom_prompt",
        "customProvider": "custom_provider",
        "bailianApiKey": "bailian_api_key",
        "bailianWorkspaceId": "bailian_workspace_id",
        "bailianLlmModel": "bailian_llm_model",
        "bailianTtsModel": "bailian_tts_model",
        "bailianTtsVoice": "bailian_tts_voice",
        "bailianWebSearchMode": "bailian_web_search_mode",
        "bailianThinkingMode": "bailian_thinking_mode",
        "customVadMode": "custom_vad_mode",
        "minimaxApiKey": "minimax_api_key",
        "minimaxRealtimeModel": "minimax_realtime_model",
        "minimaxVoice": "minimax_voice",
        "preserveBackgroundRaw": "preserve_background",
        "micId": "mic_id",
        "showHardwareStatsRaw": "hardware_stats_enabled",
        "autoReleaseModelsRaw": "auto_release_models",
    }
    for state_name, storage_key in expected_state_keys.items():
        assert re.search(
            rf"const \[{re.escape(state_name)},\s*set[A-Za-z0-9_]+\]\s*=\s*"
            rf'useLocalStorage\(\s*"{re.escape(storage_key)}",',
            source,
        ), f"{state_name} is not persisted under {storage_key}"

    assert set(re.findall(r'useLocalStorage\(\s*"([^"]+)"', source)) == set(
        expected_state_keys.values()
    )
    assert 'const preserveBackground = preserveBackgroundRaw === "true";' in source


def test_persisted_values_are_validated_and_invalid_legacy_values_fall_back() -> None:
    source = _ui_source()

    assert "function validateStoredText(value)" in source
    assert "function enumStorageValue(values)" in source
    assert 'const BOOLEAN_STORAGE_VALUE = enumStorageValue(["false", "true"]);' in source
    assert "const LOCAL_STORAGE_VALIDATORS = Object.freeze({" in source
    for key in (
        "ui_theme",
        "engine",
        "output_quality",
        "rtx_super_resolution",
        "openai_voice",
        "codex_voice",
        "custom_provider",
        "openai_local_tts_enabled",
        "codex_local_tts_enabled",
        "cloud_local_tts_enabled",
        "preserve_background",
        "hardware_stats_enabled",
        "auto_release_models",
        "bailian_web_search_mode",
        "bailian_thinking_mode",
        "custom_vad_mode",
        "custom_llm_url",
        "custom_asr_url",
        "custom_tts_url",
        "avatar_id",
        "mic_id",
    ):
        assert re.search(rf"\b{re.escape(key)}:\s*[A-Za-z0-9_(]", source), (
            f"missing restore validator for {key}"
        )

    hook_body = source.split("function useLocalStorage(key, initial) {", 1)[1].split(
        "function redactEngineCredentials", 1
    )[0]
    assert "LOCAL_STORAGE_VALIDATORS[key] || validateStoredText" in hook_body
    assert "const normalized = validateStoredValue(candidate);" in hook_body
    assert "return normalized === null ? initial : normalized;" in hook_body


def test_removed_avatar_and_microphone_preferences_recover_to_available_values() -> None:
    source = _ui_source()

    assert "const nextAvatarId = avatarsList.includes(avatarId)" in source
    assert 'avatarsList[0] || ""' in source
    assert "if (nextAvatarId !== avatarId) setAvatarId(nextAvatarId);" in source
    assert "const selectedMicAvailable = mics.some((mic) => mic.deviceId === micId);" in source
    assert "if (!selectedMicAvailable) setMicId(mics[0].deviceId);" in source


def test_transient_runtime_and_form_state_is_never_restored_from_local_storage() -> None:
    source = _ui_source()

    transient_state_names = (
        "open",
        "settingsOpen",
        "status",
        "liveVideoReady",
        "audioTrack",
        "active",
        "muted",
        "startError",
        "uploadState",
        "codexStatus",
        "engineConnectionId",
        "connectedEngineType",
        "connectedEngineScope",
        "explicitDisconnectGeneration",
        "engineConnectionSignature",
        "connectionStatus",
        "connectionComponents",
        "connectionBusy",
        "previewBusy",
        "credentialRevisions",
        "voiceCloneName",
        "voiceCloneAudio",
        "voiceCloneTranscript",
        "voiceCloneConsent",
        "voiceCloneBusy",
        "voiceCloneStatus",
        "bailianClonePrefix",
        "bailianCloneAudio",
        "bailianCloneAudioUrl",
        "bailianCloneConsent",
        "minimaxCloneVoiceId",
        "minimaxCloneAudio",
        "minimaxClonePreviewText",
        "minimaxCloneConsent",
        "cloudCloneBusy",
        "cloudCloneStatus",
        "localVoices",
        "localVoicesStatus",
        "avatarDeleteBusy",
        "avatarDeleteStatus",
        "voiceDeleteBusy",
        "voiceDeleteStatus",
        "localVoicePreviewBusy",
        "localVoicePreviewStatus",
        "modelReleaseBusy",
        "modelReleaseState",
    )
    for state_name in transient_state_names:
        assert re.search(
            rf"const \[{re.escape(state_name)},\s*set[A-Za-z0-9_]+\]\s*=\s*useState\(",
            source,
        ), f"{state_name} must remain transient"
        assert not re.search(
            rf"const \[{re.escape(state_name)},\s*set[A-Za-z0-9_]+\]\s*=\s*useLocalStorage\(",
            source,
        )


def test_model_release_preference_defaults_on_and_uses_strict_boolean_restore() -> None:
    source = _ui_source()

    assert (
        'const [autoReleaseModelsRaw, setAutoReleaseModelsRaw] = '
        'useLocalStorage("auto_release_models", "true");'
        in source
    )
    assert 'const autoReleaseModels = autoReleaseModelsRaw === "true";' in source
    assert "auto_release_models: BOOLEAN_STORAGE_VALUE" in source
    assert '<div class="card monitor-toggle model-release-card">' in source
    assert 'id="auto-release-models"' in source
    assert "checked=${autoReleaseModels}" in source
    assert (
        'setAutoReleaseModelsRaw(event.target.checked ? "true" : "false")'
        in source
    )
    assert "结束对话后自动释放模型内存" in source


def test_model_release_button_calls_aggregate_endpoint_and_is_session_safe() -> None:
    source = _ui_source()

    assert 'fetch("/system/release-models", { method: "POST" })' in source
    assert 'id="release-models-now"' in source
    assert "disabled=${active || modelReleaseBusy}" in source
    assert 'modelReleaseBusy || !engineConnectionStartable' in source
    assert "const modelReleaseBusyRef = useRef(false);" in source
    assert "if (modelReleaseBusyRef.current) return false;" in source

    release = source.split(
        'const releaseModels = useCallback(async (reason = "manual") => {', 1
    )[1].split("const releaseAfterSession", 1)[0]
    assert "setModelReleaseBusy(true);" in release
    finally_block = release.split("} finally {", 1)[1]
    assert "modelReleaseBusyRef.current = false;" in finally_block
    assert "setModelReleaseBusy(false);" in finally_block


def test_session_cleanup_finishes_before_optional_automatic_model_release() -> None:
    source = _ui_source()

    assert "const releaseAfterSession = useCallback(async (generation) => {" in source
    release_after = source.split(
        "const releaseAfterSession = useCallback(async (generation) => {", 1
    )[1].split("const finishSession", 1)[0]
    assert "if (!autoReleaseModels) return;" in release_after
    assert 'await releaseModels("automatic");' in release_after

    finish = source.split("const finishSession = useCallback((reason) => {", 1)[1].split(
        "const stop", 1
    )[0]
    assert finish.index("teardown();") < finish.index("releaseAfterSession(generation);")


def test_auto_release_enters_startable_standby_and_resumes_from_one_click() -> None:
    source = _ui_source()

    release_after = source.split(
        "const releaseAfterSession = useCallback(async (generation) => {", 1
    )[1].split("const finishSession", 1)[0]
    assert 'await releaseModels("automatic");' in release_after
    assert "recoveryGenerationRef.current !== generation" in release_after
    assert "!reconnectAfterSessionRef.current" in release_after
    assert 'setConnectionStatus("standby");' in release_after

    assert "const engineConnectionStandby = Boolean(" in source
    assert "const engineConnectionStartable = Boolean(" in source
    assert "engineConnectionReady || engineConnectionStandby" in source
    assert 'connectionStatus === "standby" ? ENGINE_CONNECTION_LABELS.standby' in source
    assert "resumeSessionAfterConnectRef.current = true;" in source
    assert re.search(
        r"if \(engineConnectionStandby\) \{\s*"
        r"resumeSessionAfterConnectRef\.current = true;\s*"
        r"connectEngine\(\);\s*return;",
        source,
    )
    assert re.search(
        r"!engineConnectionReady \|\| !resumeSessionAfterConnectRef\.current"
        r"\) return;\s*resumeSessionAfterConnectRef\.current = false;\s*start\(\);",
        source,
    )
    assert "!engineConnectionStartable" in source
    assert 'disabled=${sessionActive || connectionBusy || !viewingConnectedEngine}' in source


def test_session_finish_is_single_shot_for_manual_and_remote_end() -> None:
    source = _ui_source()

    assert "const sessionRunningRef = useRef(false);" in source
    finish = source.split("const finishSession = useCallback((reason) => {", 1)[1].split(
        "const stop", 1
    )[0]
    assert "if (!sessionRunningRef.current) return;" in finish
    assert "sessionRunningRef.current = false;" in finish
    assert "const generation = ++recoveryGenerationRef.current;" in finish
    assert "void cleanupPreparedConnection(engineConnectionId);" in finish
    assert "void releaseAfterSession(generation);" in finish
    assert 'setConnectionStatus("disconnected");' in finish

    stop = source.split("const stop = useCallback(() => {", 1)[1].split(
        "const handleRemoteEnd", 1
    )[0]
    remote_end = source.split("const handleRemoteEnd = useCallback(() => {", 1)[1].split(
        "const start", 1
    )[0]
    assert 'finishSession("manual");' in stop
    assert 'finishSession("remote");' in remote_end


def test_engine_connection_attempts_discard_late_server_connections() -> None:
    source = _ui_source()
    connect = re.search(
        r"const connectEngine\s*=\s*useCallback\(async\s*\(\)\s*=>\s*\{"
        r"(?P<body>.*?)\n\s*\},\s*\[",
        source,
        re.DOTALL,
    )
    assert connect is not None
    body = connect.group("body")

    assert "const isCurrentConnectionAttempt = () =>" in body
    assert "connectionAttemptRef.current === attemptId" in body
    assert body.count("if (!isCurrentConnectionAttempt())") >= 5
    assert "await discardPreparedConnectionResponse(response);" in body
    assert "await cleanupPreparedConnection(connection.connection_id);" in body
    assert "let preparedConnectionId = null;" in body
    assert "preparedConnectionId = connection.connection_id;" in body
    assert "await cleanupPreparedConnection(preparedConnectionId);" in body
    assert "resumeSessionAfterConnectRef.current = false;" in body


def test_model_release_reports_released_partial_and_lazy_reload_states() -> None:
    source = _ui_source()

    assert "function collectModelReleaseResults(payload)" in source
    assert "payload?.services" in source
    assert 'payload?.status === "partial"' in source
    for label in ("释放中", "已释放", "部分失败", "将在下次会话重新加载", "自动恢复"):
        assert label in source
    assert 'id="model-release-status"' in source
    assert 'role="status"' in source
    assert 'aria-live="polite"' in source


def test_qianwen_web_search_mode_migrates_the_legacy_boolean_and_is_persistent() -> None:
    source = _ui_source()

    assert (
        'const [bailianWebSearchMode, setBailianWebSearchMode] = '
        'useLocalStorage("bailian_web_search_mode", initialBailianWebSearchMode());'
        in source
    )
    assert 'localStorage.getItem("bailian_web_search_enabled")' in source
    assert 'return legacy === "true" ? "auto" : "off";' in source
    assert 'bailian_web_search_mode: enumStorageValue(["off", "auto", "always"])' in source


def test_qianwen_web_search_control_is_scoped_to_qianwen_and_explains_costs() -> None:
    source = _ui_source()

    assert source.count('id="bailian-web-search-mode"') == 1
    qianwen_panel = source.split(
        '<div class="endpoint-title">千问 LLM + CosyVoice</div>', 1
    )[1].split("<details class=\"voice-clone\">", 1)[0]
    assert 'id="bailian-web-search-mode"' in qianwen_panel
    assert "value=${bailianWebSearchMode}" in qianwen_panel
    assert (
        "setBailianWebSearchMode(event.target.value)"
        in qianwen_panel
    )
    for value in ("off", "auto", "always"):
        assert f'value="{value}"' in qianwen_panel
    assert "联网搜索" in qianwen_panel
    assert "开启会增加响应延迟、输入 token 与搜索调用费用" in qianwen_panel
    assert "Qwen3.5–3.8 Max/Plus/Flash 会自动走 Responses API" in qianwen_panel


def test_qianwen_web_search_is_sent_and_invalidates_connected_config_signature() -> None:
    source = _ui_source()

    build = source.split("const buildEngineConfig = useCallback(() => {", 1)[1].split(
        "let currentEngineSignature", 1
    )[0]
    provider = build.split('kind: "aliyun_bailian"', 1)[1].split("},", 1)[0]
    assert "web_search_mode: bailianWebSearchMode" in provider
    dependencies = build.rsplit("],", 1)[0]
    assert "bailianWebSearchMode" in dependencies
    assert "currentEngineSignature = connectionSignatureFor(" in source
    assert "engineConnectionSignature !== currentEngineSignature" in source


def test_custom_api_vad_mode_is_persisted_and_sent_as_220_320_or_550_ms() -> None:
    source = _ui_source()

    assert (
        'const [customVadMode, setCustomVadMode] = '
        'useLocalStorage("custom_vad_mode", "fast");'
        in source
    )
    assert "function vadStorageValue(value)" in source
    assert "if (value === \"low_latency\") return \"fast\";" in source
    assert "custom_vad_mode: vadStorageValue" in source
    assert "const VAD_SILENCE_MS = Object.freeze({ ultra: 220, fast: 320, balanced: 550 });" in source
    assert 'id="custom-vad-mode"' in source
    assert 'value="ultra"' in source
    assert 'value="fast"' in source
    assert 'value="balanced"' in source
    build = source.split("const buildEngineConfig = useCallback(() => {", 1)[1].split(
        "let currentEngineSignature", 1
    )[0]
    assert "silence_ms: VAD_SILENCE_MS[customVadMode] || VAD_SILENCE_MS.fast" in build
    assert "vad," in build
    assert "customVadMode" in build.rsplit("],", 1)[0]


def test_qianwen_thinking_mode_is_explicit_persisted_and_sent_to_provider() -> None:
    source = _ui_source()

    assert (
        'const [bailianThinkingMode, setBailianThinkingMode] = '
        'useLocalStorage("bailian_thinking_mode", "fast");'
        in source
    )
    assert 'bailian_thinking_mode: enumStorageValue(["fast", "deep"])' in source
    assert 'id="bailian-thinking-mode"' in source
    assert 'value="fast"' in source
    assert 'value="deep"' in source
    build = source.split("const buildEngineConfig = useCallback(() => {", 1)[1].split(
        "let currentEngineSignature", 1
    )[0]
    provider = build.split('kind: "aliyun_bailian"', 1)[1].split("},", 1)[0]
    assert "thinking_mode: bailianThinkingMode" in provider
    assert "bailianThinkingMode" in build.rsplit("],", 1)[0]


def test_local_clone_overrides_request_incremental_websocket_text() -> None:
    source = _ui_source()

    local_override = source.split("const buildLocalTtsOverride = useCallback(() => {", 1)[1].split(
        "const buildEngineConfig", 1
    )[0]
    cloud_override = source.split("const ttsOverride = cloudLocalTtsEnabled ? {", 1)[1].split(
        "} : null;", 1
    )[0]
    generic_tts = source.split('if (customProvider !== "generic")', 1)[1].split(
        "throw new Error(`", 1
    )[1]

    assert 'streaming_text: "websocket"' in local_override
    assert 'streaming_text: "websocket"' in cloud_override
    assert 'streaming_text: "websocket"' not in generic_tts


def test_performance_overlay_polls_and_displays_latest_turn_latency() -> None:
    source = _ui_source()

    assert "function useTurnMetrics(enabled, live)" in source
    assert 'fetch("/turn-metrics")' in source
    assert "setInterval(refresh, 500)" in source
    assert "clearInterval(timer)" in source
    assert "turnMetrics=${turnMetrics}" in source
    assert "function HardwareStatsOverlay({ enabled, state, fps, live, turnMetrics })" in source
    for key in (
        "speech_end_to_frame",
        "asr",
        "llm_ttft",
        "chunk_wait",
        "tts_ttfa",
        "renderer",
        "speech_end_to_audible",
    ):
        assert key in source
    assert "turnMetrics?.latest" in source
    assert "turnMetrics?.summary" in source
    assert "fast_slo_complete_turns" in source
    assert "P50" in source
    assert "P95" in source
    assert 'class="deep-thinking-slo-excluded"' in source


def test_browser_reports_each_first_non_silent_remote_audio_transition() -> None:
    source = _ui_source()

    assert "function monitorFirstAudibleRemoteAudio(" in source
    assert "audioContext.createMediaStreamSource(remoteStream)" in source
    assert "analyser.getFloatTimeDomainData(samples)" in source
    assert "FIRST_AUDIBLE_RMS_THRESHOLD" in source
    assert 'fetch("/turn-metrics/browser-first-audible", { method: "POST" })' in source
    assert "audibleMonitorCleanupRef" in source
    assert "monitorFirstAudibleRemoteAudio(remoteStream" in source


def test_hardware_monitor_floats_outside_settings_and_reports_displayed_fps() -> None:
    source = _ui_source()

    assert "function useVideoFps(videoRef, enabled, live)" in source
    assert "requestVideoFrameCallback" in source
    assert "cancelVideoFrameCallback" in source
    assert "metadata?.presentedFrames" in source
    assert (
        'const videoFps = useVideoFps(videoRef, showHardwareStats, status === "live");'
        in source
    )
    assert '<aside class="hardware-monitor-overlay"' in source
    assert 'aria-label="实时性能监控"' in source
    monitor = source.split('<aside class="hardware-monitor-overlay"', 1)[1].split(
        "</aside>", 1
    )[0]
    assert 'aria-live="polite"' not in monitor
    assert 'metric("实时帧率"' in source
    assert "markStalled();\n          frameCallbackId = video.requestVideoFrameCallback(onFrame);" in source

    settings = source.split('<div class="settings-scroll">', 1)[1].split("</aside>", 1)[0]
    assert 'id="hardware-stats-toggle"' in settings
    assert "<${HardwareStatsOverlay}" not in settings
    assert source.count("<${HardwareStatsOverlay}") == 1


def test_interface_is_fully_localized_to_simplified_chinese() -> None:
    source = _ui_source()

    assert '<html lang="zh-CN">' in source
    for text in ("DigiBox 本地互动", "人物", "背景", "麦克风", "开始会话", "网络连通性", "运行日志"):
        assert text in source
    for legacy in (
        ">Local Stream<",
        ">Avatar<",
        ">Background<",
        "Start session",
        "<summary>Log</summary>",
        "Session timed out",
        "Connectivity</div>",
    ):
        assert legacy not in source
    for mapping in (
        "CANDIDATE_TYPE_LABELS",
        "CONNECTION_STATE_LABELS",
        "MEDIA_KIND_LABELS",
        "ENGINE_NAME_LABELS",
    ):
        assert mapping in source
    assert 'text.includes("Failed to fetch")' in source
    assert 'text.includes("NetworkError when attempting")' in source
    assert "await readApiError(r)" in source
    assert 'key === "openai_prompt" && v === LEGACY_DEFAULT_OPENAI_PROMPT' in source


def test_local_connectivity_card_marks_turn_relay_as_unneeded_not_failed() -> None:
    source = _ui_source()

    assert (
        'const isLocalDeployment = ["127.0.0.1", "localhost", "::1"].includes('
        "window.location.hostname);"
        in source
    )
    assert 'row("中继候选（relay，通过 TURN）：本地直连，无需使用", "neutral")' in source
    assert '.probe li.neutral { color: var(--muted); }' in source


def test_ui_uploads_one_person_image_with_background_preservation_choice() -> None:
    source = _ui_source()

    assert 'id="avatar-upload"' in source
    assert 'id="preserve-background"' in source
    assert 'id="background-upload"' not in source
    assert source.count('accept="image/png,image/jpeg,image/webp"') == 1
    assert 'new FormData()' in source
    assert 'form.append("preserve_background", String(preserveBackground));' in source
    assert 'fetch("/assets/avatar"' in source
    assert "await registry.refresh()" in source
    assert "setAvatarId(data.id)" in source
    assert "setBackgroundId" not in source
    assert 'const DEFAULT_TECH_BACKGROUND_ID = "tech_particles_dark";' in source


def test_selected_avatar_is_previewed_below_the_upload_controls() -> None:
    source = _ui_source()

    assert 'id="avatar-preview"' in source
    assert '`/avatars/${encodeURIComponent(avatarId)}/preview`' in source
    assert 'class="avatar-preview"' in source
    assert "当前人物预览" in source


def test_custom_api_can_register_a_local_cosyvoice_clone() -> None:
    source = _ui_source()

    for field_id in (
        "voice-clone-name",
        "voice-clone-audio",
        "voice-clone-transcript",
        "voice-clone-consent",
        "voice-clone-create",
    ):
        assert f'id="{field_id}"' in source
    assert 'form.append("reference_audio", voiceCloneAudio, voiceCloneAudio.name);' in source
    assert 'form.append("consent", "true");' in source
    assert 'fetch(`${baseUrl}/audio/voices`' in source
    assert 'setCustomTtsVoice(String(data.id));' in source
    assert "await loadLocalVoices(baseUrl, String(data.id));" in source
    assert 'http://127.0.0.1:8768/v1' in source
    assert 'Fun-CosyVoice3-0.5B-2512' in source


def test_local_cosyvoice_uses_a_refreshable_quality_aware_voice_dropdown() -> None:
    source = _ui_source()

    assert "const loadLocalVoices = useCallback" in source
    assert 'fetch(`${baseUrl}/audio/voices`' in source
    assert 'id="custom-tts-voice"' in source
    assert 'id="cloud-local-tts-voice"' in source
    assert 'id="local-voices-refresh"' in source
    assert "voice.selectable === false" in source
    assert "参考音频异常，请重新创建" in source
    assert "const invalidLocalVoiceBlocksConnection" in source
    assert "uploadBusy || invalidLocalVoiceBlocksConnection" in source


def test_invalid_local_voice_remains_selectable_for_deletion() -> None:
    source = _ui_source()

    assert 'disabled=${voice.selectable === false}' not in source
    assert source.count('<option value=${voice.id}>') >= 2
    assert "selectedLocalVoiceInvalid" in source
    assert 'id="local-voice-delete"' in source
    assert 'id="cloud-local-voice-delete"' in source


def test_local_cosyvoice_voice_can_be_previewed_before_connecting() -> None:
    source = _ui_source()

    assert "function pcm16ToWavBlob" in source
    assert "const previewSelectedLocalVoice = useCallback" in source
    assert 'fetch(`${localCosyBaseUrl}/audio/speech`' in source
    assert 'response.headers.get("X-Audio-Sample-Rate")' in source
    assert "await response.arrayBuffer()" in source
    assert "URL.createObjectURL(wavBlob)" in source
    assert 'id="local-voice-preview"' in source
    assert 'id="cloud-local-voice-preview"' in source
    assert 'id="local-voice-preview-audio"' in source
    assert "selectedLocalVoiceInvalid" in source


def test_custom_api_provider_selector_preserves_generic_mode_and_adds_cloud_presets() -> None:
    source = _ui_source()

    assert 'useLocalStorage("custom_provider", "generic")' in source
    assert 'id="custom-provider"' in source
    assert '<option value="generic">通用兼容接口</option>' in source
    assert '<option value="aliyun_bailian">千问 AI 平台（单 Key）</option>' in source
    assert 'id="qianwen-platform-link"' in source
    assert 'href="https://platform.qianwenai.com/"' in source
    assert 'target="_blank"' in source
    assert '<option value="minimax">' not in source
    assert "const MINIMAX_PROVIDER_ENABLED = false;" in source
    assert 'customProvider === "generic"' in source

    # Selecting a cloud preset must not remove the existing independently
    # configurable OpenAI-compatible endpoints or local CosyVoice clone flow.
    for field_id in (
        "custom-llm-url",
        "custom-llm-key",
        "custom-asr-url",
        "custom-asr-key",
        "custom-tts-url",
        "custom-tts-key",
        "voice-clone-use-local",
    ):
        assert f'id="{field_id}"' in source


def test_bailian_preset_sends_one_key_and_current_full_chain_defaults() -> None:
    source = _ui_source()

    assert 'useLocalStorage("bailian_api_key", "")' in source
    assert 'const [bailianApiKey, setBailianApiKey] = useState("");' not in source
    assert source.count('id="bailian-api-key"') == 1
    assert 'id="bailian-region"' not in source
    assert 'useLocalStorage("bailian_region"' not in source
    assert 'ap-southeast-1' not in source
    assert 'region: "cn-beijing"' in source
    assert 'id="bailian-workspace-id"' in source
    for default in (
        '"qwen3.7-flash"',
        '"qwen3-asr-flash-realtime"',
        '"cosyvoice-v3-flash"',
        '"longanhuan_v3"',
    ):
        assert default in source

    assert re.search(
        r'kind:\s*"aliyun_bailian".*?api_key:\s*bailianApiKey.*?'
        r'asr_model:\s*DEFAULT_BAILIAN_ASR_MODEL.*?'
        r'tts_voice:\s*bailianTtsVoice',
        source,
        re.DOTALL,
    )
    assert (
        '...(bailianWorkspaceId.trim() ? { workspace_id: bailianWorkspaceId.trim() } : {})'
        in source
    )
    assert 'workspace_id: (bailianWorkspaceId || "").trim()' not in source
    assert 'if (bailianTtsVoice === "longanyang")' in source
    assert "20 ms" in source
    assert "WebSocket" in source


def test_minimax_native_provider_is_hidden_and_cannot_be_selected() -> None:
    source = _ui_source()

    assert '<option value="minimax">' not in source
    assert 'if (customProvider === "minimax") setCustomProvider("generic");' in source
    assert 'MINIMAX_PROVIDER_ENABLED && customProvider === "minimax"' in source
    assert 'useLocalStorage("minimax_api_key", "")' in source
    assert 'const [minimaxApiKey, setMinimaxApiKey] = useState("");' not in source
    assert source.count('id="minimax-api-key"') == 1
    assert "https://api.minimaxi.com" in source
    assert '"abab6.5s-chat"' in source
    assert '"male-qn-qingse"' in source
    assert 'kind: "minimax"' in source  # dormant backend retained for a verified API later
    assert re.search(
        r'kind:\s*"minimax".*?api_key:\s*minimaxApiKey.*?'
        r'realtime_model:\s*minimaxRealtimeModel.*?voice:\s*minimaxVoice',
        source,
        re.DOTALL,
    )
    assert 'kind: "bailian_minimax"' not in source
    assert "MiniMax（百炼单 Key）" not in source
    assert "三项共用当前百炼 API Key" not in source
    assert "官方历史接口" in source
    assert "需账号具备权限" in source


def test_all_provider_api_keys_have_independent_persistent_storage_keys() -> None:
    source = _ui_source()

    for storage_key in (
        "openai_api_key",
        "custom_llm_key",
        "custom_asr_key",
        "custom_tts_key",
        "bailian_api_key",
        "minimax_api_key",
    ):
        assert f'useLocalStorage("{storage_key}",' in source
    assert "只保存在当前页面会话，刷新后需重新填写" not in source
    assert "保存在当前浏览器的此地址下" in source


def test_cloud_provider_voice_clone_forms_use_their_native_audio_inputs_and_proxy() -> None:
    source = _ui_source()

    for field_id in (
        "bailian-clone-prefix",
        "bailian-clone-audio-file",
        "bailian-clone-audio-url",
        "bailian-clone-consent",
        "bailian-clone-create",
        "minimax-clone-voice-id",
        "minimax-clone-audio-file",
        "minimax-clone-preview-text",
        "minimax-clone-consent",
        "minimax-clone-create",
    ):
        assert f'id="{field_id}"' in source

    assert 'fetch("/provider-voice-clones"' in source
    assert 'form.append("reference_audio", bailianCloneAudio' in source
    assert "setBailianTtsVoice(createdVoiceId);" in source
    assert "setMinimaxVoice(createdVoiceId);" in source
    assert "公网可访问" in source
    assert 'form.append("reference_audio", minimaxCloneAudio' in source
    assert 'form.append("consent", "true")' in source
    assert "语音服务将自动应用新音色" in source
    assert 'fetch("/provider-voice-clones/query"' in source


def test_cloud_providers_can_override_only_tts_with_the_local_cloned_voice() -> None:
    source = _ui_source()

    assert 'useLocalStorage("cloud_local_tts_enabled", "false")' in source
    assert 'id="cloud-local-tts-toggle"' in source
    assert 'id="cloud-local-tts-url"' in source
    assert 'id="cloud-local-tts-model"' in source
    assert 'id="cloud-local-tts-voice"' in source
    assert 'id="cloud-local-voice-clone-audio"' in source
    assert "云端 ASR 与 LLM 保持不变" in source
    assert "TTS 不再请求服务商" in source
    assert re.search(
        r"const ttsOverride = cloudLocalTtsEnabled.*?"
        r"base_url: ttsUrl.*?model: ttsModel.*?voice: ttsVoice",
        source,
        re.DOTALL,
    )
    assert source.count("...(ttsOverride ? { tts_override: ttsOverride } : {})") == 2


def test_openai_and_codex_offer_persistent_local_cosyvoice_hybrid_modes() -> None:
    source = _ui_source()

    assert 'useLocalStorage("openai_local_tts_enabled", "false")' in source
    assert 'useLocalStorage("codex_local_tts_enabled", "false")' in source
    assert 'openaiLocalTtsEnabledRaw === "true"' in source
    assert 'codexLocalTtsEnabledRaw === "true"' in source
    assert 'setOpenaiLocalTtsEnabledRaw(enabled ? "true" : "false")' in source
    assert 'setCodexLocalTtsEnabledRaw(enabled ? "true" : "false")' in source
    assert 'renderRealtimeLocalTts("openai"' in source
    assert 'renderRealtimeLocalTts("codex"' in source
    assert "OpenAI / Codex 只负责理解与生成文本" in source
    assert "云端语音播放及数字人驱动已屏蔽" in source
    assert "本地 CosyVoice 流式合成并驱动数字人" in source

    assert "const buildLocalTtsOverride = useCallback(() =>" in source
    assert "...(openaiLocalTtsEnabled ? { tts_override: buildLocalTtsOverride() } : {})" in source
    assert "...(codexLocalTtsEnabled ? { tts_override: buildLocalTtsOverride() } : {})" in source
    assert "openaiLocalTtsEnabled, codexLocalTtsEnabled" in source
    for fragment in (
        'throw new Error("请填写本地 TTS API 地址、模型并选择本地音色")',
        "timeout_seconds: 120",
        'response_format: "pcm"',
        "sample_rate: 24000",
        "auth: { api_key: (customTtsKey || \"\").trim() }",
    ):
        assert fragment in source
    for fragment in (
        '${provider}-local-tts-voice',
        '${provider}-local-voice-preview',
        'onClick=${previewSelectedLocalVoice}',
        'ref=${localVoicePreviewAudioRef}',
    ):
        assert fragment in source
    bridge_start = source.index('if (selectedEngine === "codex") {')
    bridge_end = source.index('sessionPc.addTransceiver("video"', bridge_start)
    bridge_block = source[bridge_start:bridge_end]
    assert 'createDataChannel("avtr-codex-audio")' in bridge_block
    assert 'if (!codexLocalTtsEnabled)' in bridge_block
    assert 'type === "input_audio_buffer.speech_started" && !state.userTurnActive' in source
    assert bridge_block.index('if (!codexLocalTtsEnabled)') < bridge_block.index(
        "bridgeCodexAudio("
    )
    assert '!codexLocalTtsEnabled && codexPreviewAudioRef.current' in source


def test_provider_tts_fields_remain_editable_while_local_override_is_enabled() -> None:
    source = _ui_source()

    for field_id in ("bailian-tts-model", "minimax-voice"):
        field = re.search(rf'<input id="{field_id}"(?P<attrs>.*?)/>', source, re.DOTALL)
        assert field is not None
        assert "cloudLocalTtsEnabled" not in field.group("attrs")
    bailian_voice = re.search(
        r'<select id="bailian-tts-voice"(?P<attrs>.*?)>', source, re.DOTALL
    )
    assert bailian_voice is not None
    assert "cloudLocalTtsEnabled" not in bailian_voice.group("attrs")


def test_switching_engine_tabs_keeps_the_existing_connection_alive() -> None:
    source = _ui_source()

    assert 'const [connectedEngineType, setConnectedEngineType] = useState(null);' in source
    assert 'const [connectedEngineScope, setConnectedEngineScope] = useState(null);' in source
    assert "connectedEngineType === engine" in source
    assert "connectedEngineScope === currentEngineScope" in source
    assert "const selectedConnectionStale = Boolean(" in source
    assert "setConnectedEngineType(engineConfig.type);" in source
    assert "setConnectedEngineScope(credentialScope);" in source
    assert "setConnectedEngineType(null);" in source
    assert "setConnectedEngineScope(null);" in source
    assert 'setConnectionStatus("stale")' not in source

    tabs = re.search(r'<div class="tabs">(?P<body>.*?)</div>', source, re.DOTALL)
    assert tabs is not None
    assert 'onClick=${() => setEngine(t.id)}' in tabs.group("body")
    assert "disconnectEngine" not in tabs.group("body")


def test_credential_changes_only_invalidate_their_own_engine_chain() -> None:
    source = _ui_source()

    assert "const [credentialRevisions, setCredentialRevisions] = useState({});" in source
    assert "credentialScopeFor(engine, customProvider)" in source
    assert 'updateCredential("openai", setOpenaiKey, value)' in source
    assert source.count('updateCredential("custom_api:generic",') >= 2
    assert "const updateSharedTtsCredential = useCallback((value) =>" in source
    assert "customTtsKey=${customTtsKey} setCustomTtsKey=${updateSharedTtsCredential}" in source
    assert 'updateCredential("custom_api:aliyun_bailian", setBailianApiKey, value)' in source
    assert 'updateCredential("custom_api:minimax", setMinimaxApiKey, value)' in source


def test_failed_connection_attempt_does_not_poison_a_held_connection() -> None:
    source = _ui_source()
    connect = re.search(
        r"const connectEngine\s*=\s*useCallback\(async\s*\(\)\s*=>\s*\{"
        r"(?P<body>.*?)\n\s*\},\s*\[",
        source,
        re.DOTALL,
    )
    assert connect is not None
    body = connect.group("body")

    assert "const hadExistingConnection = Boolean(engineConnectionId);" in body
    assert "let replacementStarted = false;" in body
    assert body.index("const engineConfig = buildEngineConfig();") < body.index(
        'setConnectionStatus("testing");'
    )
    assert body.index('method: "DELETE"') < body.index('setConnectionStatus("testing");')
    assert "if (!hadExistingConnection || replacementStarted)" in body


def test_codex_stale_configuration_is_not_displayed_as_connected() -> None:
    source = _ui_source()

    assert re.search(
        r"const statusLabel = otherEngineConnected.*?"
        r": selectedConnectionStale\s*\? ENGINE_CONNECTION_LABELS\.stale.*?"
        r": engine === \"codex\"",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"const statusClass = otherEngineConnected.*?"
        r": selectedConnectionStale\s*\? \"stale\"",
        source,
        re.DOTALL,
    )


def test_uploaded_avatar_and_local_voice_have_confirmed_delete_actions() -> None:
    source = _ui_source()

    assert 'id="avatar-delete-button"' in source
    assert 'id="local-voice-delete"' in source
    assert 'id="cloud-local-voice-delete"' in source
    assert 'window.confirm(`确定删除人物“${selectedAvatarId}”吗？' in source
    assert 'fetch(`/assets/avatar/${encodeURIComponent(selectedAvatarId)}`, {' in source
    assert 'window.confirm(`确定删除本地音色“${selectedVoiceId}”吗？' in source
    assert 'fetch(`${localCosyBaseUrl}/audio/voices/${encodeURIComponent(selectedVoiceId)}`, {' in source
    assert 'method: "DELETE"' in source


def test_cloud_provider_controls_do_not_expand_the_two_button_landing_page() -> None:
    source = _ui_source()
    launch = re.search(
        r'<div class=\$\{`launch-controls.*?>(?P<body>.*?)'
        r'\n\s*</div>',
        source,
        re.DOTALL,
    )

    assert launch is not None
    body = launch.group("body")
    assert body.count("<button") == 2
    assert 'id="open-settings"' in body
    assert 'id="start-conversation"' in body
    assert "custom-provider" not in body
    assert "bailian" not in body.lower()
    assert "minimax" not in body.lower()


def test_engine_connection_signature_never_retains_raw_api_keys() -> None:
    source = _ui_source()

    assert "function connectionSignatureFor" in source
    assert 'key === "api_key"' in source
    assert "credentialRevision" in source
    assert "JSON.stringify(buildEngineConfig())" not in source
    assert "const signature = JSON.stringify(engineConfig);" not in source


def test_session_cannot_start_while_an_asset_is_still_processing() -> None:
    source = _ui_source()

    assert 'id="start-conversation"' in source
    assert 'uploadState?.status === "uploading"' in source


def test_immersive_stage_uses_particle_background_and_settings_drawer() -> None:
    source = _ui_source()

    assert 'url("/assets/preset-background")' in source
    assert 'id="open-settings"' in source
    assert 'id="start-conversation"' in source
    assert 'class="settings-drawer"' in source
    assert "stage-video-shell" in source
    assert "preserve-background" in source
    assert "radial-gradient" in source
    assert 'output_aspect: "16:9"' in source


def test_live_hud_does_not_squash_status_or_end_button_in_narrow_windows() -> None:
    source = _ui_source()

    assert "width: min(640px, calc(100vw - 32px));" in source
    assert ".stage-hud .status-pill," in source
    assert ".stage-hud .end-session" in source
    assert "white-space: nowrap;" in source
    assert 'showHardwareStats ? " monitor-visible" : ""' in source
    assert ".monitor-visible .stage-hud" in source
    assert ".monitor-visible .launch-controls" in source
    assert "right: clamp(14px, 2.2vw, 30px);" in source


def test_codex_failure_status_is_not_overwritten_by_close_events() -> None:
    source = _ui_source()

    preserve_failure = re.findall(
        r'setCodexStatus\(\s*\(current\)\s*=>\s*current === "failed"\s*'
        r'\? current : "disconnected"\s*\)',
        source,
    )
    assert len(preserve_failure) >= 2


def test_openai_engine_config_still_requires_and_sends_its_api_key() -> None:
    source = _ui_source()
    openai_start = 'if (engine === "openai") {'
    custom_start = 'if (engine === "custom_api") {'

    assert openai_start in source
    openai_and_following = source.split(openai_start, 1)[1]
    assert custom_start in openai_and_following
    openai_config = openai_and_following.split(custom_start, 1)[0]

    assert 'if (!apiKey) throw new Error("请输入 OpenAI API 密钥");' in openai_config
    assert 'type: "openai"' in openai_config
    assert "api_key: apiKey" in openai_config


def test_codex_connect_creates_a_second_peer_before_session_start() -> None:
    source = _ui_source()

    assert "const codexPcRef = useRef(null);" in source
    assert re.search(r"const codexPc = new RTCPeerConnection\(", source)
    assert "codexPcRef.current = codexPc;" in source
    assert re.search(
        r"codexPc\.addTrack\(\s*track(?:,\s*localStream)?\s*\)",
        source,
    )
    assert 'codexPc.createDataChannel("oai-events")' in source
    connect_block = re.search(
        r"const connectEngine\s*=\s*useCallback\(async\s*\(\)\s*=>\s*\{(?P<body>.*?)\n\s*\},\s*\[",
        source,
        re.DOTALL,
    )
    assert connect_block is not None
    assert "createCodexPeer" in connect_block.group("body")
    assert 'fetch("/engine-connections"' in connect_block.group("body")


def test_codex_sdp_is_exchanged_during_preconnection_not_offer() -> None:
    source = _ui_source()
    connect_body = re.search(
        r'fetch\("/engine-connections",\s*\{.*?'
        r"body:\s*JSON\.stringify\(\{(?P<body>.*?)\}\),",
        source,
        re.DOTALL,
    )

    assert connect_body is not None
    assert re.search(
        r"\bcodex_sdp:\s*codexPc\.localDescription\.sdp\b",
        connect_body.group("body"),
    ), "preconnection must send the browser-owned Codex SDP"

    remote_answer = re.search(
        r"await codexPc\.setRemoteDescription\((?P<answer>.*?)\);",
        source,
        re.DOTALL,
    )
    assert remote_answer is not None
    assert "connection.codex_sdp" in remote_answer.group("answer")


def test_codex_audio_is_bridged_over_a_named_avtr_data_channel() -> None:
    source = _ui_source()

    assert (
        'const codexAudioChannel = sessionPc.createDataChannel("avtr-codex-audio");'
        in source
    )
    assert re.search(r"codexPc\.ontrack\s*=", source)
    assert "AudioContext" in source
    assert "createMediaStreamSource" in source
    assert "codexAudioChannel.send(" in source
    assert "codexAudioContextRef.current = audioContext;" in source


def test_codex_v3_turn_events_map_to_avtr_bridge_controls() -> None:
    source = _ui_source()

    assert 'type === "input_transcript.added"' in source
    assert re.search(
        r'type\s*===\s*"turn\.done"\s*&&\s*'
        r'event\.turn\?\.role\s*===\s*"assistant"',
        source,
    )
    assert 'type: "speech_started"' in source
    assert 'type: "output_audio_done"' in source


def test_codex_v3_user_turn_completion_resets_transcript_interrupt_state() -> None:
    source = _ui_source()

    assert re.search(
        r'type\s*===\s*"turn\.done"\s*&&\s*'
        r'event\.turn\?\.role\s*===\s*"user"',
        source,
    )


def test_teardown_closes_the_codex_peer_and_audio_context() -> None:
    source = _ui_source()

    assert "codexPcRef.current?.close();" in source
    assert "codexAudioContextRef.current?.close();" in source
    assert "codexPcRef.current = null;" in source
    assert "codexAudioContextRef.current = null;" in source


def test_theme_registry_exposes_completed_aurora_and_removes_dawn_placeholder() -> None:
    source = _ui_source()

    assert 'const DEFAULT_THEME_ID = "starfield";' in source
    assert "const THEME_REGISTRY = Object.freeze" in source
    assert 'id: "starfield", label: "星空"' in source
    assert 'id: "aurora", label: "极光"' in source
    assert '{ id: "dawn", label: "晨曦"' not in source
    assert ".theme-swatch.dawn" not in source
    assert 'role="radiogroup" aria-label="界面主题"' in source
    assert 'disabled=${!theme.available}' in source


def test_theme_registry_exposes_all_bundled_background_themes() -> None:
    source = _ui_source()

    themes = (
        ("starfield", "星空", "tech_particles_dark"),
        ("aurora", "极光", "theme_aurora"),
        ("winter-hearth", "冬日火炉", "theme_winter_hearth"),
        ("romantic", "绯红浪漫", "theme_romantic"),
        ("cozy-cabin", "晨光小屋", "theme_cozy_cabin"),
        ("pearl", "珍珠柔光", "theme_pearl"),
        ("cyberspace", "赛博空间", "theme_cyberspace"),
        ("rainforest", "翡翠雨林", "theme_rainforest"),
    )
    for theme_id, label, renderer_background_id in themes:
        match = re.search(
            rf'\{{\s*id:\s*"{re.escape(theme_id)}",(?P<body>[^}}]*)\}}',
            source,
        )
        assert match is not None, f"missing theme registry entry for {theme_id}"
        body = match.group("body")
        assert f'label: "{label}"' in body
        assert (
            f'rendererBackgroundId: "{renderer_background_id}"' in body
        ), f"{theme_id} does not map to {renderer_background_id}"
        assert "available: true" in body


def test_offer_uses_active_theme_renderer_background_without_session_override() -> None:
    source = _ui_source()

    assert "background_id: activeTheme.rendererBackgroundId," in source
    assert "sessionBackgroundId" not in source


def test_bundled_background_themes_style_every_immersive_surface() -> None:
    source = _ui_source()
    theme_ids = (
        "aurora",
        "winter-hearth",
        "romantic",
        "cozy-cabin",
        "pearl",
        "cyberspace",
        "rainforest",
    )
    required_tokens = (
        "--bg",
        "--panel",
        "--panel-2",
        "--border",
        "--text",
        "--muted",
        "--accent",
        "--accent-hover",
        "--theme-stage-background-image",
        "--theme-button-background",
        "--theme-button-primary-background",
        "--theme-button-particle-accent",
        "--theme-button-particle-animation",
        "--theme-settings-surface",
        "--theme-settings-divider",
        "--theme-monitor-surface",
        "--theme-monitor-border",
        "--theme-monitor-indicator",
        "--theme-hud-surface",
        "--theme-focus-ring",
    )

    for theme_id in theme_ids:
        match = re.search(
            rf':root\[data-theme="{re.escape(theme_id)}"\]\s*\{{(?P<body>.*?)\n\s*\}}',
            source,
            re.DOTALL,
        )
        assert match is not None, f"missing CSS variables for {theme_id}"
        block = match.group("body")
        assert f'url("/assets/theme-background/{theme_id}")' in block
        for token in required_tokens:
            assert token in block, f"{theme_id} does not define {token}"


def test_stage_background_covers_viewport_and_crops_horizontally() -> None:
    source = _ui_source()

    assert "--theme-stage-background-size: cover;" in source
    assert "--theme-stage-background-size: contain;" not in source
    assert "--theme-stage-background-position: center center;" in source
    assert "background-position: var(--theme-stage-background-position);" in source
    assert "background-size: var(--theme-stage-background-size);" in source


def test_selected_theme_is_persisted_and_applied_to_the_document_root() -> None:
    source = _ui_source()

    assert '<html lang="zh-CN">' in source
    assert 'useLocalStorage("ui_theme", DEFAULT_THEME_ID)' in source
    assert "document.documentElement.dataset.theme = activeThemeId;" in source
    assert "THEME_REGISTRY.find((theme) => theme.id === themeId && theme.available)" in source
    assert '<${ThemePicker} themeId=${activeThemeId} onChange=${setThemeId} />' in source


def test_starfield_theme_tokens_drive_all_immersive_surfaces() -> None:
    source = _ui_source()

    assert ':root[data-theme="starfield"]' in source
    for token in (
        "--theme-stage-background-image",
        "--theme-button-background",
        "--theme-button-primary-background",
        "--theme-button-particle-animation",
        "--theme-settings-surface",
        "--theme-monitor-surface",
    ):
        assert token in source

    for declaration in (
        "background-image: var(--theme-stage-background-image);",
        "background: var(--theme-button-background);",
        "background: var(--theme-button-primary-background);",
        "animation: var(--theme-button-particle-animation);",
        "background: var(--theme-settings-surface);",
        "background: var(--theme-monitor-surface);",
    ):
        assert declaration in source


def test_hardware_monitor_collapses_on_short_viewports_without_covering_controls() -> None:
    source = _ui_source()

    assert "@media (max-width: 1100px)" in source
    assert ".monitor-visible .launch-controls" in source
    assert ".monitor-visible .stage-hud" in source
    assert "@media (max-width: 720px)" in source
    assert "bottom: calc(clamp(28px, 7vh, 72px) + 68px);" in source
    assert "@media (max-width: 640px)" in source

    assert "@media (max-height: 720px)" in source
    assert "max-height: min(220px, calc(100dvh - 112px));" in source
    assert ".hardware-monitor-overlay .metric:nth-child(n+5)" in source
    assert "@media (max-height: 520px)" in source
    assert ".hardware-monitor-overlay .metric:nth-child(n+3)" in source


def test_session_modal_receives_avatar_and_renders_poster_plus_idle_loop() -> None:
    source = _ui_source()
    session_modal = source.split("function SessionModal(", 1)[1].split(
        "function LogPanel", 1
    )[0]
    session_call = source.split("<${SessionModal}", 1)[1].split("/>", 1)[0]

    assert "avatarId" in session_modal.split(") {", 1)[0]
    assert "avatarId=${avatarId}" in session_call
    assert 'class="stage-avatar-poster"' in session_modal
    assert '`/avatars/${encodeURIComponent(avatarId)}/preview`' in session_modal

    idle_image = re.search(
        r'<img\b(?=[^>]*class="stage-avatar-idle-loop")[^>]*>',
        session_modal,
        re.DOTALL,
    )
    assert idle_image is not None
    assert '`/avatars/${encodeURIComponent(avatarId)}/idle-loop?recipe=v3`' in idle_image.group(0)
    assert 'alt=""' in idle_image.group(0)


def test_avatar_stage_stays_visible_before_session_and_crossfades_on_live_frame() -> None:
    source = _ui_source()
    session_modal = source.split("function SessionModal(", 1)[1].split(
        "function LogPanel", 1
    )[0]

    assert 'avatarId ? " has-avatar" : ""' in session_modal
    assert 'liveVideoReady ? " live-video-ready" : ""' in session_modal

    has_avatar = re.search(
        r"\.session-stage\.has-avatar\s*\{(?P<body>.*?)\}", source, re.DOTALL
    )
    assert has_avatar is not None
    assert "opacity: 1" in has_avatar.group("body")
    assert "visibility: visible" in has_avatar.group("body")

    live_video = re.search(
        r"\.stage-live-video\s*\{(?P<body>.*?)\}", source, re.DOTALL
    )
    assert live_video is not None
    assert "opacity: 0" in live_video.group("body")

    live_ready = re.search(
        r"\.session-stage\.live-video-ready\s+\.stage-live-video\s*"
        r"\{(?P<body>.*?)\}",
        source,
        re.DOTALL,
    )
    assert live_ready is not None
    assert "opacity: 1" in live_ready.group("body")

    idle_hidden = re.search(
        r"\.session-stage\.live-video-ready\s+\.stage-avatar-idle-loop\s*"
        r"\{(?P<body>.*?)\}",
        source,
        re.DOTALL,
    )
    assert idle_hidden is not None
    assert "opacity: 0" in idle_hidden.group("body")


def test_live_video_ready_waits_for_a_presented_frame_with_loadeddata_fallback() -> None:
    source = _ui_source()

    assert "const [liveVideoReady, setLiveVideoReady] = useState(false);" in source
    assert "liveVideoReady=${liveVideoReady}" in source
    assert "setLiveVideoReady(true)" in source
    first_frame_ready = source.index("setLiveVideoReady(true)")
    first_frame_logic = source[
        max(0, first_frame_ready - 1500) : first_frame_ready + 1500
    ]
    assert "requestVideoFrameCallback" in first_frame_logic
    assert 'addEventListener("loadeddata"' in first_frame_logic

    connection_state = source.split(
        "sessionPc.onconnectionstatechange = () => {", 1
    )[1].split("};", 1)[0]
    assert "setLiveVideoReady(true)" not in connection_state


def test_stopping_session_restores_the_idle_avatar_layer() -> None:
    source = _ui_source()
    teardown_and_stop = source.split("const teardown = useCallback(() => {", 1)[1].split(
        "const handleRemoteEnd", 1
    )[0]

    assert "setLiveVideoReady(false)" in teardown_and_stop
    assert "videoRef.current.srcObject = null" in teardown_and_stop


def test_reduced_motion_uses_the_static_avatar_poster() -> None:
    source = _ui_source()
    reduced_motion = source.split("@media (prefers-reduced-motion: reduce)", 1)[1].split(
        "</style>", 1
    )[0]

    idle_rule = re.search(
        r"\.stage-avatar-idle-loop\s*\{(?P<body>.*?)\}",
        reduced_motion,
        re.DOTALL,
    )
    assert idle_rule is not None
    assert "display: none" in idle_rule.group("body")

    poster_rule = re.search(
        r"\.stage-avatar-poster\s*\{(?P<body>.*?)\}",
        reduced_motion,
        re.DOTALL,
    )
    assert poster_rule is not None
    assert "opacity: 1" in poster_rule.group("body")


def test_session_attempt_guards_every_async_continuation_and_peer_callback() -> None:
    source = _ui_source()
    start = source.split("const start = useCallback(async () => {", 1)[1].split(
        "const switchMic", 1
    )[0]

    assert "const isCurrentMediaAttempt = () =>" in start
    guarded_awaits = (
        r"await fetchIceConfig\(\)",
        r"await navigator\.mediaDevices\.getUserMedia",
        r"await sessionPc\.createOffer\(\)",
        r"await sessionPc\.setLocalDescription",
        r"await waitForIceGatheringComplete\(sessionPc\)",
        r'await fetch\("/offer"',
        r"await readApiError\(response\)",
        r"await response\.json\(\)",
        r"await sessionPc\.setRemoteDescription",
    )
    for awaited in guarded_awaits:
        match = re.search(awaited, start)
        assert match is not None, f"missing awaited operation: {awaited}"
        continuation = start[match.end() : match.end() + 1000]
        assert "if (!isCurrentMediaAttempt())" in continuation, awaited

    ontrack = start.split("sessionPc.ontrack = (event) => {", 1)[1].split(
        "sessionPc.onconnectionstatechange", 1
    )[0]
    assert ontrack.index("if (!isCurrentMediaAttempt()) return;") < ontrack.index(
        "srcObject ="
    )

    connection_state = start.split(
        "sessionPc.onconnectionstatechange = () => {", 1
    )[1].split("};", 1)[0]
    assert "if (!isCurrentMediaAttempt()) return;" in connection_state

    outer_catch = start.rsplit("} catch (error) {", 1)[1]
    assert outer_catch.index("if (!isCurrentMediaAttempt())") < outer_catch.index(
        "setStartError(message)"
    )
    assert "abandonStaleMediaAttempt();" in outer_catch


def test_stale_session_attempt_cleanup_never_touches_current_global_refs() -> None:
    source = _ui_source()
    cleanup = source.split("function disposeStaleMediaAttempt(", 1)[1].split(
        "\n    function ", 1
    )[0]

    assert "pc.close()" in cleanup
    assert "ownedStream.getTracks()" in cleanup
    for forbidden in ("pcRef", "streamRef", "teardown", "setLiveVideoReady"):
        assert forbidden not in cleanup


def test_teardown_aborts_an_inflight_offer_without_touching_a_new_attempt() -> None:
    source = _ui_source()
    refs = source.split("function App()", 1)[1].split("const videoFps", 1)[0]
    teardown = source.split("const teardown = useCallback(() => {", 1)[1].split(
        "const stop", 1
    )[0]
    start = source.split("const start = useCallback(async () => {", 1)[1].split(
        "const switchMic", 1
    )[0]

    assert "const offerAbortRef = useRef(null);" in refs
    assert "offerAbortRef.current = null;" in teardown
    assert ".controller.abort()" in teardown
    assert "const offerAbort = new AbortController();" in start
    assert "offerAbortRef.current = { attempt: mediaAttempt, controller: offerAbort };" in start
    assert "signal: offerAbort.signal" in start
    assert "offerAbortRef.current?.attempt === mediaAttempt" in start


def test_first_frame_reveal_rejects_a_replaced_stream_from_the_same_attempt() -> None:
    source = _ui_source()
    reveal = source.split(
        "const revealLiveVideoAfterFirstFrame = useCallback(", 1
    )[1].split("const teardown", 1)[0]

    assert "expectedStream" in reveal
    assert "video.srcObject !== expectedStream" in reveal


def test_stage_media_cannot_cover_or_intercept_status_and_session_controls() -> None:
    source = _ui_source()
    media = re.search(
        r"\.stage-avatar-poster,\s*"
        r"\.stage-avatar-idle-loop,\s*"
        r"\.stage-live-video\s*\{(?P<body>.*?)\}",
        source,
        re.DOTALL,
    )
    assert media is not None
    assert "pointer-events: none" in media.group("body")

    expected_layers = {
        ".stage-status": "z-index: 3",
        ".stage-ended": "z-index: 4",
        ".stage-hud": "z-index: 5",
    }
    for selector, declaration in expected_layers.items():
        rule = re.search(
            rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", source, re.DOTALL
        )
        assert rule is not None
        assert declaration in rule.group("body")


def test_idle_loop_readiness_is_keyed_to_the_avatar_that_finished_loading() -> None:
    source = _ui_source()
    session_modal = source.split("function SessionModal(", 1)[1].split(
        "function LogPanel", 1
    )[0]

    assert "const [readyAvatarId, setReadyAvatarId] = useState(null);" in session_modal
    assert "readyAvatarId === avatarId" in session_modal
    assert "key=${avatarId}" in session_modal
    assert "setReadyAvatarId(avatarId)" in session_modal
    assert "setIdleLoopReady" not in session_modal


def test_local_memory_settings_explain_retention_and_silent_local_storage() -> None:
    source = _ui_source()

    assert "function MemorySettingsCard" in source
    assert "本地记忆" in source
    assert "静默保存在这台电脑" in source
    assert "只有低可信度候选会在 30 天未确认后自动清理" in source
    assert "正式记忆不会自动删除" in source
    assert 'fetch("/memory/stats"' in source
    assert 'useLocalStorage("memory' not in source
    assert 'localStorage.setItem("memory' not in source


def test_local_memory_manager_supports_delete_clear_and_cursor_filters() -> None:
    source = _ui_source()
    memory_card = source.split("function MemorySettingsCard", 1)[1].split(
        "function HardwareStatsOverlay", 1
    )[0]

    assert 'fetch(`/memory/records?${params.toString()}`' in memory_card
    assert 'params.set("kind", memoryKind)' in memory_card
    assert 'params.set("state", memoryState)' in memory_card
    assert 'params.set("cursor", cursor)' in memory_card
    assert 'fetch(`/memory/records/${encodeURIComponent(record.id)}?expected_revision=${record.revision}`' in memory_card
    assert 'method: "DELETE"' in memory_card
    assert 'confirmation: "清空全部本地记忆"' in memory_card
    assert 'fetch("/memory/clear"' in memory_card
    assert "永久保留" in memory_card


def test_local_memory_export_and_import_are_safe_merge_only() -> None:
    source = _ui_source()
    memory_card = source.split("function MemorySettingsCard", 1)[1].split(
        "function HardwareStatsOverlay", 1
    )[0]

    assert 'fetch("/memory/export"' in memory_card
    assert "URL.createObjectURL" in memory_card
    assert "URL.revokeObjectURL" in memory_card
    assert 'fetch("/memory/import/preview"' in memory_card
    assert 'fetch("/memory/import/apply"' in memory_card
    assert "await file.arrayBuffer()" in memory_card
    assert '"Content-Type": "application/json"' in memory_card
    assert "new FormData" not in memory_card
    assert "安全合并导入" in memory_card
    assert "只新增无冲突记忆，不覆盖或删除本机内容" in memory_card
    assert "重复跳过" in memory_card
    assert "冲突跳过" in memory_card
    assert "格式或字段无效会拒绝整个文件" in memory_card
    assert "无效：" not in memory_card
    assert "仅合并" in memory_card
    preview_at = memory_card.index('fetch("/memory/import/preview"')
    apply_at = memory_card.index('fetch("/memory/import/apply"')
    assert preview_at < apply_at


def test_local_memory_import_rejects_oversize_file_before_reading_bytes() -> None:
    source = _ui_source()
    memory_card = source.split("function MemorySettingsCard", 1)[1].split(
        "function HardwareStatsOverlay", 1
    )[0]

    size_guard_at = memory_card.index("if (file.size > 16 * 1024 * 1024)")
    array_buffer_at = memory_card.index("await file.arrayBuffer()")

    assert size_guard_at < array_buffer_at
    assert "记忆文件不能超过 16 MiB" in memory_card


def test_local_memory_delete_conflict_refreshes_stats_and_records() -> None:
    source = _ui_source()
    delete_handler = source.split("const deleteRecord = useCallback", 1)[1].split(
        "const clearAll = useCallback", 1
    )[0]

    assert "if (response.status === 409)" in delete_handler
    assert "await Promise.all([loadStats(), loadRecords()]);" in delete_handler


def test_local_memory_clear_conflict_refreshes_stats_and_records() -> None:
    source = _ui_source()
    clear_handler = source.split("const clearAll = useCallback", 1)[1].split(
        "const exportMemory = useCallback", 1
    )[0]

    assert "if (response.status === 409)" in clear_handler
    assert "await Promise.all([loadStats(), loadRecords()]);" in clear_handler
