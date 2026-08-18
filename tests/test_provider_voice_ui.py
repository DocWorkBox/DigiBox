from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "avaturn_live_streamer"
    / "local_stream_ui.html"
)


def _source() -> str:
    return UI_PATH.read_text(encoding="utf-8")


def _run_ui_function(name: str, expression: str) -> object:
    source = _source()
    start = source.index(f"function {name}(")
    match = re.search(r"\n    \}\n", source[start:])
    assert match is not None, f"top-level function {name} is not closed"
    function_source = source[start : start + match.end()]
    completed = subprocess.run(
        ["node", "-e", f"{function_source}\nconsole.log(JSON.stringify({expression}));"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_qianwen_voice_id_is_a_provider_inventory_select() -> None:
    source = _source()

    assert '<select id="bailian-tts-voice"' in source
    assert not re.search(r'<input id="bailian-tts-voice"', source)
    assert "bailianVoices.map" in source
    assert "longanhuan_v3（系统预设）" in source  # noqa: RUF001
    assert "voice.status !== \"OK\"" in source
    assert "voice.compatible === false" in source


def test_qianwen_voice_inventory_loads_only_for_the_connected_provider_scope() -> None:
    source = _source()

    assert 'fetch("/provider-voice-clones/query"' in source
    assert 'action: "list_voice"' not in source  # provider protocol stays server-side
    assert re.search(
        r'connectionStatus\s*!==\s*"ready".*?return;.*?'
        r'loadBailianVoices\(',
        source,
        re.DOTALL,
    )
    assert "connectedEngineType !== \"custom_api\"" in source
    assert 'connectedEngineScope !== "custom_api:aliyun_bailian"' in source
    assert "connectedEngineScope !== currentEngineScope" in source
    assert "selectedConnectionStale" in source


def test_deploying_qianwen_clone_is_polled_and_selected_only_after_ok() -> None:
    source = _source()

    start = source.index("const createCloudVoice = async (kind) => {")
    end = source.index("const renderRealtimeLocalTts", start)
    body = source[start:end]
    inventory_start = source.index("const loadBailianVoices = useCallback")
    inventory_end = source.index("const selectedLocalVoice", inventory_start)
    inventory_body = source[inventory_start:inventory_end]

    assert "mergeBailianVoice" in body
    assert 'String(data.status || "UNKNOWN").toUpperCase()' in body
    assert "waitForBailianVoiceReady" in body
    assert re.search(
        r'cloneResult\.outcome\s*===\s*"ready".*?setBailianTtsVoice',
        body,
        re.DOTALL,
    )
    assert 'status: "OK"' not in body
    assert "preferredVoice" not in inventory_body
    assert "BAILIAN_VOICE_POLL_TIMEOUT_MS" in inventory_body
    assert re.search(
        r'const pending = current\.filter\(.*?'
        r'\["DEPLOYING", "CREATING", "PENDING", "UNKNOWN"\].*?'
        r'return \[\.\.\.pending, \.\.\.nextVoices\];',
        inventory_body,
        re.DOTALL,
    )
    assert not re.search(
        r'nextVoices\.unshift\(\{.*?status:\s*"OK"',
        inventory_body,
        re.DOTALL,
    )
    assert "需重新连接并测试" not in body


def test_clone_operation_token_precedes_post_and_scope_churn_does_not_cancel() -> None:
    source = _source()
    create = source.split("const createCloudVoice = async (kind) => {", 1)[1].split(
        "const renderRealtimeLocalTts", 1
    )[0]
    poller = source.split(
        "const waitForBailianVoiceReady = useCallback", 1
    )[1].split("const selectedLocalVoice", 1)[0]
    cancellation = source.split(
        "const cancelBailianCloneOperation = useCallback", 1
    )[1].split("useEffect(() => () =>", 1)[0]

    token = "const cloneOperationToken = ++bailianCloneOperationGenerationRef.current;"
    assert token in create
    assert create.index(token) < create.index('fetch("/provider-voice-clones"')
    assert re.search(
        r"waitForBailianVoiceReady\(\s*createdVoiceId,\s*cloneOperationToken,?\s*\)",
        create,
    )
    assert "(voiceId, operationToken)" in poller
    assert "++bailianCloneOperationGenerationRef.current" not in poller
    assert "connectedEngineScope" not in cancellation
    assert "currentEngineScope" not in cancellation
    for dependency in (
        "customProvider",
        "bailianApiKey",
        "bailianWorkspaceId",
        "bailianTtsModel",
        "explicitDisconnectGeneration",
    ):
        assert dependency in cancellation


def test_clone_poll_has_a_hard_deadline_short_requests_and_error_cap() -> None:
    source = _source()
    poller = source.split(
        "const waitForBailianVoiceReady = useCallback", 1
    )[1].split("const selectedLocalVoice", 1)[0]

    assert "BAILIAN_VOICE_POLL_TIMEOUT_MS = 180000" in source
    assert "BAILIAN_INVENTORY_REQUEST_TIMEOUT_MS" in source
    assert "BAILIAN_VOICE_QUERY_ERROR_LIMIT" in source
    assert "const deadline = monotonicNow() + BAILIAN_VOICE_POLL_TIMEOUT_MS;" in poller
    assert "while (monotonicNow() < deadline)" in poller
    assert "Math.min(BAILIAN_INVENTORY_REQUEST_TIMEOUT_MS, remainingMs)" in poller
    assert "consecutiveQueryErrors >= BAILIAN_VOICE_QUERY_ERROR_LIMIT" in poller
    assert 'outcome: "query_failed"' in poller
    assert 'outcome: "timeout"' in poller


def test_post_success_clone_outcomes_have_specific_non_payment_retry_copy() -> None:
    source = _source()
    create = source.split("const createCloudVoice = async (kind) => {", 1)[1].split(
        "const renderRealtimeLocalTts", 1
    )[0]

    assert "let postSucceeded = false;" in create
    assert "postSucceeded = true;" in create
    assert "自动等待已停止" in create
    assert re.search(
        r'cloneResult\.outcome === "cancelled".*?'
        r'mergeBailianVoice\(cloneResult\.voice \|\| initialVoice\)',
        create,
        re.DOTALL,
    )
    assert "审核或部署失败" in create
    assert "与当前 TTS 模型不兼容" in create
    assert "仍在部署，可刷新音色列表继续查看" in create  # noqa: RUF001
    assert "状态查询失败，可刷新音色列表继续查看" in create  # noqa: RUF001
    assert "避免重复付费创建" in create
    assert re.search(
        r'catch \(error\).*?postSucceeded.*?'
        r'创建请求已成功受理.*?避免重复付费创建.*?'
        r'云端音色创建失败',
        create,
        re.DOTALL,
    )


def test_bailian_poll_decisions_execute_for_ready_pending_and_terminal_states() -> None:
    decisions = _run_ui_function(
        "bailianVoicePollDecision",
        "["
        "bailianVoicePollDecision(null),"
        "bailianVoicePollDecision({status:'DEPLOYING', compatible:true}),"
        "bailianVoicePollDecision({status:'OK', compatible:true}),"
        "bailianVoicePollDecision({status:'OK', compatible:false}),"
        "bailianVoicePollDecision({status:'UNDEPLOYED', compatible:true})"
        "]",
    )
    assert decisions == [
        "pending",
        "pending",
        "ready",
        "incompatible",
        "review_failed",
    ]


def test_qianwen_inventory_aborts_stale_requests_and_invalidates_generations() -> None:
    source = _source()

    assert "const bailianVoicesAbortRef = useRef(null);" in source
    assert "const bailianVoicesGenerationRef = useRef(0);" in source
    assert "new AbortController()" in source
    assert "signal: controller.signal" in source
    assert "bailianVoicesAbortRef.current?.abort();" in source
    assert "bailianVoicesGenerationRef.current += 1;" in source
    assert re.search(
        r'const isCurrentInventoryRequest = \(\) =>.*?'
        r'bailianVoicesGenerationRef\.current === generation.*?'
        r'!controller\.signal\.aborted',
        source,
        re.DOTALL,
    )
    assert re.search(
        r'useEffect\(\(\) => \(\) => \{\s*'
        r'cancelBailianCloneOperation\(\);',
        source,
    )


def test_clone_busy_locks_engine_and_provider_switches() -> None:
    source = _source()

    tabs = re.search(r'<div class="tabs">(?P<body>.*?)</div>', source, re.DOTALL)
    assert tabs is not None
    assert (
        "disabled=${sessionActive || cloudCloneBusy || voiceCloneBusy}"
        in tabs.group("body")
    )

    provider_start = source.index('<select id="custom-provider"')
    provider_opening = source[
        provider_start : source.index('<option value="generic">', provider_start)
    ]
    assert (
        "disabled=${sessionActive || cloudCloneBusy || voiceCloneBusy}"
        in provider_opening
    )


def test_releasing_voice_change_registers_a_delayed_reconnect() -> None:
    source = _source()

    change = source.split(
        "const changeVoiceWithReconnect = useCallback", 1
    )[1].split("const changeOpenaiVoice", 1)[0]
    assert '["ready", "standby", "releasing", "recovering", "testing", "in_session"]' in source
    assert "shouldArmVoiceReconnect" in change
    assert "reconnectAfterVoiceChangeRef.current = true;" in change

    reconnect_effect = source.split(
        "!reconnectAfterVoiceChangeRef.current", 1
    )[0].rsplit("useEffect(() => {", 1)[1]
    assert 'connectionStatus === "releasing"' in reconnect_effect


def test_voice_reconnect_snapshot_blocks_old_callbacks_after_explicit_disconnect() -> None:
    source = _source()
    decisions = _run_ui_function(
        "shouldArmVoiceReconnect",
        "["
        "shouldArmVoiceReconnect({engineType:null,scope:null,status:'disconnected',active:false},'custom_api','custom_api:aliyun_bailian'),"
        "shouldArmVoiceReconnect({engineType:'custom_api',scope:'custom_api:aliyun_bailian',status:'testing',active:false},'custom_api','custom_api:aliyun_bailian'),"
        "shouldArmVoiceReconnect({engineType:'custom_api',scope:'custom_api:aliyun_bailian',status:'recovering',active:false},'custom_api','custom_api:aliyun_bailian'),"
        "shouldArmVoiceReconnect({engineType:'custom_api',scope:'custom_api:aliyun_bailian',status:'in_session',active:true},'custom_api','custom_api:aliyun_bailian')"
        "]",
    )
    assert decisions == [False, True, True, True]

    disconnect = source.split("const disconnectEngine = useCallback(async () => {", 1)[1].split(
        "const connectEngine", 1
    )[0]
    assert "explicitDisconnectGenerationRef.current += 1;" in disconnect
    assert "setExplicitDisconnectGeneration" in disconnect
    assert "engineLifecycleSnapshotRef.current =" in disconnect
    assert disconnect.index("engineLifecycleSnapshotRef.current =") < disconnect.index("await fetch")

    change = source.split(
        "const changeVoiceWithReconnect = useCallback", 1
    )[1].split("const changeOpenaiVoice", 1)[0]
    assert "engineLifecycleSnapshotRef.current" in change
    assert "shouldArmVoiceReconnect" in change
    assert "connectedEngineType === engine" not in change
    assert "active &&" not in change

    finish = source.split("const finishSession = useCallback((reason) => {", 1)[1].split(
        "const stop", 1
    )[0]
    assert 'status: autoReleaseModels ? "releasing" : "recovering"' in finish


def test_standby_copy_promises_automatic_start_and_no_manual_reconnect() -> None:
    source = _source()

    assert 'standby: "待机（开始时自动连接）"' in source  # noqa: RUF001
    for stale_copy in (
        "需重新连接并测试",
        "请重新连接并测试",
        "需要重新连接语音链路",
    ):
        assert stale_copy not in source


def test_every_tts_voice_change_uses_the_shared_automatic_reconnect_path() -> None:
    source = _source()

    assert "const reconnectAfterVoiceChangeRef = useRef(false);" in source
    assert "const changeVoiceWithReconnect = useCallback" in source
    assert re.search(
        r'!selectedConnectionStale.*?'
        r'!reconnectAfterVoiceChangeRef\.current\s*\) return;.*?'
        r'connectEngine\(\);',
        source,
        re.DOTALL,
    )
    for prop in (
        "openaiVoice=${openaiVoice} setOpenaiVoice=${changeOpenaiVoice}",
        "codexVoice=${codexVoice} setCodexVoice=${changeCodexVoice}",
        "customTtsVoice=${customTtsVoice} setCustomTtsVoice=${changeCustomTtsVoice}",
        "bailianTtsVoice=${bailianTtsVoice} setBailianTtsVoice=${changeBailianTtsVoice}",
        "minimaxVoice=${minimaxVoice} setMinimaxVoice=${changeMinimaxVoice}",
    ):
        assert prop in source


def test_manual_connect_is_blocked_while_models_are_releasing() -> None:
    source = _source()

    assert "modelReleaseBusy," in source.split("function ConfigCard", 1)[1].split(") {", 1)[0]
    connect_button = source.split('id="engine-connect"', 1)[1].split("</button>", 1)[0]
    assert "modelReleaseBusy" in connect_button
    assert 'connectionStatus === "releasing"' in connect_button

    connect_engine = source.split("const connectEngine = useCallback(async () => {", 1)[1].split(
        "connectionBusyRef.current = true;", 1
    )[0]
    assert "modelReleaseBusyRef.current" in connect_engine
    assert 'connectionStatus === "releasing"' in connect_engine
    assert "modelReleaseBusy=${modelReleaseBusy}" in source
