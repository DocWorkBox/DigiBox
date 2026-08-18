from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_SOURCE = ROOT / "src-tauri" / "src" / "app.rs"


def _source() -> str:
    return APP_SOURCE.read_text(encoding="utf-8")


def test_external_backend_uses_the_same_generation_scoped_failure_recovery() -> None:
    source = _source()

    assert "start_external_backend_monitor" in source
    assert "transition_backend_failure" in source
    assert "startup_generation.load(Ordering::Acquire) != generation" in source

    transition = source[
        source.index("async fn transition_backend_failure") :
        source.index("fn start_external_backend_monitor")
    ]
    assert "create_splash_window" in transition
    assert "main.destroy()" in transition
    assert "BackendOwnership::Owned" in transition
    assert "stop_current_session" in transition
    assert transition.index("BackendOwnership::Owned") < transition.index(
        "stop_current_session"
    )


def test_every_ready_window_waits_for_the_root_document_before_creation() -> None:
    source = _source()
    ready_window = source[
        source.index("async fn show_ready_window") :
        source.index("async fn transition_backend_failure")
    ]

    assert "wait_for_app_root(app, generation).await?" in ready_window
    assert ready_window.index(
        "wait_for_app_root(app, generation).await?"
    ) < ready_window.index(
        "create_main_window(app)?"
    )


def test_root_readiness_wait_is_generation_scoped_and_owned_exit_aware() -> None:
    source = _source()
    guard = source[
        source.index("fn ensure_app_root_wait_active") :
        source.index("async fn wait_for_app_root")
    ]
    wait_root = source[
        source.index("async fn wait_for_app_root") :
        source.index("fn current_owned_exit")
    ]
    owned_exit = source[
        source.index("fn current_owned_exit") :
        source.index("async fn stop_current_session")
    ]

    assert "generation: u64" in wait_root
    assert "startup_generation.load(Ordering::Acquire) != generation" in guard
    assert "current_owned_exit(state)?" in guard
    assert wait_root.count("ensure_app_root_wait_active(&state, generation)?") >= 2

    # External backends are observed, never mistaken for a dead owned child.
    assert "session.ownership() != BackendOwnership::Owned" in owned_exit
    assert "return Ok(None);" in owned_exit
