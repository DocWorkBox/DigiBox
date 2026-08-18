from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TAURI_ROOT = ROOT / "src-tauri"
TAURI_FRONTEND = ROOT / "desktop" / "tauri"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing Tauri migration file: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_tauri_configuration_uses_a_local_splash_and_exact_windows_identity() -> None:
    config = json.loads(_read(TAURI_ROOT / "tauri.conf.json"))

    assert config["identifier"] == "live.avaturn.avtr1.desktop"
    assert config["productName"] == "DigiBox"
    assert config["build"]["frontendDist"] == "../desktop/tauri"
    assert config["app"]["windows"] == [
        {
            "label": "splash",
            "title": "DigiBox 正在启动",
            "url": "index.html",
            "width": 760,
            "height": 560,
            "minWidth": 680,
            "minHeight": 500,
            "resizable": True,
            "fullscreen": False,
            "visible": True,
        },
        {
            "label": "main",
            "title": "DigiBox",
            "url": "http://127.0.0.1:7860/",
            "create": False,
            "width": 1440,
            "height": 900,
            "minWidth": 900,
            "minHeight": 620,
            "resizable": True,
            "fullscreen": False,
            "visible": False,
        },
    ]


def test_remote_loopback_page_is_not_granted_tauri_ipc_or_shell_permissions() -> None:
    config = json.loads(_read(TAURI_ROOT / "tauri.conf.json"))
    security = config["app"]["security"]
    serialized = json.dumps(config, sort_keys=True)

    assert "dangerousRemoteDomainIpcAccess" not in security
    assert "shell:allow-spawn" not in serialized
    assert "shell:allow-execute" not in serialized
    assert "fs:allow" not in serialized
    assert "http://127.0.0.1:7860" not in json.dumps(
        security.get("capabilities", []), sort_keys=True
    )


def test_tauri_capability_is_limited_to_the_bundled_main_window() -> None:
    capability = json.loads(_read(TAURI_ROOT / "capabilities" / "default.json"))

    assert capability["windows"] == ["splash"]
    permissions = set(capability["permissions"])
    assert permissions == {"core:event:allow-listen", "core:event:allow-unlisten"}
    assert all(not item.startswith("shell:") for item in permissions)
    assert all(not item.startswith("fs:") for item in permissions)


def test_bundled_splash_exposes_only_the_desktop_command_surface() -> None:
    html = _read(TAURI_FRONTEND / "index.html")
    script = _read(TAURI_FRONTEND / "app.js")

    assert "Content-Security-Policy" in html
    assert "unsafe-eval" not in html
    assert "unsafe-inline" not in html
    for command in (
        "get_desktop_state",
        "retry_startup",
        "select_runtime",
        "open_logs",
    ):
        assert command in script
    for forbidden in ("Command.sidecar", "shell:allow", "fetch("):
        assert forbidden not in script


def test_rust_commands_reject_callers_after_navigation_to_the_remote_app() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")

    assert "ensure_trusted_splash" in source
    assert "window.label()" in source
    assert '"splash"' in source
    assert "is_trusted_splash_url" in source
    assert "get_desktop_state" in source
    assert "retry_startup" in source
    assert "select_runtime" in source
    assert "open_logs" in source


def test_remote_application_uses_a_separate_zero_capability_webview() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")
    capability_files = list((TAURI_ROOT / "capabilities").glob("*.json"))
    capabilities = [json.loads(_read(path)) for path in capability_files]

    assert "WebviewWindowBuilder::from_config" in source
    assert "on_navigation" in source
    assert "on_new_window" in source
    assert "get_webview_window(MAIN_LABEL)" in source
    assert all("main" not in capability.get("windows", []) for capability in capabilities)


def test_remote_links_reach_the_native_allowlisted_new_window_handler() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")

    # The remote application intentionally has no opener IPC capability.  The
    # opener plugin must therefore leave `_blank` clicks to WebView2 so the
    # native `on_new_window` callback can classify and open allowlisted URLs.
    assert "tauri_plugin_opener::Builder::new()" in source
    assert ".open_js_links_on_click(false)" in source
    assert "tauri_plugin_opener::init()" not in source
    assert "on_new_window" in source
    assert "open_external(&new_window_app, &url)" in source


def test_application_serializes_startup_and_preserves_external_ownership() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")

    assert "startup_gate" in source
    assert "BackendSession::external" in source
    assert "BackendOwnership::External" in source
    assert "BackendOwnership::Owned" in source
    assert "probe_avtr_service" in source
    assert "spawn_owned" in source
    assert "ExitRequested" in source
    assert "prevent_exit" in source


def test_ready_transition_destroys_splash_without_requesting_application_exit() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")
    ready_window = source[source.index("async fn show_ready_window") : source.index("async fn startup_inner")]

    assert "splash.destroy()" in ready_window
    assert "splash.close()" not in ready_window


def test_owned_backend_is_instance_bound_and_monitored_after_ready() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")

    assert "AVTR1_DESKTOP_INSTANCE_ID" in source
    assert "wait_for_owned_service(app, &instance_id)" in source
    assert "start_owned_backend_monitor" in source
    assert "startup_generation.load" in source
    assert "current_owned_exit" in source
    assert "create_splash_window" in source
    assert "main.destroy()" in source
    assert "stop_current_session" in source

    transition = source[
        source.index("async fn transition_backend_failure") :
        source.index("fn start_owned_backend_monitor")
    ]
    splash_error = transition[
        transition.index("if let Err(error) = create_splash_window") :
        transition.index("if let Some(main)")
    ]
    assert "return;" not in splash_error
    assert transition.index("update_state") < transition.index("create_splash_window")


def test_owned_backend_routes_mutable_user_data_outside_the_runtime() -> None:
    source = _read(TAURI_ROOT / "src" / "app.rs")

    assert "app_local_data_dir" in source
    assert "AVTR1_USER_ASSETS_ROOT" in source
    assert "AVTR1_COSYVOICE_SPEAKER_CACHE" in source
    assert 'join("user_assets")' in source
    assert 'join("cosyvoice").join("spk2info.pt")' in source


def test_tauri_shell_keeps_the_electron_fallback_during_migration() -> None:
    package = json.loads(_read(ROOT / "package.json"))

    scripts = package["scripts"]
    assert "desktop:dev" in scripts
    assert "desktop:dist" in scripts
    assert "tauri:dev" in scripts
    assert "tauri:build" in scripts
