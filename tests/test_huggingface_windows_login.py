from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TOKEN = "hf_test_token_value"
MODULE_PATH = Path("scripts/login_huggingface_windows.py")


def _load_login_module() -> Any:
    assert MODULE_PATH.exists(), "Windows Hugging Face login helper is missing"
    spec = importlib.util.spec_from_file_location(
        "login_huggingface_windows",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_login_saves_trimmed_token_and_checks_gated_model_access() -> None:
    login_module = _load_login_module()
    calls: list[tuple[str, str]] = []
    output: list[str] = []

    result = login_module.login_and_check(
        f"  {TOKEN}  ",
        login_fn=lambda *, token, add_to_git_credential: calls.append(
            ("login", token)
        ),
        metadata_fn=lambda url, *, token: calls.append((url, token)),
        output=output.append,
    )

    assert result == login_module.SUCCESS
    assert calls[0] == ("login", TOKEN)
    assert calls[1][1] == TOKEN
    assert any("AVTR-1" in line and "success" in line.lower() for line in output)
    assert TOKEN not in "\n".join(output)


def test_login_rejects_decorated_token_before_network_call() -> None:
    login_module = _load_login_module()
    called = False
    output: list[str] = []

    def unexpected_login(**_: object) -> None:
        nonlocal called
        called = True

    result = login_module.login_and_check(
        f"Bearer {TOKEN}",
        login_fn=unexpected_login,
        metadata_fn=lambda *_args, **_kwargs: None,
        output=output.append,
    )

    assert result == login_module.INVALID_TOKEN
    assert called is False
    assert any("raw hf_" in line.lower() for line in output)
    assert TOKEN not in "\n".join(output)


def test_login_reports_invalid_token_without_echoing_exception() -> None:
    login_module = _load_login_module()
    output: list[str] = []

    def invalid_login(**_: object) -> None:
        raise ValueError(f"Invalid token: {TOKEN}")

    result = login_module.login_and_check(
        TOKEN,
        login_fn=invalid_login,
        metadata_fn=lambda *_args, **_kwargs: None,
        output=output.append,
    )

    assert result == login_module.INVALID_TOKEN
    assert any("invalid" in line.lower() for line in output)
    assert TOKEN not in "\n".join(output)


def test_login_distinguishes_gated_access_denied() -> None:
    login_module = _load_login_module()
    output: list[str] = []
    forbidden = RuntimeError(f"request contained {TOKEN}")
    forbidden.response = SimpleNamespace(status_code=403)  # type: ignore[attr-defined]

    def deny_access(*_args: object, **_kwargs: object) -> None:
        raise forbidden

    result = login_module.login_and_check(
        TOKEN,
        login_fn=lambda **_kwargs: None,
        metadata_fn=deny_access,
        output=output.append,
    )

    assert result == login_module.ACCESS_DENIED
    assert any("access" in line.lower() and "terms" in line.lower() for line in output)
    assert TOKEN not in "\n".join(output)


def test_login_reports_network_failure_without_leaking_details() -> None:
    login_module = _load_login_module()
    output: list[str] = []

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"proxy exposed {TOKEN}")

    result = login_module.login_and_check(
        TOKEN,
        login_fn=lambda **_kwargs: None,
        metadata_fn=fail_network,
        output=output.append,
    )

    assert result == login_module.NETWORK_ERROR
    assert any("network" in line.lower() for line in output)
    assert TOKEN not in "\n".join(output)


def test_double_click_launcher_keeps_window_open_and_never_passes_token() -> None:
    launcher = Path("login_huggingface_windows.cmd").read_text(encoding="utf-8")
    source_launcher = Path("scripts/login_huggingface_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "pause" in launcher.lower()
    assert "login_huggingface_windows.ps1" in launcher
    assert "login_huggingface_windows.py" in source_launcher
    assert "dev_runtime_windows.ps1" in source_launcher
    assert "--token" not in launcher
    assert "HF_TOKEN" not in launcher
