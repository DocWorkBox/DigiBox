# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Safely authenticate Hugging Face and verify access to AVTR-1 on Windows."""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable
from typing import Any

from huggingface_hub import get_hf_file_metadata, hf_hub_url, login


SUCCESS = 0
INVALID_TOKEN = 2
ACCESS_DENIED = 3
NETWORK_ERROR = 4
INPUT_ERROR = 5

MODEL_URL = hf_hub_url(
    "avaturn-live/avtr-1",
    "build_artifacts/avtr1.scripted.pt",
)


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _normalize_token(value: str) -> str | None:
    token = value.strip()
    if not token.startswith("hf_") or len(token) <= 3:
        return None
    if any(character.isspace() or not character.isprintable() for character in token):
        return None
    return token


def login_and_check(
    entered_token: str,
    *,
    login_fn: Callable[..., Any] = login,
    metadata_fn: Callable[..., Any] = get_hf_file_metadata,
    output: Callable[[str], Any] = print,
) -> int:
    """Save a personal token and check gated AVTR-1 access without echoing it."""
    token = _normalize_token(entered_token)
    if token is None:
        output("Invalid input. Paste only the raw hf_... personal token.")
        return INVALID_TOKEN

    try:
        login_fn(token=token, add_to_git_credential=False)
    except Exception as exc:
        if isinstance(exc, ValueError) or _status_code(exc) in {401, 403}:
            output("Hugging Face rejected the token as invalid.")
            return INVALID_TOKEN
        output("Hugging Face login failed because of a network, proxy, or TLS error.")
        return NETWORK_ERROR

    try:
        metadata_fn(MODEL_URL, token=token)
    except Exception as exc:
        if _status_code(exc) in {401, 403}:
            output(
                "Token accepted, but AVTR-1 access is denied. "
                "Accept the model terms in the browser first."
            )
            return ACCESS_DENIED
        output("AVTR-1 access check failed because of a network, proxy, or TLS error.")
        return NETWORK_ERROR

    output("Hugging Face login and AVTR-1 gated access check were successful.")
    return SUCCESS


def main() -> int:
    if os.environ.get("HF_TOKEN"):
        print(
            "Warning: HF_TOKEN is set in this process and can override the saved token."
        )
    print("Paste the raw personal token beginning with hf_. Input is hidden.")
    try:
        token = getpass.getpass("Hugging Face token: ")
    except (EOFError, KeyboardInterrupt):
        print("\nToken input was cancelled.")
        return INPUT_ERROR
    return login_and_check(token)


if __name__ == "__main__":
    raise SystemExit(main())
