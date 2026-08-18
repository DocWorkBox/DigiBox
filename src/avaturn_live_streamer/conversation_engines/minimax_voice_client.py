# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Current native MiniMax voice-cloning HTTP workflow."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from avaturn_live_streamer.conversation_engines.builders import MiniMaxProvider

_UPLOAD_URL = "https://api.minimaxi.com/v1/files/upload"
_CLONE_URL = "https://api.minimaxi.com/v1/voice_clone"
_MAX_AUDIO_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MiniMaxVoiceCloneResult:
    voice_id: str
    preview_url: str | None = None


def _ensure_success(payload: object, operation: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"MiniMax {operation} response must be an object")
    base_resp = payload.get("base_resp")
    if not isinstance(base_resp, dict):
        raise RuntimeError(f"MiniMax {operation} response is missing base_resp")
    status_code = base_resp.get("status_code")
    if status_code != 0:
        status_msg = base_resp.get("status_msg")
        raise RuntimeError(
            f"MiniMax {operation} failed ({status_code}): {status_msg}"
        )
    return payload


async def clone_minimax_voice(
    provider: MiniMaxProvider,
    *,
    filename: str,
    content_type: str,
    audio: bytes,
    voice_id: str,
    preview_text: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> MiniMaxVoiceCloneResult:
    if not audio:
        raise ValueError("MiniMax clone audio cannot be empty")
    if len(audio) > _MAX_AUDIO_BYTES:
        raise ValueError("MiniMax clone audio cannot exceed 20 MiB")
    headers = {"Authorization": f"Bearer {provider.api_key.get_secret_value()}"}
    async with httpx.AsyncClient(
        transport=transport,
        timeout=provider.timeout_seconds,
    ) as client:
        upload = await client.post(
            _UPLOAD_URL,
            headers=headers,
            data={"purpose": "voice_clone"},
            files={"file": (filename, audio, content_type)},
        )
        upload.raise_for_status()
        upload_payload = _ensure_success(upload.json(), "file upload")
        file_info = upload_payload.get("file")
        file_id = file_info.get("file_id") if isinstance(file_info, dict) else None
        if not isinstance(file_id, int):
            raise RuntimeError("MiniMax file upload response is missing file.file_id")

        clone_payload: dict[str, object] = {
            "file_id": file_id,
            "voice_id": voice_id,
        }
        if preview_text:
            clone_payload.update(
                text=preview_text,
                model="speech-2.8-turbo",
            )
        clone = await client.post(
            _CLONE_URL,
            headers=headers,
            json=clone_payload,
        )
        clone.raise_for_status()
        result = _ensure_success(clone.json(), "voice clone")
    preview_url = result.get("demo_audio")
    return MiniMaxVoiceCloneResult(
        voice_id=voice_id,
        preview_url=preview_url if isinstance(preview_url, str) and preview_url else None,
    )
