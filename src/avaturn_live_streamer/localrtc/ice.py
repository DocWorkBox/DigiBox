# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Resolve ICE servers (STUN/TURN) for local-stream sessions.

Order of preference (first match wins for TURN; STUN is always included):

1. **Manual TURN**: `TURN_URL` (+ optional `TURN_USERNAME`, `TURN_CREDENTIAL`).
   Use with a self-hosted coturn or any standard TURN server.
2. **Cloudflare TURN**: `CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_TOKEN`.
   Auto-fetches short-lived credentials from Cloudflare's TURN service.
3. **STUN-only fallback**: Google's public STUN server. Sufficient if the
   server has a public IP and outbound UDP works, useless behind symmetric
   NAT or strict firewalls.

`STUN_URL` (default `stun:stun.l.google.com:19302`) overrides the STUN entry.
Set `STUN_URL=` (empty) to disable STUN.
"""

import os
from typing import TypedDict

import httpx
from aiortc import RTCIceServer

from avaturn_live_streamer.core.logs import get_logger


class IceServerJson(TypedDict, total=False):
    urls: list[str]
    username: str
    credential: str


def serialize_ice_servers(servers: list[RTCIceServer]) -> list[IceServerJson]:
    """JSON-serializable form of aiortc ``RTCIceServer`` for the browser."""
    out: list[IceServerJson] = []
    for s in servers:
        urls = s.urls if isinstance(s.urls, list) else [s.urls]
        entry: IceServerJson = {"urls": urls}
        if s.username is not None:
            entry["username"] = s.username
        if s.credential is not None:
            entry["credential"] = s.credential
        out.append(entry)
    return out


def has_turn(servers: list[RTCIceServer]) -> bool:
    for s in servers:
        urls = s.urls if isinstance(s.urls, list) else [s.urls]
        if any(u.startswith(("turn:", "turns:")) for u in urls):
            return True
    return False

_LOGGER = get_logger()

_CLOUDFLARE_TURN_CREDENTIALS_URL = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{kid}/credentials/generate"
)
_DEFAULT_STUN = "stun:stun.l.google.com:19302"


async def _fetch_cloudflare(
    kid: str, token: str, *, ttl_seconds: int = 86400
) -> list[RTCIceServer]:
    url = _CLOUDFLARE_TURN_CREDENTIALS_URL.format(kid=kid)
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"ttl": ttl_seconds},
        )
        r.raise_for_status()
        body = r.json()

    ice = body.get("iceServers")
    if not ice:
        raise RuntimeError(f"Cloudflare TURN response missing iceServers: {body!r}")
    urls = ice.get("urls") if isinstance(ice, dict) else None
    if not urls:
        raise RuntimeError(f"Cloudflare TURN response missing urls: {ice!r}")
    return [
        RTCIceServer(
            urls=urls if isinstance(urls, list) else [urls],
            username=ice.get("username"),
            credential=ice.get("credential"),
        )
    ]


async def resolve_ice_servers() -> list[RTCIceServer]:
    servers: list[RTCIceServer] = []

    stun_url = os.environ.get("STUN_URL", _DEFAULT_STUN)
    if stun_url:
        servers.append(RTCIceServer(urls=[stun_url]))

    turn_url = os.environ.get("TURN_URL")
    if turn_url:
        servers.append(
            RTCIceServer(
                urls=[turn_url],
                username=os.environ.get("TURN_USERNAME"),
                credential=os.environ.get("TURN_CREDENTIAL"),
            )
        )
        _LOGGER.info("ice: using manual TURN", url=turn_url)
        return servers

    kid = os.environ.get("CLOUDFLARE_TURN_KEY_ID")
    token = os.environ.get("CLOUDFLARE_TURN_KEY_TOKEN")
    if kid and token:
        try:
            servers.extend(await _fetch_cloudflare(kid, token))
            _LOGGER.info("ice: using Cloudflare TURN")
        except Exception as e:
            _LOGGER.warning(
                "ice: Cloudflare TURN fetch failed, falling back to STUN-only", error=str(e)
            )
        return servers

    _LOGGER.info("ice: STUN-only (no TURN credentials configured)", stun=stun_url)
    return servers
