# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared engine builders used by local CLI entry points.

Returns `(engine_config, engine_run)` for each supported engine kind, where
`engine_run` is the worklet-shaped callable `(EventBus, StreamClocks) -> Coroutine`.

Credentials are supplied inline via `EngineOptions` so the local UI can pass
them per-session instead of relying on process env vars.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.cartesia_client import CartesiaApiClient
from avaturn_live_streamer.conversation_engines.codex_realtime_client import (
    CodexConversationEngine,
    CodexRealtimeClient,
    build_codex_app_server_command,
)
from avaturn_live_streamer.conversation_engines.configs import (
    CartesiaConversationEngineConfig,
    CodexRealtimeConversationEngineConfig,
    ConversationEngineConfig,
    OpenAIRealtimeAPIConversationEngineConfig,
    OpenaiRealtimeApiVoice,
)
from avaturn_live_streamer.conversation_engines.realtime_api_client import RealtimeApiClient
from avaturn_live_streamer.event_bus import EventBus

EngineKind = Literal["openai", "custom_api", "codex"]
ENGINE_KINDS: tuple[EngineKind, ...] = ("openai", "custom_api", "codex")
CodexRealtimeVoice = Literal[
    "juniper",
    "maple",
    "spruce",
    "ember",
    "vale",
    "breeze",
    "arbor",
    "sol",
    "cove",
]

type EngineRun = Callable[[EventBus, StreamClocks], Coroutine[None, None, None]]
type BuiltEngine = tuple[ConversationEngineConfig, EngineRun]

_CARTESIA_TOKEN_URL = "https://api.cartesia.ai/access-token"
_CARTESIA_VERSION = "2025-04-16"

DEFAULT_OPENAI_PROMPT = (
    "You are a friendly, concise voice assistant. Speak naturally and keep "
    "answers under 50 words. Avoid emojis or unreadable symbols."
)
DEFAULT_OPENAI_VOICE: OpenaiRealtimeApiVoice = "shimmer"
DEFAULT_OPENAI_MODEL = "gpt-realtime-2"
DEFAULT_CODEX_PROMPT = DEFAULT_OPENAI_PROMPT
DEFAULT_CODEX_VOICE: CodexRealtimeVoice = "cove"


def append_memory_prompt(prompt: str, memory_prompt: str = "") -> str:
    """Append an already-bounded, untrusted memory block after the persona."""

    memory = memory_prompt.strip()
    if not memory:
        return prompt
    return f"{prompt}\n\n{memory}" if prompt.strip() else memory


class OpenAIEngineOptions(BaseModel):
    type: Literal["openai"] = "openai"
    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    prompt: str = DEFAULT_OPENAI_PROMPT
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE
    tts_override: TTSAPIOptions | None = None


class CartesiaEngineOptions(BaseModel):
    type: Literal["cartesia"] = "cartesia"
    api_key: str
    agent_id: str


class CodexEngineOptions(BaseModel):
    type: Literal["codex"] = "codex"
    voice: CodexRealtimeVoice = DEFAULT_CODEX_VOICE
    prompt: str = Field(default=DEFAULT_CODEX_PROMPT, max_length=8000)
    tts_override: TTSAPIOptions | None = None


class APIAuth(BaseModel):
    mode: Literal["bearer", "raw", "none"] = "bearer"
    api_key: SecretStr = SecretStr("")
    header_name: str = Field(default="Authorization", min_length=1, max_length=128)


class CompatibleEndpoint(BaseModel):
    base_url: AnyHttpUrl
    auth: APIAuth = Field(default_factory=APIAuth)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


class LLMAPIOptions(CompatibleEndpoint):
    model: str = Field(min_length=1, max_length=256)
    path: str = "/chat/completions"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class ASRAPIOptions(CompatibleEndpoint):
    model: str = Field(min_length=1, max_length=256)
    path: str = "/audio/transcriptions"
    language: str | None = Field(default=None, max_length=32)
    sample_rate: Literal[16_000, 24_000] = 16_000


class TTSAPIOptions(CompatibleEndpoint):
    model: str = Field(min_length=1, max_length=256)
    path: str = "/audio/speech"
    voice: str = Field(min_length=1, max_length=256)
    response_format: Literal["pcm", "wav"] = "pcm"
    sample_rate: int = Field(default=24_000, ge=8_000, le=48_000)
    instructions: str = Field(default="", max_length=2000)
    streaming_text: Literal["off", "websocket"] = "off"

    @field_validator("model", "voice")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


# The realtime option models are declared before TTSAPIOptions so the simple
# public API ordering stays intact. Resolve their postponed annotations now.
OpenAIEngineOptions.model_rebuild()
CodexEngineOptions.model_rebuild()


class GenericAPIProvider(BaseModel):
    kind: Literal["generic"] = "generic"
    llm: LLMAPIOptions
    asr: ASRAPIOptions
    tts: TTSAPIOptions


class AliyunBailianProvider(BaseModel):
    kind: Literal["aliyun_bailian"] = "aliyun_bailian"
    api_key: SecretStr
    workspace_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$",
    )
    region: Literal["cn-beijing"] = "cn-beijing"
    llm_model: str = Field(default="qwen3.7-flash", min_length=1, max_length=256)
    asr_model: str = Field(
        default="qwen3-asr-flash-realtime",
        min_length=1,
        max_length=256,
    )
    tts_model: str = Field(default="cosyvoice-v3-flash", min_length=1, max_length=256)
    tts_voice: str = Field(default="longanhuan_v3", min_length=1, max_length=256)
    asr_language: str | None = Field(default="zh", max_length=32)
    web_search_mode: Literal["off", "auto", "always"] = "off"
    thinking_mode: Literal["fast", "deep"] = "fast"
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_web_search(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "web_search_mode" in value:
            return value
        legacy = value.get("enable_web_search")
        if not isinstance(legacy, bool):
            return value
        migrated = dict(value)
        migrated.pop("enable_web_search", None)
        migrated["web_search_mode"] = "auto" if legacy else "off"
        return migrated

    @property
    def enable_web_search(self) -> bool:
        """Compatibility view for integrations that still inspect the old flag."""

        return self.web_search_mode != "off"

    @field_validator("workspace_id", mode="before")
    @classmethod
    def _normalise_workspace_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class MiniMaxProvider(BaseModel):
    kind: Literal["minimax"] = "minimax"
    api_key: SecretStr
    realtime_model: str = Field(
        default="abab6.5s-chat",
        min_length=1,
        max_length=256,
    )
    voice: str = Field(default="male-qn-qingse", min_length=1, max_length=256)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


CustomAPIProvider = Annotated[
    GenericAPIProvider | AliyunBailianProvider | MiniMaxProvider,
    Discriminator("kind"),
]


class VADOptions(BaseModel):
    rms_threshold: float = Field(default=0.015, ge=0.001, le=0.25)
    pre_roll_ms: int = Field(default=200, ge=0, le=1000)
    min_speech_ms: int = Field(default=160, ge=40, le=2000)
    silence_ms: int = Field(default=320, ge=100, le=3000)
    max_turn_seconds: int = Field(default=30, ge=2, le=120)


class CustomAPIConnectionConfig(BaseModel):
    type: Literal["custom_api"] = "custom_api"
    provider: CustomAPIProvider
    tts_override: TTSAPIOptions | None = None
    vad: VADOptions = Field(default_factory=VADOptions)
    prompt: str = Field(default="", max_length=8000)
    history_turns: int = Field(default=12, ge=0, le=50)
    fast_history_turns: int = Field(default=6, ge=4, le=6)

    model_config = ConfigDict(title="CustomAPIConnectionConfig")

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_generic_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "provider" in value:
            return value
        if not all(name in value for name in ("llm", "asr", "tts")):
            return value
        migrated = dict(value)
        migrated["provider"] = {
            "kind": "generic",
            "llm": migrated.pop("llm"),
            "asr": migrated.pop("asr"),
            "tts": migrated.pop("tts"),
        }
        return migrated

    def _generic_provider(self) -> GenericAPIProvider:
        if not isinstance(self.provider, GenericAPIProvider):
            raise AttributeError("llm/asr/tts are only available for the generic provider")
        return self.provider

    @property
    def llm(self) -> LLMAPIOptions:
        """Compatibility accessor for legacy generic-provider callers."""

        return self._generic_provider().llm

    @property
    def asr(self) -> ASRAPIOptions:
        """Compatibility accessor for legacy generic-provider callers."""

        return self._generic_provider().asr

    @property
    def tts(self) -> TTSAPIOptions:
        """Compatibility accessor for legacy generic-provider callers."""

        return self._generic_provider().tts


class CustomAPIEngineOptions(BaseModel):
    type: Literal["custom_api"] = "custom_api"
    connection_id: UUID


EngineOptions = Annotated[
    OpenAIEngineOptions | CustomAPIEngineOptions | CodexEngineOptions,
    Discriminator("type"),
]


async def _mint_cartesia_token(api_key: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.post(
            _CARTESIA_TOKEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": _CARTESIA_VERSION,
            },
            json={"grants": {"agent": True}, "expires_in": 300},
        )
        r.raise_for_status()
        token = r.json().get("token")
        if not token:
            raise RuntimeError("Cartesia access-token response missing 'token' field")
        return token


async def mint_openai_realtime_secret(
    *,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    prompt: str = DEFAULT_OPENAI_PROMPT,
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE,
    tracing: dict[str, object] | str | None = "auto",
    text_only: bool = False,
) -> str:
    from openai import AsyncClient

    oai = AsyncClient(api_key=api_key)
    audio: dict[str, object] = {
        "input": {
            "turn_detection": {
                "type": "semantic_vad",
                "eagerness": "high",
                "interrupt_response": True,
            },
        },
    }
    if not text_only:
        audio["output"] = {"voice": voice}
    session: dict[str, object] = {
        "type": "realtime",
        "model": model,
        "audio": audio,
    }
    if text_only:
        session["output_modalities"] = ["text"]
    if prompt.strip():
        session["instructions"] = prompt
    if tracing is not None:
        session["tracing"] = tracing
    secret = await oai.realtime.client_secrets.create(
        expires_after={"seconds": 7200, "anchor": "created_at"},
        session=session,  # pyright: ignore [reportArgumentType]
    )
    return secret.value


async def build_cartesia(
    *,
    stream_id: str,
    options: CartesiaEngineOptions,
) -> BuiltEngine:
    token = await _mint_cartesia_token(options.api_key)
    cfg = CartesiaConversationEngineConfig(access_token=token, agent_id=options.agent_id)
    return cfg, CartesiaApiClient(cfg, stream_id=stream_id).run


async def build_openai(
    *,
    stream_id: str,
    options: OpenAIEngineOptions,
    memory_prompt: str = "",
) -> BuiltEngine:
    tracing: dict[str, object] = {
        "workflow_name": "avaturn-live-local",
        "group_id": stream_id,
        "metadata": {"engine": "openai-realtime", "stream_id": stream_id},
    }
    secret = await mint_openai_realtime_secret(
        api_key=options.api_key,
        model=options.model,
        prompt=append_memory_prompt(options.prompt, memory_prompt),
        voice=options.voice,
        tracing=tracing,
        text_only=options.tts_override is not None,
    )
    cfg = OpenAIRealtimeAPIConversationEngineConfig(client_secret=secret)
    local_tts = None
    if options.tts_override is not None:
        from avaturn_live_streamer.conversation_engines.custom_api_client import (
            OpenAICompatibleTTS,
            probe_tts,
        )

        local_tts = OpenAICompatibleTTS(options.tts_override)
        try:
            await probe_tts(local_tts)
        except BaseException:
            await local_tts.close()
            raise
    return cfg, RealtimeApiClient(cfg, tts=local_tts).run


async def build_codex(
    *,
    stream_id: str,
    options: CodexEngineOptions,
    sdp: str,
    memory_prompt: str = "",
) -> BuiltEngine:
    workspace = Path(__file__).resolve().parents[3]
    client = CodexRealtimeClient(
        command=build_codex_app_server_command(),
        workspace=workspace,
    )
    local_tts = None
    try:
        if options.tts_override is not None:
            from avaturn_live_streamer.conversation_engines.custom_api_client import (
                OpenAICompatibleTTS,
                probe_tts,
            )

            local_tts = OpenAICompatibleTTS(options.tts_override)
            await probe_tts(local_tts)
        await client.start()
        answer_sdp = await client.start_realtime(
            sdp=sdp,
            prompt=append_memory_prompt(options.prompt, memory_prompt),
            voice=options.voice,
            # Codex's ChatGPT-authenticated WebRTC route currently accepts the
            # v3 audio protocol only.  In hybrid mode the browser and engine
            # discard that remote audio while assistant transcript deltas feed
            # the local TTS bridge, so CosyVoice remains the sole audible path.
            output_modality="audio",
        )
    except BaseException:
        if local_tts is not None:
            await local_tts.close()
        await client.close()
        raise
    cfg = CodexRealtimeConversationEngineConfig()
    engine = CodexConversationEngine(
        client=client,
        answer_sdp=answer_sdp,
        stream_id=stream_id,
        tts=local_tts,
    )
    return cfg, engine


async def build_engine(
    options: EngineOptions,
    *,
    stream_id: str,
    codex_sdp: str | None = None,
    memory_prompt: str = "",
) -> BuiltEngine:
    match options:
        case OpenAIEngineOptions():
            return await build_openai(
                stream_id=stream_id,
                options=options,
                memory_prompt=memory_prompt,
            )
        case CartesiaEngineOptions():
            return await build_cartesia(stream_id=stream_id, options=options)
        case CustomAPIEngineOptions():
            raise ValueError("custom_api connections must be prepared before starting a session")
        case CodexEngineOptions():
            if codex_sdp is None or not codex_sdp.strip():
                raise ValueError("codex_sdp is required for the Codex WebRTC engine")
            return await build_codex(
                stream_id=stream_id,
                options=options,
                sdp=codex_sdp,
                memory_prompt=memory_prompt,
            )
