from __future__ import annotations

import builtins
import gc
import importlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

_MODULE_NAME = "avaturn_live_streamer.integrations.cosyvoice_server"


def _load_server_module() -> ModuleType:
    try:
        return importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"missing production module {_MODULE_NAME}; implement it to satisfy this contract"
        ) from exc


def test_incremental_websocket_runtime_remains_python_310_compatible() -> None:
    module = _load_server_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "asyncio.TaskGroup" not in source


def test_writable_speaker_cache_migrates_legacy_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    model_dir = tmp_path / "immutable-model"
    model_dir.mkdir()
    legacy_cache = model_dir / "spk2info.pt"
    legacy_cache.write_bytes(b"legacy-speaker-cache")
    writable_cache = tmp_path / "writable" / "cosyvoice" / "spk2info.pt"
    monkeypatch.setenv("AVTR1_COSYVOICE_SPEAKER_CACHE", str(writable_cache))

    first = module.CosyVoice3Service(model_dir)

    assert first.speaker_cache_path == writable_cache
    assert writable_cache.read_bytes() == b"legacy-speaker-cache"

    # The sidecar migration state prevents a deleted/empty user store from
    # being repopulated from the immutable model on a later process start.
    writable_cache.unlink()
    second = module.CosyVoice3Service(model_dir)
    assert second.speaker_cache_path == writable_cache
    assert not writable_cache.exists()


def test_writable_speaker_cache_replaces_model_embedded_legacy_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    model_dir = tmp_path / "immutable-model"
    model_dir.mkdir()
    writable_cache = tmp_path / "writable" / "spk2info.pt"
    writable_cache.parent.mkdir()
    writable_cache.write_bytes(b"writable-speaker-cache")
    monkeypatch.setenv("AVTR1_COSYVOICE_SPEAKER_CACHE", str(writable_cache))
    expected = {"writable_voice": {"sentinel": object()}}
    load_calls: list[Path] = []

    def fake_load(path, *, map_location, weights_only):
        assert map_location == "cuda:0"
        assert weights_only is True
        load_calls.append(Path(path))
        return expected

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    service = module.CosyVoice3Service(model_dir)
    model = SimpleNamespace(
        frontend=SimpleNamespace(
            spk2info={"legacy_voice": {"old": True}}, device="cuda:0"
        )
    )

    service._apply_speaker_cache_overlay(model)

    assert model.frontend.spk2info is expected
    assert load_calls == [writable_cache]


def test_writable_speaker_cache_is_the_atomic_persistence_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    model_dir = tmp_path / "immutable-model"
    model_dir.mkdir()
    writable_cache = tmp_path / "writable" / "cosyvoice" / "spk2info.pt"
    monkeypatch.setenv("AVTR1_COSYVOICE_SPEAKER_CACHE", str(writable_cache))
    saved_paths: list[Path] = []

    def fake_save(_value, path) -> None:
        target = Path(path)
        saved_paths.append(target)
        target.write_bytes(b"saved")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(save=fake_save))
    service = module.CosyVoice3Service(model_dir)
    model = SimpleNamespace(frontend=SimpleNamespace(spk2info={"voice": {}}))

    service._persist_speaker_cache(model)

    assert writable_cache.read_bytes() == b"saved"
    assert not (model_dir / "spk2info.pt").exists()
    assert len(saved_paths) == 1
    assert saved_paths[0].parent == writable_cache.parent


def test_voice_clone_never_asks_vendor_model_to_write_into_immutable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    model_dir = tmp_path / "immutable-model"
    model_dir.mkdir()
    writable_cache = tmp_path / "writable" / "cosyvoice" / "spk2info.pt"
    monkeypatch.setenv("AVTR1_COSYVOICE_SPEAKER_CACHE", str(writable_cache))
    service = module.CosyVoice3Service(model_dir)
    persisted: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.frontend = SimpleNamespace(spk2info={})

        def add_zero_shot_spk(self, _transcript, _prompt_wav, voice_id) -> None:
            self.frontend.spk2info[voice_id] = {
                "llm_prompt_speech_token_len": np.array([100], dtype=np.int32),
                "prompt_speech_feat_len": np.array([200], dtype=np.int32),
                "prompt_speech_feat": np.linspace(
                    -6.0, 2.0, num=200 * 80, dtype=np.float32
                ).reshape(1, 200, 80),
            }

        def save_spkinfo(self) -> None:
            raise AssertionError("vendor save would write into the immutable model")

    service._model = FakeModel()
    monkeypatch.setattr(
        module,
        "_inspect_reference_audio",
        lambda _path: SimpleNamespace(
            duration_seconds=4.0,
            sample_rate=24_000,
            rms=0.12,
            peak=0.8,
        ),
    )
    monkeypatch.setattr(
        service,
        "_persist_speaker_cache",
        lambda model: persisted.append(dict(model.frontend.spk2info)),
    )

    result = service.add_voice(
        name="portable voice",
        transcript="Authorised reference transcript.",
        reference_audio=b"RIFF-test-only",
        reference_filename="reference.wav",
    )

    assert result["id"] == "portable_voice"
    assert list(persisted[0]) == ["portable_voice"]


class _ServiceMustNotRun:
    def __getattr__(self, name: str):
        raise AssertionError(f"voice service must not be called during validation: {name}")


class _ReleaseService:
    def __init__(self) -> None:
        self.calls = 0

    def release(self) -> dict[str, object]:
        self.calls += 1
        return {
            "service": "cosyvoice",
            "status": "released",
            "released": True,
            "loaded": False,
            "active_requests": 0,
        }


class _SpeechTensor:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([self.value], dtype=np.float32)


def _voice_upload(client: TestClient, *, consent: str | None):
    data = {
        "name": "测试音色",
        "transcript": "这是一段由本人授权使用的参考音频。",
    }
    if consent is not None:
        data["consent"] = consent
    return client.post(
        "/v1/audio/voices",
        data=data,
        files={
            "reference_audio": (
                "reference.wav",
                b"RIFF-test-only",
                "audio/wav",
            )
        },
    )


def test_module_import_does_not_require_torch_or_cosyvoice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(sys.modules):
        if name == _MODULE_NAME:
            sys.modules.pop(name)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith(("torch.", "cosyvoice")):
            raise AssertionError(f"optional runtime dependency imported eagerly: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = _load_server_module()

    assert module.__name__ == _MODULE_NAME


def test_pcm16_chunks_are_reassembled_across_odd_transport_boundaries() -> None:
    module = _load_server_module()
    source = [
        bytes((0x00, 0x01, 0x02)),
        bytes((0x03, 0x04, 0x05, 0x06)),
        bytes((0x07,)),
    ]

    chunks = list(module.iter_pcm16_chunks(source))

    assert b"".join(chunks) == b"".join(source)
    assert chunks
    assert all(chunk and len(chunk) % 2 == 0 for chunk in chunks)


def test_pcm16_chunks_reject_a_terminal_half_sample() -> None:
    module = _load_server_module()

    with pytest.raises(ValueError, match=r"(?i)pcm16|sample|odd"):
        list(module.iter_pcm16_chunks([b"\x00\x01", b"\x02"]))


def test_successful_model_retry_clears_a_stale_health_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    loaded_model = object()
    attempts = 0
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal attempts
        if name == "cosyvoice.cli.cosyvoice":
            fake_module = ModuleType(name)

            def auto_model(**_kwargs):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient load failure")
                return loaded_model

            fake_module.AutoModel = auto_model
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="transient"):
        service._load()
    assert service.health()["error"] == "transient load failure"

    assert service._load() is loaded_model
    assert service.health()["status"] == "ready"
    assert service.health()["error"] is None


def test_concurrent_model_loads_share_one_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    loaded_model = object()
    calls = 0
    calls_lock = threading.Lock()
    start_barrier = threading.Barrier(3)
    release_load = threading.Event()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cosyvoice.cli.cosyvoice":
            fake_module = ModuleType(name)

            def auto_model(**_kwargs):
                nonlocal calls
                with calls_lock:
                    calls += 1
                assert release_load.wait(timeout=2)
                return loaded_model

            fake_module.AutoModel = auto_model
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def load_from_worker():
        start_barrier.wait(timeout=2)
        return service._load()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(load_from_worker) for _ in range(2)]
        start_barrier.wait(timeout=2)
        time.sleep(0.1)
        release_load.set()
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert results == [loaded_model, loaded_model]


def test_release_drops_the_model_cleans_cuda_and_allows_lazy_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    service._model = object()
    cleanup_calls: list[str] = []
    monkeypatch.setattr(gc, "collect", lambda: cleanup_calls.append("gc") or 0)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: cleanup_calls.append("empty_cache"),
                ipc_collect=lambda: cleanup_calls.append("ipc_collect"),
            )
        ),
    )

    result = service.release()

    assert result == {
        "service": "cosyvoice",
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }
    assert cleanup_calls == ["gc", "empty_cache", "ipc_collect"]
    assert service.health()["status"] == "released"
    assert service.health()["loaded"] is False

    reloaded_model = object()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cosyvoice.cli.cosyvoice":
            fake_module = ModuleType(name)
            fake_module.AutoModel = lambda **_kwargs: reloaded_model
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert service._load() is reloaded_model
    assert service.health()["status"] == "ready"
    assert service.health()["released"] is False


def test_release_refuses_to_drop_a_model_while_speech_is_streaming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class SpeechTensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([0.25], dtype=np.float32)

    class StreamingModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            yield {"tts_speech": SpeechTensor()}

    model = StreamingModel()
    service._model = model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")
    chunks = service.stream_speech(request)

    assert next(chunks)
    assert service.health()["active_requests"] == 1
    with pytest.raises(module.ModelBusyError, match="active"):
        service.release()
    assert service._model is model

    chunks.close()
    assert service.health()["active_requests"] == 0


def test_closing_an_unstarted_speech_stream_releases_its_model_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class StreamingModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            raise AssertionError("an unstarted stream must not invoke inference")

    service._model = StreamingModel()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")

    chunks = service.stream_speech(request)
    assert service.health()["active_requests"] == 1
    chunks.close()

    assert service.health()["active_requests"] == 0


def test_each_cosyvoice_stream_restores_the_original_token_hop_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    starting_hops: list[int] = []

    class FakeRuntime:
        token_hop_len = 25

    class FakeModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")
        model = FakeRuntime()

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            starting_hops.append(self.model.token_hop_len)
            self.model.token_hop_len = 100
            yield {"tts_speech": _SpeechTensor()}

    fake_model = FakeModel()
    service._model = fake_model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")

    assert list(service.stream_speech(request))
    assert fake_model.model.token_hop_len == 25
    assert list(service.stream_speech(request))

    assert starting_hops == [25, 25]
    assert fake_model.model.token_hop_len == 25
    assert service.health()["active_requests"] == 0


def test_cosyvoice_stream_closes_vendor_generator_and_restores_hop_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class FakeRuntime:
        token_hop_len = 25

    class VendorStream:
        def __init__(self, runtime: FakeRuntime) -> None:
            self.runtime = runtime
            self.closed = False
            self.emitted = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.emitted:
                raise AssertionError("test stream must be cancelled after its first chunk")
            self.emitted = True
            self.runtime.token_hop_len = 100
            return {"tts_speech": _SpeechTensor()}

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")
        model = FakeRuntime()

        def __init__(self) -> None:
            self.vendor_stream = VendorStream(self.model)

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            return self.vendor_stream

    fake_model = FakeModel()
    service._model = fake_model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")
    managed = service.stream_speech(request)
    aligned = module.iter_pcm16_chunks(managed)

    assert next(aligned)
    aligned.close()

    assert fake_model.vendor_stream.closed is True
    assert fake_model.model.token_hop_len == 25
    assert service.health()["active_requests"] == 0


def test_cosyvoice_serialises_model_inference_across_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    start_barrier = threading.Barrier(3)

    class FakeRuntime:
        token_hop_len = 25

    class FakeModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")
        model = FakeRuntime()

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.1)
                yield {"tts_speech": _SpeechTensor()}
            finally:
                with state_lock:
                    active -= 1

    service._model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")

    def run_stream() -> list[bytes]:
        start_barrier.wait(timeout=2)
        return list(service.stream_speech(request))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_stream) for _ in range(2)]
        start_barrier.wait(timeout=2)
        results = [future.result(timeout=3) for future in futures]

    assert all(results)
    assert max_active == 1
    assert service.health()["active_requests"] == 0


def test_cosyvoice_restores_token_hop_after_vendor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class FakeRuntime:
        token_hop_len = 25

    class FakeModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")
        model = FakeRuntime()

        def list_available_spks(self):
            return ["demo"]

        def inference_zero_shot(self, *_args, **_kwargs):
            self.model.token_hop_len = 100
            raise RuntimeError("vendor stream failed")
            yield  # pragma: no cover

    fake_model = FakeModel()
    service._model = fake_model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="demo")

    with pytest.raises(RuntimeError, match="vendor stream failed"):
        list(service.stream_speech(request))

    assert fake_model.model.token_hop_len == 25
    assert service.health()["active_requests"] == 0


def test_incremental_cosyvoice_stream_uses_one_generator_and_selected_clone(
    tmp_path: Path,
) -> None:
    module = _load_server_module()
    received: dict[str, object] = {}

    class RuntimeModel:
        token_hop_len = 25

    class FakeModel:
        def __init__(self) -> None:
            self.model = RuntimeModel()
            self.frontend = SimpleNamespace(
                text_frontend="wetext",
                spk2info={
                    "voice_local_clone": {
                        "prompt_speech_token": np.ones((1, 20), dtype=np.float32),
                        "prompt_speech_feat": np.ones((1, 150, 80), dtype=np.float32),
                    }
                },
            )

        def list_available_spks(self):
            return ["voice_local_clone"]

        def inference_zero_shot(self, text, *_args, **kwargs):
            import inspect

            received["is_generator"] = inspect.isgenerator(text)
            received["voice"] = kwargs["zero_shot_spk_id"]
            received["text"] = list(text)
            yield {"tts_speech": _SpeechTensor(0.25)}

    service = module.CosyVoice3Service(tmp_path)
    service._model = FakeModel()
    request = module.StreamingSpeechRequest(
        model=module.MODEL_ID,
        voice="voice_local_clone",
    )

    chunks = service.stream_speech_incremental(
        request,
        iter(("第一段", "第二段")),
    )

    assert list(chunks)
    assert received == {
        "is_generator": True,
        "voice": "voice_local_clone",
        "text": ["第一段", "第二段"],
    }


def test_incremental_cosyvoice_websocket_preserves_voice_and_streams_binary() -> None:
    module = _load_server_module()

    class FakeStreamingService:
        def health(self):
            return {"service": "cosyvoice", "status": "ready"}

        def list_models(self):
            return []

        def list_voices(self):
            return []

        def stream_speech_incremental(self, request, text_chunks):
            assert request.voice == "voice_local_clone"
            assert request.model == module.MODEL_ID
            assert list(text_chunks) == ["你好", "世界"]
            yield b"\x01\x00\x02\x00"

        def release(self):
            return {"released": True}

    client = TestClient(module.create_app(service=FakeStreamingService()))
    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "model": module.MODEL_ID,
                "voice": "voice_local_clone",
            }
        )
        assert websocket.receive_json() == {
            "type": "started",
            "sample_rate": module.SAMPLE_RATE,
            "sample_format": "s16le",
        }
        websocket.send_json({"type": "append", "text": "你好"})
        websocket.send_json({"type": "append", "text": "世界"})
        websocket.send_json({"type": "finish"})
        assert websocket.receive_bytes() == b"\x01\x00\x02\x00"
        assert websocket.receive_json() == {"type": "completed"}


def test_incremental_cosyvoice_websocket_disconnect_releases_flooding_stream() -> None:
    module = _load_server_module()
    stream_closed = threading.Event()

    class FloodingService:
        def stream_speech_incremental(self, _request, _text_chunks):
            try:
                for _ in range(1_000):
                    yield b"\x01\x00" * 1_024
            finally:
                stream_closed.set()

    client = TestClient(module.create_app(service=FloodingService()))
    with client.websocket_connect("/v1/audio/speech/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "model": module.MODEL_ID,
                "voice": "voice_local_clone",
            }
        )
        assert websocket.receive_json()["type"] == "started"

    assert stream_closed.wait(timeout=2.0)

def test_voice_clone_uses_a_reopenable_temp_file_and_removes_it(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    reference_audio = b"RIFF-test-only"

    class FakeModel:
        prompt_path: Path | None = None
        prompt_bytes: bytes | None = None
        saved = False

        def __init__(self) -> None:
            self.frontend = SimpleNamespace(spk2info={})

        def add_zero_shot_spk(self, transcript, prompt_wav, voice_id) -> None:
            assert transcript == (
                "You are a helpful assistant.<|endofprompt|>"
                "Authorised reference transcript."
            )
            assert voice_id == "demo_voice"
            self.prompt_path = Path(prompt_wav)
            assert self.prompt_path.suffix == ".wav"
            self.prompt_bytes = self.prompt_path.read_bytes()
            self.frontend.spk2info[voice_id] = {
                "prompt_text_len": np.array([12], dtype=np.int32),
                "llm_prompt_speech_token_len": np.array([100], dtype=np.int32),
                "prompt_speech_feat_len": np.array([200], dtype=np.int32),
                "prompt_speech_feat": np.linspace(
                    -6.0, 2.0, num=200 * 80, dtype=np.float32
                ).reshape(1, 200, 80),
            }

        def save_spkinfo(self) -> None:
            self.saved = True

    model = FakeModel()
    service._model = model
    monkeypatch.setattr(
        module,
        "_inspect_reference_audio",
        lambda _path: SimpleNamespace(
            duration_seconds=4.0,
            sample_rate=24_000,
            rms=0.12,
            peak=0.8,
        ),
    )

    result = service.add_voice(
        name="demo voice",
        transcript=" Authorised reference transcript. ",
        reference_audio=reference_audio,
        reference_filename="reference.wav",
    )

    assert result["id"] == "demo_voice"
    assert model.prompt_bytes == reference_audio
    assert model.saved is True
    assert model.prompt_path is not None
    assert not model.prompt_path.exists()


def test_short_silent_reference_is_rejected_before_model_registration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class FakeModel:
        added = False
        saved = False

        def add_zero_shot_spk(self, *_args) -> None:
            self.added = True

        def save_spkinfo(self) -> None:
            self.saved = True

    model = FakeModel()
    service._model = model
    inspector = getattr(module, "_inspect_reference_audio", None)
    assert callable(inspector), "decoded reference-audio validation is not implemented"
    monkeypatch.setattr(
        module,
        "_inspect_reference_audio",
        lambda _path: SimpleNamespace(
            duration_seconds=0.16,
            sample_rate=24_000,
            rms=0.0,
            peak=0.0,
        ),
    )

    with pytest.raises(ValueError, match=r"(?i)short|3 seconds|3 秒|silent|静音"):
        service.add_voice(
            name="Doc",
            transcript="大家好，我是 Doc。",
            reference_audio=b"invalid-short-silence",
            reference_filename="reference.wav",
        )

    assert model.added is False
    assert model.saved is False


def test_reference_inspector_counts_stereo_frames_not_channel_samples(tmp_path) -> None:
    module = _load_server_module()
    soundfile = pytest.importorskip("soundfile")
    sample_rate = 24_000
    seconds = 4
    time_axis = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    mono = 0.15 * np.sin(2 * np.pi * 220 * time_axis)
    stereo = np.column_stack((mono, mono * 0.8))
    path = tmp_path / "stereo.wav"
    soundfile.write(path, stereo, sample_rate)

    info = module._inspect_reference_audio(path)

    assert info.duration_seconds == pytest.approx(4.0)
    assert info.sample_rate == sample_rate
    assert info.rms > 0.01
    assert info.peak > 0.1


def test_unsupported_reference_container_is_rejected_before_model_load(tmp_path) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    with pytest.raises(ValueError, match=r"(?i)WAV|FLAC|OGG|MP3"):
        service.add_voice(
            name="unsupported",
            transcript="Authorised exact transcript.",
            reference_audio=b"not-used",
            reference_filename="recording.webm",
        )

    assert service._model is None


def test_bad_extracted_features_do_not_overwrite_an_existing_voice(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    previous = {"sentinel": object()}

    class FakeModel:
        saved = False

        def __init__(self) -> None:
            self.frontend = SimpleNamespace(spk2info={"demo_voice": previous})

        def add_zero_shot_spk(self, _text, _path, voice_id) -> None:
            self.frontend.spk2info[voice_id] = {
                "prompt_text_len": np.array([14], dtype=np.int32),
                "llm_prompt_speech_token_len": np.array([4], dtype=np.int32),
                "prompt_speech_feat_len": np.array([8], dtype=np.int32),
                "prompt_speech_feat": np.full((1, 8, 80), -11.5129, dtype=np.float32),
            }

        def save_spkinfo(self) -> None:
            self.saved = True

    model = FakeModel()
    service._model = model
    inspector = getattr(module, "_inspect_reference_audio", None)
    assert callable(inspector), "decoded reference-audio validation is not implemented"
    monkeypatch.setattr(
        module,
        "_inspect_reference_audio",
        lambda _path: SimpleNamespace(
            duration_seconds=4.0,
            sample_rate=24_000,
            rms=0.12,
            peak=0.8,
        ),
    )

    with pytest.raises(ValueError, match=r"(?i)feature|token|acoustic|声学|无效"):
        service.add_voice(
            name="demo voice",
            transcript="这是一段准确的参考音频逐字稿。",
            reference_audio=b"valid-container-but-bad-decoder-result",
            reference_filename="reference.wav",
        )

    assert model.frontend.spk2info["demo_voice"] is previous
    assert model.saved is False


def test_voice_list_marks_legacy_short_silent_cache_as_unselectable(tmp_path) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    service._model = SimpleNamespace(
        list_available_spks=lambda: ["Doc", "healthy_voice"],
        frontend=SimpleNamespace(
            spk2info={
                "Doc": {
                    "prompt_text_len": np.array([14], dtype=np.int32),
                    "llm_prompt_speech_token_len": np.array([4], dtype=np.int32),
                    "prompt_speech_feat_len": np.array([8], dtype=np.int32),
                    "prompt_speech_feat": np.full((1, 8, 80), -11.5129, dtype=np.float32),
                },
                "healthy_voice": {
                    "prompt_text_len": np.array([22], dtype=np.int32),
                    "llm_prompt_speech_token_len": np.array([87], dtype=np.int32),
                    "prompt_speech_feat_len": np.array([174], dtype=np.int32),
                    "prompt_speech_feat": np.linspace(
                        -8.0, 2.0, num=174 * 80, dtype=np.float32
                    ).reshape(1, 174, 80),
                },
            }
        ),
    )

    voices = {item["id"]: item for item in service.list_voices()}

    assert voices["Doc"]["quality"] == "invalid"
    assert voices["Doc"]["selectable"] is False
    assert voices["Doc"]["reference_duration_seconds"] == pytest.approx(0.16)
    assert voices["healthy_voice"]["quality"] == "ready"
    assert voices["healthy_voice"]["selectable"] is True


def test_released_voice_list_reads_disk_cache_without_reloading_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    cache_path = tmp_path / "spk2info.pt"
    cache_path.write_bytes(b"persisted-speaker-cache")
    cached = {
        "healthy_voice": {
            "llm_prompt_speech_token_len": np.array([87], dtype=np.int32),
            "prompt_speech_feat_len": np.array([174], dtype=np.int32),
            "prompt_speech_feat": np.linspace(
                -8.0, 2.0, num=174 * 80, dtype=np.float32
            ).reshape(1, 174, 80),
        }
    }
    load_calls: list[tuple[Path, str, bool]] = []

    def fake_torch_load(path, *, map_location, weights_only):
        load_calls.append((Path(path), map_location, weights_only))
        return cached

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_torch_load))
    monkeypatch.setattr(
        service,
        "_load",
        lambda: (_ for _ in ()).throw(
            AssertionError("listing persisted voices must not reload CosyVoice")
        ),
    )
    service._released = True

    voices = service.list_voices()

    assert [voice["id"] for voice in voices] == ["healthy_voice"]
    assert voices[0]["quality"] == "ready"
    assert load_calls == [(cache_path, "cpu", True)]
    assert service.health()["loaded"] is False
    assert service.health()["status"] == "released"


def test_stream_speech_rejects_a_known_invalid_legacy_voice_before_inference(
    tmp_path,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    class FakeModel:
        frontend = SimpleNamespace(
            spk2info={
                "Doc": {
                    "prompt_text_len": np.array([14], dtype=np.int32),
                    "llm_prompt_speech_token_len": np.array([4], dtype=np.int32),
                    "prompt_speech_feat_len": np.array([8], dtype=np.int32),
                    "prompt_speech_feat": np.full((1, 8, 80), -11.5129, dtype=np.float32),
                }
            }
        )

        def list_available_spks(self):
            return ["Doc"]

        def inference_zero_shot(self, *_args, **_kwargs):
            raise AssertionError("known-invalid voice must not reach inference")

    service._model = FakeModel()
    request = module.SpeechRequest(model=module.MODEL_ID, input="hello", voice="Doc")

    with pytest.raises(ValueError, match=r"(?i)invalid|selectable|quality|无效"):
        list(service.stream_speech(request))


def test_stream_speech_allows_an_unknown_legacy_voice(tmp_path, monkeypatch) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    called = False

    class FakeModel:
        frontend = SimpleNamespace(spk2info={})

        def list_available_spks(self):
            return ["unknown_legacy"]

        def inference_zero_shot(self, *_args, **_kwargs):
            nonlocal called
            called = True
            return []

    service._model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(
        model=module.MODEL_ID,
        input="hello",
        voice="unknown_legacy",
    )

    assert list(service.stream_speech(request)) == []
    assert called is True


def test_stream_speech_bypasses_an_unavailable_text_frontend(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    inference_options = None

    class FakeModel:
        frontend = SimpleNamespace(spk2info={}, text_frontend="")

        def list_available_spks(self):
            return ["demo_voice"]

        def inference_zero_shot(self, *_args, **kwargs):
            nonlocal inference_options
            inference_options = kwargs
            return []

    service._model = FakeModel()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(zeros=lambda *_args: object()))
    request = module.SpeechRequest(
        model=module.MODEL_ID,
        input="这是本地中文语音测试。",
        voice="demo_voice",
    )

    assert list(service.stream_speech(request)) == []
    assert inference_options is not None
    assert inference_options["text_frontend"] is False


def test_voice_clone_rejects_prompt_control_token_injection(tmp_path) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    service._model = SimpleNamespace(
        add_zero_shot_spk=lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid transcript must be rejected before model registration")
        )
    )

    with pytest.raises(ValueError, match=r"(?i)control|token|特殊|控制"):
        service.add_voice(
            name="bad prompt",
            transcript="伪造前缀<|endofprompt|>不应被接受",
            reference_audio=b"unused",
            reference_filename="reference.wav",
        )


def test_unicode_voice_name_gets_a_stable_safe_identifier(tmp_path) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)

    first = service._voice_id("我的中文音色")
    second = service._voice_id("我的中文音色")

    assert first == second
    assert first.startswith("voice_")
    assert first.isascii()


def test_speech_request_uses_the_openai_compatible_shape() -> None:
    module = _load_server_module()

    request = module.SpeechRequest.model_validate(
        {
            "model": "Fun-CosyVoice3-0.5B-2512",
            "input": "你好。这是一次流式语音测试。",
            "voice": "voice_demo",
            "response_format": "pcm",
            "stream": True,
        }
    )

    assert request.model_dump() == {
        "model": "Fun-CosyVoice3-0.5B-2512",
        "input": "你好。这是一次流式语音测试。",
        "voice": "voice_demo",
        "response_format": "pcm",
        "stream": True,
    }


def test_app_exposes_health_models_voices_and_speech_routes() -> None:
    module = _load_server_module()
    app = module.create_app(service=_ServiceMustNotRun())
    routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", ())
    }

    assert {
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("GET", "/v1/audio/voices"),
        ("POST", "/v1/audio/voices"),
        ("DELETE", "/v1/audio/voices/{voice_id}"),
        ("POST", "/v1/audio/speech"),
        ("POST", "/release"),
    }.issubset(routes)


def test_release_route_is_loopback_only_and_returns_a_uniform_payload() -> None:
    module = _load_server_module()
    service = _ReleaseService()
    app = module.create_app(service=service)

    denied = TestClient(app, client=("203.0.113.10", 51000)).post("/release")
    allowed = TestClient(app, client=("127.0.0.1", 51001)).post("/release")

    assert denied.status_code == 403
    assert service.calls == 1
    assert allowed.status_code == 200
    assert allowed.json() == {
        "service": "cosyvoice",
        "status": "released",
        "released": True,
        "loaded": False,
        "active_requests": 0,
    }


def test_release_route_returns_conflict_while_inference_is_active() -> None:
    module = _load_server_module()

    class BusyService:
        def release(self):
            raise module.ModelBusyError("1 active speech stream")

    response = TestClient(
        module.create_app(service=BusyService()),
        client=("127.0.0.1", 51002),
    ).post("/release")

    assert response.status_code == 409
    assert "active" in response.json()["detail"]


def test_voice_upload_rejects_a_missing_consent_field_before_service_call() -> None:
    module = _load_server_module()
    client = TestClient(
        module.create_app(service=_ServiceMustNotRun()),
        raise_server_exceptions=False,
    )

    response = _voice_upload(client, consent=None)

    assert response.status_code == 422
    assert "consent" in response.text


def test_voice_upload_rejects_false_consent_before_service_call() -> None:
    module = _load_server_module()
    client = TestClient(
        module.create_app(service=_ServiceMustNotRun()),
        raise_server_exceptions=False,
    )

    response = _voice_upload(client, consent="false")

    assert response.status_code == 422
    assert "consent" in response.text


def test_local_browser_origin_is_allowed_for_voice_clone_requests() -> None:
    module = _load_server_module()
    client = TestClient(module.create_app(service=_ServiceMustNotRun()))

    response = client.options(
        "/v1/audio/voices",
        headers={
            "Origin": "http://localhost:7860",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:7860"


def test_local_voice_delete_persists_and_removes_the_cached_voice(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    cached = {"prompt_speech_feat": object()}
    model = SimpleNamespace(frontend=SimpleNamespace(spk2info={"demo_voice": cached}))
    service._model = model
    persisted = []
    monkeypatch.setattr(
        service,
        "_persist_speaker_cache",
        lambda value: persisted.append(dict(value.frontend.spk2info)),
    )

    result = service.delete_voice("demo_voice")

    assert result == {
        "id": "demo_voice",
        "object": "audio.voice.deleted",
        "deleted": True,
    }
    assert model.frontend.spk2info == {}
    assert persisted == [{}]


def test_local_voice_delete_rolls_back_if_persistence_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_server_module()
    service = module.CosyVoice3Service(tmp_path)
    cached = {"prompt_speech_feat": object()}
    model = SimpleNamespace(frontend=SimpleNamespace(spk2info={"demo_voice": cached}))
    service._model = model

    def fail_persist(_model) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(service, "_persist_speaker_cache", fail_persist)

    with pytest.raises(OSError, match="disk full"):
        service.delete_voice("demo_voice")

    assert model.frontend.spk2info == {"demo_voice": cached}


def test_local_voice_delete_route_returns_404_for_an_unknown_voice() -> None:
    module = _load_server_module()

    class FakeService:
        def delete_voice(self, voice_id: str):
            raise KeyError(voice_id)

    response = TestClient(
        module.create_app(service=FakeService()),
        raise_server_exceptions=False,
    ).delete("/v1/audio/voices/missing_voice")

    assert response.status_code == 404
    assert "missing_voice" in response.json()["detail"]
