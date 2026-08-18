# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""End-to-end API test: send audio to the server, write frames to video.

Can connect to an already-running server or start one automatically.

Usage — against a running server:
    pixi run python scripts/test_api.py \\
        --url http://localhost:8000 \\
        --audio example/basic.mp3 \\
        --avatar anya_03_studio \\
        --out api_test_output.mp4

Usage — start server automatically:
    pixi run python scripts/test_api.py \\
        --audio example/basic.mp3 \\
        --avatar anya_03_studio \\
        --out api_test_output.mp4
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16_000
FPS = 25
# Default pipeline params (must match server's motion generator config).
CHUNK_SIZE = 5
FUTURE_SIZE = 5
FRAME_LEN = 640
AUDIO_SHIFT = 80
CURRENT_SAMPLES = CHUNK_SIZE * FRAME_LEN          # 3200
FUTURE_SAMPLES = FUTURE_SIZE * FRAME_LEN + AUDIO_SHIFT  # 3280


def _load_mono_16k_pcm(path: Path) -> bytes:
    """Load audio as int16 PCM bytes at 16 kHz mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return (audio.clip(-1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _slice_pcm(pcm: bytes, cur_n: int, fut_n: int) -> list[tuple[bytes, bytes]]:
    """Slice PCM bytes into (current, future) pairs advancing by cur_n samples."""
    sample_size = 2  # int16
    step = cur_n * sample_size
    window = (cur_n + fut_n) * sample_size
    total = len(pcm)
    n_steps = max(1, (total + step - 1) // step)
    pairs = []
    for i in range(n_steps):
        start = i * step
        chunk = pcm[start : start + window]
        if len(chunk) < window:
            chunk = chunk + bytes(window - len(chunk))
        pairs.append((chunk[:cur_n * sample_size], chunk[cur_n * sample_size:]))
    return pairs


def _wait_healthy(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(1.0)
    raise TimeoutError(f"Server at {url} not healthy after {timeout}s: {last_exc}")


def _post_chunk(
    client: httpx.Client,
    url: str,
    cur_speech: bytes,
    fut_speech: bytes,
    cur_listen: bytes,
    fut_listen: bytes,
    state_blob: bytes | None,
    avatar_id: str,
    bg_id: str,
) -> tuple[bytes, bytes, dict]:
    """POST one chunk; return (next_state_blob, raw_frames_bytes, headers)."""
    files: dict = {
        "current_chunk": ("cur.raw", cur_speech, "application/octet-stream"),
        "future_chunk": ("fut.raw", fut_speech, "application/octet-stream"),
        "current_chunk_listen": ("curl.raw", cur_listen, "application/octet-stream"),
        "future_chunk_listen": ("futl.raw", fut_listen, "application/octet-stream"),
    }
    if state_blob is not None:
        files["state"] = ("state.bin", state_blob, "application/octet-stream")

    params = {"avatar_id": avatar_id, "bg_id": bg_id, "pixel_format": "yuv_i420"}
    r = client.post(f"{url}/process-audio-v3", files=files, params=params, timeout=60.0)
    r.raise_for_status()

    state_len = int(r.headers["X-State-Length-Bytes"])
    body = r.content
    return body[:state_len], body[state_len:], dict(r.headers)


def _mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None,
                        help="Server URL (default: start one automatically)")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--listen", type=Path, default=None,
                        help="Listener audio (default: silence)")
    parser.add_argument("--avatar", default="anya_03_studio")
    parser.add_argument("--bg", default="default")
    parser.add_argument("--out", type=Path, default=Path("api_test_output.mp4"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server_proc: subprocess.Popen | None = None
    server_url = args.url

    if server_url is None:
        server_url = f"http://localhost:{args.port}"
        print(f"Starting server on {server_url}...")
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "avtr1_renderer.api.app",
            ],
            env={**__import__("os").environ},
        )

    try:
        print("Waiting for server to be healthy...")
        _wait_healthy(server_url)
        print("Server is ready.")

        print(f"Loading audio from {args.audio}...")
        speech_pcm = _load_mono_16k_pcm(args.audio)
        listen_pcm = (
            _load_mono_16k_pcm(args.listen)
            if args.listen
            else bytes(len(speech_pcm))
        )
        # Align lengths.
        if len(listen_pcm) < len(speech_pcm):
            listen_pcm += bytes(len(speech_pcm) - len(listen_pcm))
        else:
            listen_pcm = listen_pcm[: len(speech_pcm)]

        speech_chunks = _slice_pcm(speech_pcm, CURRENT_SAMPLES, FUTURE_SAMPLES)
        listen_chunks = _slice_pcm(listen_pcm, CURRENT_SAMPLES, FUTURE_SAMPLES)
        n_chunks = len(speech_chunks)
        print(f"  {len(speech_pcm) // 2 / SAMPLE_RATE:.2f}s audio → {n_chunks} chunks")

        state_blob: bytes | None = None
        frame_h: int | None = None
        frame_w: int | None = None

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        writer = None
        t_start = time.perf_counter()

        with httpx.Client() as client:
            for i, ((cur_sp, fut_sp), (cur_ls, fut_ls)) in enumerate(
                zip(speech_chunks, listen_chunks)
            ):
                state_blob, frames_bytes, resp_headers = _post_chunk(
                    client, server_url,
                    cur_sp, fut_sp, cur_ls, fut_ls,
                    state_blob, args.avatar, args.bg,
                )

                if writer is None:
                    frame_h = int(resp_headers["x-frame-height"])
                    frame_w = int(resp_headers["x-frame-width"])
                    writer = imageio_ffmpeg.write_frames(
                        str(tmp_path),
                        size=(frame_w, frame_h),
                        fps=FPS,
                        codec="libx264",
                        pix_fmt_in="yuv420p",
                        pix_fmt_out="yuv420p",
                        quality=8,
                        macro_block_size=1,
                    )
                    writer.send(None)

                frame_bytes_per = (frame_h * frame_w * 3) // 2
                n_frames = len(frames_bytes) // frame_bytes_per
                for j in range(n_frames):
                    writer.send(frames_bytes[j * frame_bytes_per : (j + 1) * frame_bytes_per])

                if (i + 1) % 10 == 0 or i == n_chunks - 1:
                    elapsed = time.perf_counter() - t_start
                    produced = (i + 1) * CHUNK_SIZE
                    print(f"  chunk {i + 1}/{n_chunks}  {produced} frames  {elapsed:.1f}s")

        if writer is not None:
            writer.close()

        print(f"Muxing audio into {args.out}...")
        _mux_audio(tmp_path, args.audio, args.out)
        tmp_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - t_start
        total_frames = n_chunks * CHUNK_SIZE
        print(f"\nDone. {total_frames} frames ({total_frames / FPS:.1f}s) in {elapsed:.1f}s")
        print(f"Output: {args.out}")

    finally:
        if server_proc is not None:
            print("Stopping server...")
            server_proc.send_signal(signal.SIGTERM)
            server_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
