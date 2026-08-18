# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community
# Modified for DigiBox: native Windows runtime selection and portable inference support.

"""Render a talking-head video from audio.

Downloads any missing models from HuggingFace on first run.

Usage:
    pixi run generate_offline --speech path/to/speech.mp3 --bg plain_white
    pixi run generate_offline --speech s.wav --listen l.wav --bg plain_white
    pixi run generate_offline --duration 10 --bg plain_white                          # 10 s silence
    pixi run generate_offline --speech s.wav --bg plain_white --no-mux                # video only
    pixi run generate_offline --speech s.wav --bg plain_white --stream-frames         # per-frame (default)
    pixi run generate_offline --speech s.wav --bg plain_white --no-stream-frames      # batched mode

Runtime selection:
    Native Windows defaults to the TorchScript + ONNX Runtime CUDA backend.
    Linux uses local TensorRT engines when available. Pass ``--backend`` to
    select either path explicitly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr

from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, RenderOptions

SAMPLE_RATE = 16_000
FPS = 25


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return audio.astype(np.float32)


def _fit_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    n = audio.shape[0]
    if n == target_samples:
        return audio
    if n > target_samples:
        return audio[:target_samples]
    return np.concatenate([audio, np.zeros(target_samples - n, dtype=audio.dtype)])


def _align_tracks(
    speech: np.ndarray | None,
    listen: np.ndarray | None,
    duration: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speech_f32, listen_f32) of equal length.

    If ``duration`` is set, both tracks are trimmed/padded to it.
    Otherwise length = the longer of the two; missing track defaults to silence.
    """
    if speech is None and listen is None and duration is None:
        raise ValueError("Provide at least one of --speech, --listen, --duration.")
    if duration is not None:
        n = int(round(duration * SAMPLE_RATE))
    else:
        n = max(
            speech.shape[0] if speech is not None else 0,
            listen.shape[0] if listen is not None else 0,
        )
    s = speech if speech is not None else np.zeros(n, dtype=np.float32)
    l = listen if listen is not None else np.zeros(n, dtype=np.float32)
    return _fit_duration(s, n), _fit_duration(l, n)


def _chunk_window(pipeline: Pipeline) -> int:
    mg = pipeline._motion_generator
    return (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift


def _chunk_step(pipeline: Pipeline) -> int:
    return pipeline._motion_generator.chunk_size * pipeline._motion_generator.frame_len


def _slice_chunks(audio: np.ndarray, window: int, step: int) -> list[np.ndarray]:
    n_steps = max(1, (len(audio) + step - 1) // step)
    chunks = []
    for i in range(n_steps):
        start = i * step
        piece = audio[start : start + window]
        if len(piece) < window:
            piece = np.pad(piece, (0, window - len(piece)))
        chunks.append(piece)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, default=None,
                        help="Speech / self audio (any format)")
    parser.add_argument("--listen", type=Path, default=None,
                        help="Listen / other audio (any format)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Force render duration in seconds; audio is trimmed/padded to fit")
    parser.add_argument("--avatar", default="maria", help="Avatar ID to render")
    parser.add_argument("--out", type=Path, default=Path("demo_output.mp4"))
    parser.add_argument("--bg", required=True, help="Background ID (must match a file in the backgrounds artifact, e.g. 'plain_white')")
    parser.add_argument(
        "--backend",
        choices=("auto", "torchscript", "tensorrt"),
        default="auto",
        help="AVTR-1 runtime backend (auto prefers local TensorRT engines).",
    )
    parser.add_argument(
        "--no-mux", dest="mux", action="store_false",
        help="Don't mux the speech audio into the output video.",
    )
    parser.add_argument(
        "--stream-frames", action=argparse.BooleanOptionalAction, default=True,
        help="Yield each frame as soon as it's ready (default: on). "
             "Use --no-stream-frames for batched mode.",
    )
    args = parser.parse_args()

    if args.speech is None and args.listen is None and args.duration is None:
        parser.error("Provide at least one of --speech, --listen, --duration.")

    speech_raw = _load_mono_16k(args.speech) if args.speech else None
    listen_raw = _load_mono_16k(args.listen) if args.listen else None
    speech, listen = _align_tracks(speech_raw, listen_raw, args.duration)

    print(
        f"Audio: {speech.shape[0] / SAMPLE_RATE:.2f}s "
        f"(speech={'set' if args.speech else 'silence'}, "
        f"listen={'set' if args.listen else 'silence'})"
    )

    print(f"Loading pipeline for avatar '{args.avatar}'...")
    print("  (models are downloaded from HuggingFace on first run — this may take a few minutes)")
    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=[args.avatar],
        avtr1_backend=args.backend,
    )
    avatar = registry[args.avatar]

    window = _chunk_window(pipeline)
    step = _chunk_step(pipeline)
    frames_per_chunk = pipeline._motion_generator.chunk_size

    speech_chunks = _slice_chunks(speech, window, step)
    listen_chunks = _slice_chunks(listen, window, step)
    n_chunks = len(speech_chunks)
    print(f"Chunks: {n_chunks}  frames: {n_chunks * frames_per_chunk}  "
          f"({n_chunks * frames_per_chunk / FPS:.1f}s at {FPS} fps)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_h, out_w = avatar.source.shape[-2:]

    audio_path = args.speech if (args.mux and args.speech is not None) else None
    writer = imageio_ffmpeg.write_frames(
        str(args.out),
        size=(out_w, out_h),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(audio_path) if audio_path is not None else None,
        audio_codec="aac" if audio_path is not None else None,
    )
    writer.send(None)

    mode = "streaming" if args.stream_frames else "batched"
    print(f"Render mode: {mode}")
    options = RenderOptions(pixel_format="yuv_i420", bg_id=args.bg, stream_frames=args.stream_frames)
    state = None
    produced = 0
    chunk_times: list[float] = []

    try:
        for i, (sp, ls) in enumerate(zip(speech_chunks, listen_chunks)):
            t0 = time.perf_counter()
            chunk = Chunk(audio_speech=sp, audio_listen=ls)
            state, frames_iter = pipeline.process_chunk(avatar, chunk, state, options)
            for frame in frames_iter:
                writer.send(frame.data.tobytes())
                produced += 1
            chunk_times.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 25 == 0 or i == n_chunks - 1:
                avg_ms = sum(chunk_times) / len(chunk_times)
                chunk_times.clear()
                print(
                    f"  chunk {i + 1}/{n_chunks} "
                    f"({produced} frames, {produced / FPS:.1f}s, avg {avg_ms:.0f} ms/chunk)"
                )
    finally:
        writer.close()

    print(f"\nDone. {produced} frames ({produced / FPS:.1f}s) → {args.out}")


if __name__ == "__main__":
    main()
