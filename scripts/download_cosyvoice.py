"""Download the official CosyVoice3 0.5B model for the local TTS worker."""

from __future__ import annotations

import os
import time
from pathlib import Path

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
_REQUIRED_FILES = (
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
)


def _missing_files(model_dir: Path) -> list[str]:
    return [name for name in _REQUIRED_FILES if not (model_dir / name).is_file()]


def _download_from_modelscope(model_dir: Path, _workers: int) -> None:
    from modelscope import snapshot_download as modelscope_snapshot_download

    modelscope_snapshot_download(MODEL_ID, local_dir=str(model_dir))


def _download_from_huggingface(model_dir: Path, workers: int) -> None:
    from huggingface_hub import snapshot_download as huggingface_snapshot_download

    huggingface_snapshot_download(MODEL_ID, local_dir=model_dir, max_workers=workers)


def _download_with_retries(
    source: str,
    model_dir: Path,
    *,
    workers: int,
    attempts: int,
) -> None:
    download = {
        "modelscope": _download_from_modelscope,
        "huggingface": _download_from_huggingface,
    }[source]
    for attempt in range(1, attempts + 1):
        try:
            download(model_dir, workers)
            missing = _missing_files(model_dir)
            if missing:
                raise RuntimeError("missing: " + ", ".join(missing))
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(30, 2 ** (attempt - 1))
            print(
                f"{source} attempt {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {delay}s ..."
            )
            time.sleep(delay)


def main() -> None:
    default_dir = Path(__file__).resolve().parents[1] / "models" / "Fun-CosyVoice3-0.5B-2512"
    model_dir = Path(os.environ.get("AVTR_COSYVOICE_MODEL_DIR", default_dir)).resolve()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {model_dir} ...")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    workers = max(1, int(os.environ.get("AVTR_COSYVOICE_DOWNLOAD_WORKERS", "1")))
    attempts = max(1, int(os.environ.get("AVTR_COSYVOICE_DOWNLOAD_ATTEMPTS", "6")))
    configured_sources = os.environ.get(
        "AVTR_COSYVOICE_DOWNLOAD_SOURCES",
        "modelscope,huggingface",
    )
    sources = [source.strip().lower() for source in configured_sources.split(",")]
    sources = [source for source in sources if source]
    unknown_sources = sorted(set(sources) - {"modelscope", "huggingface"})
    if unknown_sources:
        raise ValueError(
            "Unknown CosyVoice download source(s): " + ", ".join(unknown_sources)
        )
    errors: list[str] = []
    for source in sources:
        try:
            print(f"Trying {source} ...")
            _download_with_retries(
                source,
                model_dir,
                workers=workers,
                attempts=attempts,
            )
            break
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            print(f"{source} failed after {attempts} attempts: {exc}")
    missing = _missing_files(model_dir)
    if missing:
        raise RuntimeError(
            "CosyVoice3 download is incomplete; missing: "
            + ", ".join(missing)
            + "; source errors: "
            + " | ".join(errors)
        )
    print(f"CosyVoice3 ready: {model_dir}")


if __name__ == "__main__":
    main()
