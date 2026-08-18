"""Download and validate the pinned FeyNoBg model in the isolated venv."""

from __future__ import annotations

import os

from nobg import AutoModel, AutoProcessor

MODEL_ID = os.environ.get("AVTR1_FEYNOBG_MODEL", "feyninc/FeyNobg")
REVISION = os.environ.get(
    "AVTR1_FEYNOBG_REVISION",
    "c1fd67fbefe3efeb78fe2a003270fb5350a0bb1c",
)


def main() -> None:
    print(f"Downloading {MODEL_ID}@{REVISION} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "FeyNoBg ready: "
        f"processor={type(processor).__name__}, model={type(model).__name__}, "
        f"parameters={parameter_count:,}"
    )


if __name__ == "__main__":
    main()
