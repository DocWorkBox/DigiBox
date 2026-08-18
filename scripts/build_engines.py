# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Build all TRT engines: renderer, AVTR1, and HuBERT.

Thin wrapper that runs the three engine build scripts in sequence.

Usage:
    pixi run python scripts/build_engines.py
    pixi run python scripts/build_engines.py --no-fp16
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fp16", action="store_true", help="Build fp32 engines")
    args = parser.parse_args(argv)

    extra = ["--no-fp16"] if args.no_fp16 else []

    import build_renderer_engines
    import build_avtr1_engines
    import build_hubert_engine

    print("=== Renderer engines (decoder, warp, modnet, stitch) ===")
    build_renderer_engines.main(extra)

    print("\n=== AVTR1 engines (encode, decode, normalizer) ===")
    build_avtr1_engines.main(extra)

    print("\n=== HuBERT engine ===")
    build_hubert_engine.main(extra)

    print("\nAll engines built.")


if __name__ == "__main__":
    main()
