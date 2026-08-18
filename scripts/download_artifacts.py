# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Download all HuggingFace artifacts to local storage.

Usage:
    pixi run python scripts/download_artifacts.py
    pixi run python scripts/download_artifacts.py --workers 8
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads (default: 4)")
    args = parser.parse_args(argv)

    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_storage_root

    print(f"Storage: {get_storage_root()}")
    get_artifact_manager().ensure_all_artifacts(workers=args.workers)
    print("All artifacts ready.")


if __name__ == "__main__":
    main()
