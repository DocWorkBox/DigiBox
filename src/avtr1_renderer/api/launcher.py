# SPDX-FileCopyrightText: 2026 DigiBox contributors
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Start the renderer after NVIDIA VSR native-handle preinitialization."""

from avtr1_renderer.nvidia_vsr import preinitialize_nvidia_vsr

# Native VSR must be initialized before importing Uvicorn or the renderer app.
# Model loading remains lazy until a session enables RTX VSR.
preinitialize_nvidia_vsr()


def main() -> None:
    """Forward the remaining command line to Uvicorn after preinitialization."""

    from uvicorn.main import main as uvicorn_main

    uvicorn_main()


if __name__ == "__main__":
    main()
