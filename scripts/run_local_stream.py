# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community
# Modified for DigiBox: Windows sidecars, desktop lifecycle, and isolated package layers.

"""Top-level orchestrator: renderer subprocess + health-poll + streamer.

Spawns the avtr1_renderer FastAPI app under the renderer env (`pixi run -e default
python -m avtr1_renderer.api.app`), polls `GET /health` until it returns 200,
then spawns the localrtc streamer (`python -m avaturn_live_streamer.local_stream_cli`).
Both children inherit env so AVTR1_LOCAL_STORAGE / CLOUDFLARE_TURN_* propagate
from the parent shell. Conversation-engine credentials (OpenAI / Cartesia API
keys) are entered per-session in the local-stream UI and stored only in the
browser's localStorage, not env. The streamer's renderer wiring is injected here
(mode=single, lb_or_instance_url=http://localhost:{RENDERER_PORT}) so no backend
env file is read.

Run via the streamer env:
    pixi run -e streamer python scripts/run_local_stream.py
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import httpx
from dotenv import load_dotenv

LOG = logging.getLogger("orchestrator")

# Load .env into os.environ before anything else reads env vars. Pydantic-settings
# only populates its own Config object from .env; modules that call os.environ.get()
# directly (e.g. ice.py for CLOUDFLARE_TURN_KEY_*) need the values exported here.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _runtime_root() -> Path:
    """Return the relocatable application root selected by the desktop shell."""

    configured = os.environ.get("AVTR1_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _desktop_stop_requested(path: str | Path | None) -> bool:
    """Whether Electron requested an orderly shutdown through its stop file."""

    return bool(path) and Path(path).is_file()


def _routed_process_env(
    pythonpath_variable: str,
    *,
    legacy_python_paths: tuple[str | Path, ...] = (),
) -> dict[str, str]:
    """Build one child environment without mixing portable package layers.

    Portable-v2 supplies a complete, ordered ``PYTHONPATH`` for each process
    class.  When that route is absent, retain the v1 inherited-path behavior;
    CosyVoice additionally keeps its historical source-path prefixes.
    """

    env = os.environ.copy()
    routed_pythonpath = env.get(pythonpath_variable)
    if routed_pythonpath:
        env["PYTHONPATH"] = routed_pythonpath
        return env

    if legacy_python_paths:
        python_paths = [str(path) for path in legacy_python_paths]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _worker_python(
    project_root: str | Path,
    *,
    environment_name: str,
    portable_directory: str,
    legacy_directory: str,
) -> Path | None:
    root = Path(project_root)
    override = os.environ.get(environment_name)
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"{environment_name} does not exist: {candidate}")
        return candidate.resolve()

    portable = root / portable_directory / "python.exe"
    if portable.is_file():
        return portable
    legacy = root / legacy_directory / "Scripts" / "python.exe"
    return legacy if legacy.is_file() else None


def _renderer_command(
    port: int,
    *,
    host: str = "127.0.0.1",
    platform: str = sys.platform,
    python_executable: str = sys.executable,
    pixi_executable: str | None = None,
) -> list[str]:
    if platform == "win32" or os.environ.get("AVTR1_SINGLE_ENV") == "1":
        prefix = [python_executable]
    else:
        pixi = pixi_executable or shutil.which("pixi")
        if pixi is None:
            raise RuntimeError(
                "pixi was not found; set AVTR1_SINGLE_ENV=1 to use the current Python"
            )
        prefix = [pixi, "run", "-e", "renderer", "python"]
    return [
        *prefix,
        "-m",
        "avtr1_renderer.api.launcher",
        "avtr1_renderer.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _feynobg_command(project_root: str | Path, *, port: int = 8767) -> list[str] | None:
    """Return the isolated Windows worker command when its venv is installed."""

    _ = port  # The module reads the port from its narrowly scoped environment.
    worker_python = _worker_python(
        project_root,
        environment_name="AVTR1_FEYNOBG_PYTHON",
        portable_directory="python-feynobg",
        legacy_directory=".venv-feynobg",
    )
    if worker_python is None:
        return None
    return [
        str(worker_python),
        "-m",
        "avaturn_live_streamer.integrations.feynobg_server",
    ]


def _start_feynobg(port: int) -> subprocess.Popen[bytes] | None:
    if os.environ.get("AVTR1_FEYNOBG_ENABLED", "1").casefold() in {"0", "false", "off"}:
        LOG.info("FeyNoBg worker disabled")
        return None
    project_root = _runtime_root()
    cmd = _feynobg_command(project_root, port=port)
    if cmd is None:
        LOG.warning(
            "FeyNoBg environment not installed; upload cutout will use AVTR's fallback. "
            "Run scripts/setup_feynobg_windows.ps1 to enable it."
        )
        return None
    env = _routed_process_env("AVTR1_FEYNOBG_PYTHONPATH")
    env["AVTR1_FEYNOBG_PORT"] = str(port)
    env.setdefault("AVTR1_FEYNOBG_DEVICE", "cpu")
    LOG.info("starting FeyNoBg worker: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _cosyvoice_command(
    project_root: str | Path,
    *,
    port: int = 8768,
) -> list[str] | None:
    """Return the isolated Windows CosyVoice worker when fully installed."""

    root = Path(project_root)
    worker_python = _worker_python(
        root,
        environment_name="AVTR1_COSYVOICE_PYTHON",
        portable_directory="python-cosyvoice",
        legacy_directory=".venv-cosyvoice",
    )
    if worker_python is None:
        return None
    app_root = Path(os.environ.get("AVTR1_APP_ROOT", root))
    models_root = Path(os.environ.get("AVTR1_MODELS_ROOT", root / "models"))
    source_package = root / "third_party" / "CosyVoice" / "cosyvoice"
    if app_root != root:
        source_package = app_root / "third_party" / "CosyVoice" / "cosyvoice"
    model_config = models_root / "Fun-CosyVoice3-0.5B-2512" / "cosyvoice3.yaml"
    if not all(path.exists() for path in (source_package, model_config)):
        return None
    return [
        str(worker_python),
        "-m",
        "uvicorn",
        "avaturn_live_streamer.integrations.cosyvoice_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _start_cosyvoice(port: int) -> subprocess.Popen[bytes] | None:
    if os.environ.get("AVTR1_COSYVOICE_ENABLED", "1").casefold() in {
        "0",
        "false",
        "off",
    }:
        LOG.info("CosyVoice worker disabled")
        return None
    project_root = _runtime_root()
    cmd = _cosyvoice_command(project_root, port=port)
    if cmd is None:
        LOG.warning(
            "CosyVoice environment or model is not installed; local cloned TTS "
            "will stay unavailable. Run scripts/setup_cosyvoice_windows.ps1 to enable it."
        )
        return None
    app_root = Path(os.environ.get("AVTR1_APP_ROOT", project_root))
    models_root = Path(os.environ.get("AVTR1_MODELS_ROOT", project_root / "models"))
    source_root = app_root / "third_party" / "CosyVoice"
    matcha_root = source_root / "third_party" / "Matcha-TTS"
    env = _routed_process_env(
        "AVTR1_COSYVOICE_PYTHONPATH",
        legacy_python_paths=(source_root, matcha_root, app_root / "src"),
    )
    env["AVTR_COSYVOICE_MODEL_DIR"] = str(
        models_root / "Fun-CosyVoice3-0.5B-2512"
    )
    LOG.info("starting CosyVoice worker: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _start_renderer(port: int) -> subprocess.Popen[bytes]:
    env = _routed_process_env("AVTR1_MAIN_PYTHONPATH")
    # The renderer reads its own settings (AVTR1_LOCAL_STORAGE etc.) from env;
    # don't fight with it. The renderer launcher preinitializes NVIDIA VSR
    # before importing Uvicorn, then forwards these CLI arguments unchanged.
    # Disable the LB keep-alive worker -- there is no LB in the local setup.
    env.setdefault("LOAD_BALANCER_URL", "disabled")
    cmd = _renderer_command(
        port,
        host=os.environ.get("RENDERER_HOST", "127.0.0.1"),
    )
    LOG.info("starting renderer: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


class _Interrupted(Exception):  # noqa: N818 - internal control-flow sentinel
    pass


class _RendererExited(RuntimeError):  # noqa: N818 - existing internal API
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        super().__init__(f"renderer exited with code {returncode} before becoming healthy")


def _wait_for_health(
    port: int,
    timeout_s: float = 300.0,
    is_interrupted: Callable[[], bool] = lambda: False,
    renderer_poll: Callable[[], int | None] | None = None,
) -> None:
    """Poll http://localhost:{port}/health once per second until 200 or timeout.

    Raises ``_Interrupted`` if ``is_interrupted()`` becomes True between polls
    so SIGINT/SIGTERM during the (potentially long) renderer warmup aborts
    promptly instead of waiting out the deadline.
    """
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    with httpx.Client(timeout=3.0) as client:
        while time.monotonic() < deadline:
            if is_interrupted():
                raise _Interrupted()
            if renderer_poll is not None:
                returncode = renderer_poll()
                if returncode is not None:
                    raise _RendererExited(returncode)
            try:
                r = client.get(f"http://localhost:{port}/health")
                if r.status_code == 200:
                    LOG.info("renderer healthy")
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
            # Sleep in short slices so a signal between polls is noticed quickly.
            slept = 0.0
            while slept < 1.0 and not is_interrupted():
                if renderer_poll is not None:
                    returncode = renderer_poll()
                    if returncode is not None:
                        raise _RendererExited(returncode)
                time.sleep(0.1)
                slept += 0.1
    raise TimeoutError(
        f"renderer /health did not become 200 within {timeout_s:.0f}s; "
        f"last error: {last_err}"
    )


def _start_streamer(host: str, port: int, renderer_port: int) -> subprocess.Popen[bytes]:
    env = _routed_process_env("AVTR1_MAIN_PYTHONPATH")
    # Note the double `RENDERERS__` segment: `Config.renderers: RenderersConfig`
    # is itself a settings model with a `renderers` field, so the env path is
    # Config.renderers.renderers["avtrn-1"]. Matches upstream local.env convention.
    env["RENDERERS__RENDERERS__AVTRN_1__MODE"] = "single"
    env["RENDERERS__RENDERERS__AVTRN_1__LB_OR_INSTANCE_URL"] = f"http://localhost:{renderer_port}"
    cmd = [
        sys.executable, "-m", "avaturn_live_streamer.local_stream_cli",
        "--host", host,
        "--port", str(port),
    ]
    LOG.info("starting streamer: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _terminate(proc: subprocess.Popen[bytes], name: str, grace_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    LOG.info("terminating %s (pid=%d)", name, proc.pid)
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        LOG.warning("%s did not exit gracefully; killing", name)
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s orchestrator: %(message)s",
    )

    host = os.environ.get("STREAMER_HOST", "127.0.0.1")
    port = _env_int("STREAMER_PORT", 7860)
    renderer_port = _env_int("RENDERER_PORT", 8000)
    feynobg_port = _env_int("AVTR1_FEYNOBG_PORT", 8767)
    cosyvoice_port = _env_int("AVTR1_COSYVOICE_PORT", 8768)
    desktop_stop_file = os.environ.get("AVTR1_DESKTOP_STOP_FILE")

    feynobg: subprocess.Popen[bytes] | None = None
    cosyvoice: subprocess.Popen[bytes] | None = None
    renderer = _start_renderer(renderer_port)
    streamer: subprocess.Popen[bytes] | None = None

    interrupted = False

    def _sig_handler(signum: int, _frame: object) -> None:
        nonlocal interrupted
        LOG.info("orchestrator received signal %d; shutting down", signum)
        interrupted = True

    def _shutdown_requested() -> bool:
        return interrupted or _desktop_stop_requested(desktop_stop_file)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        # Wait for renderer health.
        try:
            _wait_for_health(
                renderer_port,
                is_interrupted=_shutdown_requested,
                renderer_poll=renderer.poll,
            )
        except _Interrupted:
            LOG.info("interrupted during renderer warmup; stopping renderer")
            return 130
        except _RendererExited as e:
            LOG.error(str(e))
            return e.returncode or 1
        except TimeoutError as e:
            LOG.error(str(e))
            return 1
        if renderer.poll() is not None:
            LOG.error("renderer exited with code %d before becoming healthy", renderer.returncode)
            return renderer.returncode or 1

        streamer = _start_streamer(host, port, renderer_port)
        LOG.info("streamer started pid=%d", streamer.pid)
        # Initialise the latency-critical renderer/VSR path without competing
        # worker imports. The optional workers remain lazy and can start after
        # the renderer is healthy and the streamer process has been spawned.
        feynobg = _start_feynobg(feynobg_port)
        cosyvoice = _start_cosyvoice(cosyvoice_port)

        # Wait for either child to exit (or signal).
        while not _shutdown_requested():
            time.sleep(0.5)
            if renderer.poll() is not None:
                LOG.error("renderer exited (code=%d) — stopping streamer", renderer.returncode)
                break
            if streamer.poll() is not None:
                LOG.error("streamer exited (code=%d) — stopping renderer", streamer.returncode)
                break

        rc = streamer.returncode if streamer and streamer.returncode is not None else 0
        if renderer.returncode is not None and renderer.returncode != 0:
            rc = renderer.returncode
        return rc
    finally:
        if streamer is not None:
            _terminate(streamer, "streamer")
        _terminate(renderer, "renderer")
        if feynobg is not None:
            _terminate(feynobg, "FeyNoBg worker")
        if cosyvoice is not None:
            _terminate(cosyvoice, "CosyVoice worker")


if __name__ == "__main__":
    sys.exit(main())
