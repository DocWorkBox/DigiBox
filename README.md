<div align="center">

<!-- Modified for the DigiBox Windows desktop project. -->

# DigiBox

[English](README.md) | [简体中文](README_zh-CN.md)

[DigiBox Product Architecture](https://docworkbox.github.io/DigiBox/)

</div>

DigiBox is a native Windows realtime digital-human application that brings
multi-provider realtime conversation, local long-term memory, hybrid cloud and
local voice, dual-track speaking/listening motion, and LivePortrait rendering into
one desktop experience.

DigiBox is an independent product and is not affiliated with or endorsed by
Goodsize Inc. It uses separately licensed upstream AVTR-1 components in part of
its motion and rendering stack; their licenses and required notices are
preserved. Where an older per-file SPDX header conflicts with the directory
component map in [LICENSE.md](LICENSE.md), the component license named there and
its controlling license file applies.

---

## 📑 What's included

- [x] Model download and setup tooling (weights are obtained separately)
- [x] Inference code
- [x] Interactive streaming demo

---

## Model download links

The Git repository does not include model weights. Source installations download
only the models needed for the enabled features:

- **Core — AVTR-1 motion, HuBERT, MODNet, avatars, and backgrounds:**
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1).
  This repository is gated; sign in and accept its terms before downloading.
- **Core — LivePortrait, InsightFace, and renderer ONNX graphs:**
  [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
- **Optional — local streaming TTS and voice cloning:**
  [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512).
- **Optional — local person cutout:**
  [feyninc/FeyNobg](https://huggingface.co/feyninc/FeyNobg).

The project download and setup scripts fetch these repositories automatically;
the links above are provided for access approval, inspection, and manual download.
TensorRT engines are machine-specific build outputs and must be built locally.

---

## Table of Contents

- [Model download links](#model-download-links)
1. [Quick Start](#1-quick-start)
2. [Native Windows (no WSL)](#2-native-windows-no-wsl)
3. [Performance](#3-performance)
4. [Troubleshooting](#troubleshooting)

---

## 1. 🚀 Quick Start

### Prerequisites

- Linux
- NVIDIA GPU (Ampere or later recommended)
- CUDA 12.x + TensorRT 10.x
- [pixi](https://prefix.dev/) — `curl -fsSL https://pixi.sh/install.sh | sh`

### Install

```bash
git clone --recurse-submodules https://github.com/DocWorkBox/DigiBox.git
cd DigiBox
pixi install
```

### Set storage path (Optional)

```bash
export AVTR1_LOCAL_STORAGE=/path/to/avtr1_storage
```

All downloaded weights and built engines go here. Defaults to `<project_root>/artifacts/` (the repo checkout, not the caller's working directory) when unset.

### Download weights

```bash
pixi run download
```

First run will prompt for a HuggingFace login via `hf auth login`
(automatically invoked as a dependency of `download`).

### Build TRT engines

The core download step uses the first two Hugging Face repositories listed
above: gated AVTR-1 weights from
[avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1) and
LivePortrait weights repackaged as ONNX graphs from
[digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
TRT engines are compute-capability specific and built locally — run the scripts
below once per machine; outputs land under `$AVTR1_LOCAL_STORAGE`.

```bash
# Build everything at once
pixi run build-trt-engines

# Or individually
pixi run build-trt-engines-avtr1
pixi run build-trt-engines-renderer
pixi run build-trt-engines-hubert
```

### Run interactive demo
```bash
pixi run interactive-demo
```


### Run offline generation

**Single speaker.** Avatar lip-syncs the given audio track.

```bash
pixi run generate_offline --speech example/speaker_1.ogg

# with a custom avatar and background:
pixi run generate_offline --speech example/speaker_1.ogg --avatar maria --bg minimal_office
```

**Two-speaker dialogue.** Avatar voices `--speech` and reacts (active listening) to the peer audio on `--listen`. Run twice with the tracks swapped to render both sides of the conversation.

```bash
# avatar = speaker 1 (elena)
pixi run generate_offline --speech example/speaker_1.ogg --listen example/speaker_2.ogg --avatar elena  --out elena.mp4
# avatar = speaker 2 (marcus)
pixi run generate_offline --speech example/speaker_2.ogg --listen example/speaker_1.ogg --avatar marcus --out marcus.mp4

# stitch both sides into a single side-by-side video:
ffmpeg -i elena.mp4 -i marcus.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2[v];[0:a][1:a]amix=inputs=2[a]" \
  -map "[v]" -map "[a]" dialogue.mp4
```

**Silence / idle motion.** No audio — renders idle micro-motion for the given duration.

```bash
pixi run generate_offline --duration 10
```

Available avatars are the filenames (without `.png`) inside
`$AVTR1_LOCAL_STORAGE/main/avatars_artifacts/reference_frames/` after downloading.
---

## 2. DigiBox native Windows desktop (no WSL)

> Tauri v2 桌面迁移与分发说明见 [Windows Tauri v2 桌面版构建与分发](docs/windows-tauri-desktop-distribution.md)。真实 Windows 麦克风/WebRTC/安装验收完成前，Electron 入口仍保留为临时回退。

The native Windows path runs directly in PowerShell and does not require WSL,
Docker, pixi, or the Linux `.so` plugin. It uses PyTorch CUDA for the AVTR-1
TorchScript model and ONNX Runtime CUDA for HuBERT and the renderer components.

### Windows one-click local package

If you do not want to install from source, download the
[DigiBox Windows one-click local package](https://pan.quark.cn/s/887c8b103c18).
Extract and keep the complete package directory instead of copying only the
executable. Package contents and version are shown on the share page.

### Prerequisites

- Windows 10 or 11 (64-bit)
- An NVIDIA GPU with a current Windows driver
- [Git](https://git-scm.com/download/win)
- An extracted DigiBox **Full** package with a schema-version 2 `portable-v2`
  Runtime. Keep its complete `avtr-runtime` directory.

Clone the source and point the development setup at that existing Runtime:

```powershell
git clone --recurse-submodules https://github.com/DocWorkBox/DigiBox.git
cd DigiBox
$runtimeRoot = "D:\DigiBox-Full-win64\avtr-runtime"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot
```

`-RuntimeRoot` may name either `avtr-runtime` itself or its parent Full package.
Instead of repeating the option, set `AVTR1_DEV_RUNTIME_ROOT` once:

```powershell
$env:AVTR1_DEV_RUNTIME_ROOT = "D:\DigiBox-Full-win64"
.\scripts\setup_windows.ps1
```

When neither is supplied, the scripts look in `desktop\dist-tauri`, the newest
matching `desktop\builds` output, and the staging Runtime. They reject LegacyV1
and incomplete Runtime manifests.

### Full portable-v2 source development

The source setup scripts reuse `avtr-runtime\python\python.exe`; they do not
create `.venv`, `.venv-cosyvoice`, or `.venv-feynobg`, and they do not reinstall
PyTorch or other Python packages. `setup_windows.ps1` validates the main
profile. The CosyVoice and FeyNoBg setup scripts validate their own dependency
profiles and can fetch their model data; use `-SkipModelDownload` when only an
offline dependency check is wanted.

All three profiles use that same CPython executable with separate ordered
package layers. Repository code takes precedence over packaged source:

- main and FeyNoBg begin with `src` from the checkout;
- CosyVoice begins with checkout `src`, `third_party\CosyVoice`, and
  `third_party\CosyVoice\third_party\Matcha-TTS`;
- only then are the matching Full Runtime package layers added.

This means edits in the checkout are exercised immediately while compiled and
third-party dependencies still come from the reproducible Full Runtime.

TensorRT 10.11 is optional and must already be available in the selected Full
Runtime. Validate its native Windows runtime and builder bindings with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -EnableTensorRT
```

NVIDIA RTX Video Super Resolution is also optional. The UI exposes it as an
independent switch. After provisioning NVIDIA's official VFX Python runtime in
the Full Runtime, validate the import with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -EnableNvidiaVfx
```

With RTX super resolution enabled, the transport mappings are 360p to 720p,
540p to 1080p, and 720p to 1080p. If the official `nvvfx` runtime is absent,
the UI reports the feature as unavailable instead of silently using a normal
resize and calling it RTX.

The Windows source launchers reuse the selected Full Runtime's `models` and
`artifacts\main` directories, including its downloaded weights and locally built
engines. Keep the extracted Full Runtime in a writable location. User uploads
and cloned-voice cache remain outside both trees in DigiBox's user-level
application data.

### Hugging Face access and model download

The AVTR-1 repository is gated. First open
[avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1), accept its
terms, and then authenticate this machine. The diagnostic checks both the CUDA
runtime and access to the gated checkpoint without printing your token:

```powershell
.\scripts\login_huggingface_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\setup_windows.ps1 -RuntimeRoot $runtimeRoot `
  -CheckHuggingFaceAccess
```

The login command prompts securely instead of placing the token in command
history or process arguments. The Full Runtime includes the Xet download backend
so interrupted multi-gigabyte downloads can resume reliably.

Alternatively, setup and download can be performed in one command after login:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
  -RuntimeRoot $runtimeRoot -CheckHuggingFaceAccess -DownloadModels
```

### Run offline generation

The Windows launcher accepts the same arguments as `generate_offline.py`:

```powershell
# Single speaker
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot --speech example\speaker_1.ogg --avatar maria --bg plain_white `
  --out output\maria.mp4

# Idle motion smoke test
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot --duration 1 --avatar maria --bg plain_white --no-mux `
  --out output\windows-smoke.mp4
```

### Run the interactive demo

Local background removal and streaming voice cloning use isolated package
profiles inside the same Full Python Runtime. Validate either profile, and
optionally fetch its model data, before starting the demo:

```powershell
# FeyNoBg person cutout
powershell -ExecutionPolicy Bypass -File .\scripts\setup_feynobg_windows.ps1 `
  -RuntimeRoot $runtimeRoot

# CosyVoice local streaming TTS and voice cloning
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cosyvoice_windows.ps1 `
  -RuntimeRoot $runtimeRoot
```

```powershell
# Auto: native TensorRT when local engines exist, otherwise portable
powershell -ExecutionPolicy Bypass -File .\scripts\run_interactive_windows.ps1 `
  -RuntimeRoot $runtimeRoot

# Full native TensorRT backend, after building with -IncludeWarp
powershell -ExecutionPolicy Bypass -File .\scripts\run_interactive_windows.ps1 `
  -RuntimeRoot $runtimeRoot -Backend tensorrt
```

The interactive launcher normally supervises the whole stack. For a focused
worker check, launch the optional workers directly; both still use the same Full
Python executable:

```powershell
.\scripts\run_feynobg_windows.ps1 -RuntimeRoot $runtimeRoot
.\scripts\run_cosyvoice_windows.ps1 -RuntimeRoot $runtimeRoot
```

Wait for both services to become ready, then open
[http://localhost:7860](http://localhost:7860) in a browser. When their full
installations are present, the launcher also starts FeyNoBg on `127.0.0.1:8767`
and CosyVoice on `127.0.0.1:8768`; otherwise the corresponding optional feature
stays unavailable or uses the built-in fallback. Press `Ctrl+C` in PowerShell
to stop the services.

### Run source tests

The test entrypoint selects the same main profile, first verifies that all three
profiles import repository source, and then forwards the remaining arguments to
pytest:

```powershell
.\scripts\test_windows.ps1 -RuntimeRoot $runtimeRoot `
  -PytestArgs @("-q", "tests")

# Python Tauri contracts plus offline Rust fmt/check/clippy/test
.\scripts\test_tauri_windows.ps1 -RuntimeRoot $runtimeRoot
```

The Tauri test entrypoint also sets `AVTR1_TEST_PYTHON` to the Full Runtime
executable, so Rust supervisor tests never fall back to a repository virtual
environment. `npm run test:tauri` invokes this same entrypoint.

Add `-PlanOnly` to any setup, launcher, worker, or test command to inspect the
resolved Python, Runtime, environment, and arguments without starting it.

For the native source client, `npm run tauri:dev` (or the Electron fallback
`npm run desktop:dev`) uses the same resolver, starts a repository-source
backend, and owns its shutdown. It refuses to start while ports 7860, 8000,
8767, or 8768 are occupied, preventing a new shell from silently attaching to
an older backend.

Qianwen AI Platform (the `aliyun_bailian` provider internally) and MiniMax are
independent providers and require independent
API keys. Their keys are stored independently in the browser's local storage for
the current site address and survive provider switches and app restarts; clear the
corresponding input to remove a saved key. One Qianwen AI Platform key runs the complete Qwen3 ASR + Qwen LLM +
CosyVoice speech chain. MiniMax uses its native historical Realtime API, which
works only when that MiniMax account has access to the interface.

When the loopback CosyVoice endpoint is selected, the settings panel loads
`/v1/audio/voices` into a refreshable dropdown. Cached voices that fail the
duration or acoustic-feature checks remain on disk but are disabled in the UI.
Create or overwrite a voice with a clear 3-10 second WAV, FLAC, OGG, or MP3 and
an exact transcript; unsupported containers, silence, and truncated references
are rejected before the speaker cache is saved.

To use the local Codex login instead of an OpenAI API key, keep Codex Desktop
signed in, select **Codex GPT-Live (experimental)** in the browser, and start
the session. AVTR-1 launches its own authenticated Codex app-server subprocess
and uses the WebRTC V3 voice transport; the browser never asks for or stores an
OpenAI API key for this mode. The integration is experimental because Codex's
realtime app-server protocol is not yet a stable public API.

### First-run conversion and TensorRT

On the first portable pipeline start, Windows prepares dynamic ONNX Runtime
graphs for warp, HuBERT, MODNet, and stitch. The converted graphs are cached
next to the downloaded artifacts, so later starts reuse them. HuBERT is the
largest conversion, so allow several minutes and several additional gigabytes
of free disk space for the first start.

The automatic backend now prefers locally built TensorRT engines on Windows
and falls back to the portable backend when the AVTR-1 engines are absent.
The portable path can also be requested explicitly with `--backend
torchscript`; it prioritizes compatibility and may be slower than the TensorRT
benchmarks below. TensorRT itself supports Windows.
The prebuilt engines and warp plugin shipped by the project are Linux artifacts,
however: engines are GPU/runtime specific, and a Linux `.so` cannot be loaded by
Windows. The build command below therefore creates native engines for the local
GPU instead of reusing the released files.

After installing TensorRT and downloading the gated models, this builds the
AVTR-1, HuBERT, decoder, MODNet, and stitch engines natively on the local GPU:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_tensorrt_windows.ps1 `
  -RuntimeRoot $runtimeRoot
powershell -ExecutionPolicy Bypass -File .\scripts\run_offline_windows.ps1 `
  -RuntimeRoot $runtimeRoot -Backend tensorrt `
  --duration 1 --avatar maria --bg plain_white --no-mux `
  --out output\windows-trt-smoke.mp4
```

That hybrid mode keeps only the volumetric warp stage on ONNX Runtime CUDA,
avoiding the custom plugin while accelerating the other stages with TensorRT.
For the full TensorRT pipeline, add `-IncludeWarp`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_tensorrt_windows.ps1 `
  -RuntimeRoot $runtimeRoot -IncludeWarp
```

When `AVTR1_WARP_PLUGIN` is not already set, that command builds
`grid_sample_3d_plugin.dll` natively. The Windows builder pins the upstream
plugin and TensorRT 10.11.0.33 headers, generates the MSVC import library from
the installed TensorRT DLL, patches the plugin to use runtime dimensions for
dynamic batch 1 through 5, applies the CUDA 12 FP16 coordinate/accumulation fix,
and compiles for the current GPU architecture. You
can build only the DLL with
`scripts\build_warp_plugin_windows.ps1`. A successful warp-engine build
copies the DLL beside the engine, so later PowerShell sessions do not need the
environment variable. Engines are stored in Windows-specific `*_cc_win64`
directories and cannot be shared with Linux.

Native Windows currently requires NVIDIA CUDA and the ONNX Runtime
`CUDAExecutionProvider`; CPU-only rendering is not supported. The first model
download is several gigabytes, and interactive use additionally requires a
browser microphone plus a configured conversation provider.

---

## 3. Performance

### Per-chunk latency

AVTR-1 generates motion in 5-frame chunks end-to-end. At 25 fps that's 200 ms
of output per chunk, so any GPU under that line runs in real-time.

| GPU         | Latency / 5-frame chunk | Real-time factor |
| ----------- | ----------------------- | ---------------- |
| L40         | 84 ms                   | 2.4×             |
| A100        | 91 ms                   | 2.2×             |
| RTX 4060 Ti | 166 ms                  | 1.2×             |
| RTX 3070    | 181 ms                  | 1.1×             |
| L4          | 202 ms                  | 0.99×            |
| RTX 3060 Ti | 206 ms                  | 0.97×            |
| RTX 4060    | 232 ms                  | 0.86×            |

Real-time factor = 200 ms / latency. ≥ 1.0× means the GPU keeps up with 25 fps.

---

## Troubleshooting

<details>
<summary><b>TURN server setup</b> (optional)</summary>

ICE tries direct UDP first (host candidates + STUN-reflexive
candidates from a public STUN server) and only needs a TURN relay when the
network in between can't pass UDP between browser and streamer — typical
when the streamer lives on a cloud VM whose security group blocks inbound
UDP, or when one peer is behind symmetric NAT.

If direct UDP works for your setup you can skip this section entirely. The
browser's connectivity card after the engine dropdown tells you which path
ICE actually picked, and the same UI links back here when the verdict is
"only TURN works" or "nothing worked".

The project is wired for **Cloudflare's Realtime TURN**. The free tier is
generous enough for development; no credit card required.

**1. Create a TURN application on Cloudflare**

- Sign in to [dash.cloudflare.com](https://dash.cloudflare.com).
- Navigate to **Realtime → TURN Server**.
- Click **Create TURN App**, give it a name (e.g. `avtr1-dev`), and submit.

**2. Copy the two credential values**

On the application's detail page you'll see:

- **Turn Key ID** — short identifier (looks like a UUID without dashes).
- **API Token** — long secret shown only once at creation. Save it before
  navigating away.

**3. Put them in `.env`**

```dotenv
CLOUDFLARE_TURN_KEY_ID="<Turn Key ID>"
CLOUDFLARE_TURN_KEY_TOKEN="<API Token>"
```

That's it. On the next `/ice-servers` request the streamer mints a fresh,
short-lived TURN credential per session via Cloudflare's
`/v1/turn/keys/{kid}/credentials/generate` endpoint — the long-lived API
token never leaves the server. You can verify it picked up the keys by
watching the streamer log for `ice: using Cloudflare TURN` on the first
browser request.

The browser-side connectivity probe (the small status card under the
controls) tells you which ICE path actually wins:

- ✓ **host** — the browser saw its own local interface; always present.
- ✓ **server-reflexive via STUN** — the browser learned its public IP via
  STUN; doesn't prove the streamer is reachable on UDP from the browser.
- ✓ **relay via TURN** — the browser successfully allocated a Cloudflare
  TURN relay; required when direct UDP can't traverse the network in
  between.

If the relay check fails while TURN is configured the most likely cause is
wrong credentials — re-check that you copied the full **API Token** (not
the Key ID twice) into `CLOUDFLARE_TURN_KEY_TOKEN`.

**Alternatives.** Anything that speaks the standard TURN protocol works.
Set `TURN_URL` (and optionally `TURN_USERNAME` / `TURN_CREDENTIAL`) instead
of the Cloudflare variables and `resolve_ice_servers()` will use it
verbatim — e.g. a self-hosted [coturn](https://github.com/coturn/coturn) on
a small VM. STUN-only also works *if* you can open the appropriate UDP
port range inbound on whatever firewall sits in front of the streamer.

</details>

---

## License

The complete source is public, but the repository is **multi-license and
source-available rather than OSI-open-source as a whole**. In particular, the
upstream Renderer and Streamer are restricted to noncommercial use. DigiBox's
new desktop shell, packaging, and tests are Apache-2.0, but that license does
not override any upstream component license.

This repository contains three separately licensed upstream components plus
the DigiBox additions:

- **`desktop/`, `src-tauri/`, `tests/`, and `docs/`** — DigiBox-specific
  desktop shell, packaging, tests, and documentation under Apache-2.0
  ([LICENSE-DIGIBOX.md](LICENSE-DIGIBOX.md)). This license applies only to
  DigiBox additions and does not override any upstream component license.

- **`scripts/`** — build and demo tooling, released under the **AVTR-1
  Community License** ([LICENSE-MODEL.md](LICENSE-MODEL.md)). Permits
  commercial use by entities under USD 10M annual revenue; entities at or
  above that threshold need a commercial agreement. The same license governs
  the AVTR-1 weights distributed at
  [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1).
- **`src/avtr1_renderer/`** — Avaturn Renderer (inference pipeline), released
  under the **PolyForm Noncommercial License 1.0.0** with a Required Notice
  ([LICENSE-RENDERER.md](LICENSE-RENDERER.md)). **Noncommercial use only**,
  regardless of revenue; any commercial use needs a separate Renderer
  Commercial License.
- **`src/avaturn_live_streamer/`** — Avaturn Streamer (orchestration
  backend), released under the **PolyForm Noncommercial License 1.0.0** with
  a Required Notice and patent reservation
  ([LICENSE-STREAMER.md](LICENSE-STREAMER.md),
  [PATENTS.md](PATENTS.md)). **Noncommercial use only**, regardless of
  revenue; any commercial use needs a separate Streamer Commercial License.

See [LICENSE.md](LICENSE.md) for the full component map and the consequences
of the multi-license structure. In any conflict between this summary and the
underlying license files, the license files control.

### Non-commercial dependency

The pipeline uses InsightFace's pretrained SCRFD detector and 2D106 landmark
model, which are licensed for **non-commercial research use only**. To use
AVTR-1 commercially you must either obtain a commercial license from
InsightFace (deepinsight@gmail.com) or replace these models with
permissively-licensed alternatives (e.g., MediaPipe). See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full picture.

**Commercial inquiries:** hello@avaturn.me
