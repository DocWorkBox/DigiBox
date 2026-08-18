# Theme-aware live background and avatar idle loop implementation plan

> Date: 2026-08-06

## Goal

Keep the selected visual theme continuous before and during a conversation, and keep the selected avatar visible before the WebRTC session starts by generating a reusable listening/idle loop when the avatar is uploaded.

## Root cause

- The browser theme currently changes only CSS variables.
- `/offer` still sends a separate `sessionBackgroundId`, initialized and reset to `tech_particles_dark`.
- The renderer startup registers only `tech_particles_dark`, even though all theme PNGs are already packaged.
- The session stage is hidden whenever a conversation is not open, and the live video is revealed when the peer connection reports `connected`, before the first decoded frame is necessarily available.

## Design

### Theme-to-renderer contract

- Add `rendererBackgroundId` to every browser theme definition.
- Register the preset plus every bundled `theme_*.png` in the renderer background registry.
- Send the active theme's renderer background ID in `/offer`.
- Validate that the selected renderer background is present before starting a session.

### Idle/listening asset

- Render a short, deterministic, silent two-track AVTR sequence for the selected avatar. AVTR's `audio_listen` path remains active, so the output is a neutral listening/idle motion rather than a speaking clip.
- Render against the reserved transparent background and export straight RGBA directly before AVTR's YUV 4:2:0 packing stage. This avoids amplified chroma/luma quantisation around semi-transparent hair.
- Ping-pong the generated frames so the cached animation loops without a hard end-to-start jump.
- Discard the first cold-start motion chunk, then keep 25 source frames and ping-pong them into a seamless 48-frame loop.
- Store the versioned asset at `user_assets/idle_loops/v1/{avatar_id}.webp`; reuse it after restart and move it to recoverable trash when the avatar is deleted.
- Generate it during avatar upload, with a lazy GET fallback for avatars that predate this feature. If generation fails, keep the upload usable and fall back to the static PNG preview.

### Browser transition

- Keep the session stage visible whenever an avatar is selected, but leave it pointer-transparent before a session.
- Layer static poster, animated idle WebP, and live WebRTC video in the same 16:9 shell.
- Reveal the live layer only after `requestVideoFrameCallback` (or `loadeddata` fallback) confirms the first decoded frame.
- Crossfade idle to live; restore idle on stop, disconnect, or failed start.
- Under `prefers-reduced-motion`, keep the static poster and disable motion/crossfade.

## TDD sequence

1. Add failing UI source-contract tests for theme renderer IDs, active-theme `/offer`, always-available idle layer, first-frame reveal, and static fallback.
2. Add failing renderer tests for the bundled background manifest, stacked-alpha decoding/WebP encoding, idle cache route, and deletion lifecycle.
3. Add failing local-stream proxy tests for the idle asset route.
4. Implement the renderer background manifest and idle-loop helper.
5. Wire upload generation, lazy route, cache headers, proxying, and recoverable deletion.
6. Implement the UI state/layers and first-frame crossfade.
7. Run focused tests, full tests, compile checks, and browser QA at portrait, desktop, and wide viewports.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_stream_ui.py tests\renderer\test_avatar_idle_loop.py tests\renderer\test_person_asset_flow.py tests\localrtc\test_avatar_preview_proxy.py tests\scripts\test_run_local_stream_windows.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile src\avtr1_renderer\api\app.py src\avtr1_renderer\idle_loop.py src\avaturn_live_streamer\local_stream_cli.py
```

Browser QA must confirm:

- switching theme changes the full viewport and the live rendered background;
- a selected avatar is visible and moving before Start;
- Start keeps the idle layer visible until the first live frame, then crossfades without a blank flash;
- End restores the idle loop and keeps Start usable;
- resize does not stretch the avatar or expose a mismatched background.
