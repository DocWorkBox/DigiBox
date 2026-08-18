# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Atomic pipeline components.

Stateless building blocks used by ``avtr1_renderer.pipeline``: each
module wraps either a single TRT engine plus its I/O dataclasses, or a
small piece of pure-tensor / pure-numpy logic. Components don't know
about each other -- the orchestrator in ``pipeline.model`` composes
them.

Submodules:
- ``components.<name>`` -- generic atoms (HuBERT, gesture decoder,
  face detection / landmarks, putback, source-crop math).
- ``components.liveportrait`` -- the LivePortrait family
  (appearance / motion extractors, warp / stitch / SPADE decoder,
  motion-stitch math).
"""
