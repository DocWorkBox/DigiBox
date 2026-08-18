# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Per-model I/O dataclasses and engine type aliases.

One file per model. Each module defines a ``${Name}Input`` and
``${Name}Output`` dataclass whose field names match the underlying engine's
tensor names exactly, plus a ``${Name}Engine`` type alias for
``InferenceEngine[Input, Output]``. Pre/post-processing helpers live with
the call sites in ``avtr1_renderer.components`` -- this package is the
contract layer only.
"""

from avtr1_renderer.models.appearance_extractor import (
    AEInput,
    AEOutput,
    AppearanceExtractorEngine,
)
from avtr1_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from avtr1_renderer.models.face_detection import (
    FaceDetEngine,
    FaceDetInput,
    FaceDetOutput,
)
from avtr1_renderer.models.hubert import HubertEngine, HubertInput, HubertOutput
from avtr1_renderer.models.landmark106 import Lm106Engine, Lm106Input, Lm106Output
from avtr1_renderer.models.landmark203 import Lm203Engine, Lm203Input, Lm203Output
from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeEngine,
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeEngine,
    Avtr1EncodeInput,
    Avtr1EncodeOutput,
)
from avtr1_renderer.models.matting import MODNetEngine, MODNetInput, MODNetOutput
from avtr1_renderer.models.motion_extractor import (
    MotionExtractorEngine,
    MotionInput,
    MotionOutput,
)
from avtr1_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from avtr1_renderer.models.warp import WarpEngine, WarpInput, WarpOutput

__all__ = [
    "AEInput",
    "AEOutput",
    "AppearanceExtractorEngine",
    "DecoderEngine",
    "DecoderInput",
    "DecoderOutput",
    "FaceDetEngine",
    "FaceDetInput",
    "FaceDetOutput",
    "HubertEngine",
    "HubertInput",
    "HubertOutput",
    "Lm106Engine",
    "Lm106Input",
    "Lm106Output",
    "Lm203Engine",
    "Lm203Input",
    "Lm203Output",
    "Avtr1DecodeEngine",
    "Avtr1DecodeInput",
    "Avtr1DecodeOutput",
    "Avtr1EncodeEngine",
    "Avtr1EncodeInput",
    "Avtr1EncodeOutput",
    "MODNetEngine",
    "MODNetInput",
    "MODNetOutput",
    "MotionExtractorEngine",
    "MotionInput",
    "MotionOutput",
    "StitchEngine",
    "StitchInput",
    "StitchOutput",
    "WarpEngine",
    "WarpInput",
    "WarpOutput",
]
