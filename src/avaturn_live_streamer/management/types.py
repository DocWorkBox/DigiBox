# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import NewType
from uuid import UUID

from typeid import TypeID
from uuid6 import uuid7

StreamId = NewType("StreamId", str)
SegmentId = NewType("SegmentId", str)
SegmentMetadata = dict[str, str]


def make_stream_id(uuid: UUID) -> StreamId:
    return StreamId(str(TypeID.from_uuid(uuid, "stream")))  # pyright: ignore[reportArgumentType]


def stream_id_to_uuid(stream_id: StreamId) -> UUID:
    return TypeID.from_string(stream_id).uuid


def make_segment_id() -> SegmentId:
    return SegmentId(str(TypeID.from_uuid(uuid7(), "segment")))
