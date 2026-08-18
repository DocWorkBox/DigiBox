# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from datetime import date, datetime, timedelta, timezone


def tzutcnow() -> datetime:
    """Creates tz-aware datetime in UTC"""
    return datetime.now(timezone.utc)


def datenow() -> date:
    """Creates current date in UTC"""
    return datetime.now(timezone.utc).date()


def timestamp_to_tzutctime(timestamp: int) -> datetime:
    """Converts timestamp to tz-aware datetime in UTC"""
    return datetime.fromtimestamp(timestamp, timezone.utc)


def timestamp_to_date(timestamp: int) -> date:
    """Converts timestamp to date in UTC"""
    return datetime.fromtimestamp(timestamp, timezone.utc).date()


def timedelta_to_float_minutes(delta: timedelta | None) -> float | None:
    if delta is None:
        return None
    return delta.total_seconds() / 60
