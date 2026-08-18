# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from fractions import Fraction
from functools import cached_property
from typing import Self, Sequence

import attrs
import numpy as np
import soxr
from numpy.typing import NDArray

from avaturn_live_streamer.types import Duration

_INT16_LE = np.dtype("int16").newbyteorder("<")


@attrs.define(repr=False)
class SpeechBuffer:
    _buffer: NDArray[np.int16]
    sample_rate: int

    @classmethod
    def from_bytes(cls, buf: bytes, sample_rate: int) -> Self:
        array = np.frombuffer(buf, dtype=_INT16_LE)
        return cls(array, sample_rate)

    @cached_property
    def duration(self) -> Fraction:
        return Fraction(self._buffer.shape[0], self.sample_rate)

    @cached_property
    def is_empty(self):
        return self.sample_rate == 1 and len(self._buffer) == 0

    def resample(self, sample_rate: int) -> Self:
        if self.is_empty:
            return self

        if sample_rate == self.sample_rate:
            return self

        # We use QQ quality because it provides fewer artifacts on boundaries
        # in a stateless chunked setup compared to higher quality settings
        # We don't use stateful streaming resampler cause it's harder to implement
        # the only viable option is again QQ with initial few-sample padding (1-4 depending on in and out sample rates)
        res = soxr.resample(self._buffer, self.sample_rate, sample_rate, quality="QQ")
        return self.__class__(res, sample_rate)

    def slice(self, start: Duration, end: Duration) -> Self:
        start_sample = self._duration_to_samples(start)
        end_sample = self._duration_to_samples(end)
        return self.__class__(self._buffer[start_sample:end_sample], self.sample_rate)

    def _duration_to_samples(self, duration: Duration) -> int:
        return int(duration * self.sample_rate)  # pyright: ignore [reportArgumentType]

    def to_bytes(self) -> bytes:
        if self.is_empty:
            return b""

        return self._buffer.astype(_INT16_LE).tobytes()

    def __add__(self, other: Self) -> Self:
        if self.is_empty:
            return other

        if other.is_empty:
            return self

        if not isinstance(other, self.__class__):
            raise NotImplementedError
        if self.sample_rate != other.sample_rate:
            raise ValueError("Incompatible sampling rates")

        return self.__class__(np.concat([self._buffer, other._buffer]), self.sample_rate)

    @classmethod
    def concat(cls, chunks: Sequence[Self]) -> Self:
        chunks = [c for c in chunks if not c.is_empty]

        if len(chunks) == 0:
            return cls.empty()

        sample_rate = chunks[0].sample_rate
        if any([sample_rate != ch.sample_rate for ch in chunks[1:]]):
            raise ValueError("Incompatible sampling rates")

        return cls(np.concat([ch._buffer for ch in chunks]), sample_rate)

    def split_in_two_parts_with_duration(self, first_part_duration: Duration) -> tuple[Self, Self]:
        if self.is_empty:
            return self, self

        if first_part_duration == self.duration:
            return self, self.empty()

        if first_part_duration == 0:
            return self.empty(), self

        samples_to_extract = self._duration_to_samples(first_part_duration)
        extracted_samples, remaining_samples = (
            self._buffer[:samples_to_extract],
            self._buffer[samples_to_extract:],
        )
        return (
            self.__class__(extracted_samples, self.sample_rate),
            self.__class__(remaining_samples, self.sample_rate),
        )

    def get_tail(self, duration: Duration) -> Self:
        if self.is_empty:
            return self

        if duration == 0:
            return self.empty()

        samples_to_keep = self._duration_to_samples(duration)
        start_index = max(len(self._buffer) - samples_to_keep, 0)
        return self.__class__(self._buffer[start_index:], self.sample_rate)

    def is_silent(self) -> bool:
        if self.is_empty:
            return True

        return (self._buffer == 0).all().item()

    @classmethod
    def silence(cls, duration: Duration, sample_rate: int) -> Self:
        return cls.full(0, duration, sample_rate)

    @classmethod
    def full(cls, fill_value: int, duration: Duration, sample_rate: int):
        samples = duration * sample_rate
        assert samples - int(samples) == 0  # pyright: ignore [reportArgumentType, reportOperatorIssue]
        buffer = np.full(
            shape=(int(samples),),  # pyright: ignore [reportArgumentType],
            fill_value=fill_value,
            dtype=_INT16_LE,
        )
        return cls(buffer, sample_rate)

    @classmethod
    def empty(cls) -> Self:
        return cls(np.empty(0, dtype=_INT16_LE), sample_rate=1)

    @classmethod
    def read(cls, file: str) -> Self:
        import soundfile

        buf, sr = soundfile.read(file, dtype="int16")
        assert len(buf.shape) == 1
        return cls(buf, sr)  # pyright: ignore [reportArgumentType]

    def write(self, file: str):
        import soundfile

        assert not self.is_empty
        soundfile.write(file, self._buffer, self.sample_rate, "PCM_16")

    def __repr__(self) -> str:
        if self.is_empty:
            params = "<empty>"
        else:
            params = (
                f"duration={float(self.duration):.4f}, sample_rate={self.sample_rate // 1000}kHz"
            )
        return f"SpeechBuffer({params})"
