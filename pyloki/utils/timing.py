from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import ContextDecorator
from ctypes.util import find_library
from typing import TYPE_CHECKING, ClassVar, Self

import attrs
import numpy as np
from llvmlite.binding import load_library_permanently
from numba import njit
from numba.core import types, typing

if TYPE_CHECKING:
    from collections.abc import Callable


# Locate the standard C library
libc_path = find_library("c")
load_library_permanently(libc_path)
clock_name = "clock"
return_type = types.int64
c_sig = typing.signature(return_type)
clock = types.ExternalFunction(clock_name, c_sig)


@njit(cache=True)
def nb_time_now() -> float:
    time = clock()
    return float(time) / 1000000  # convert to seconds (for POSIX systems)


@attrs.define(slots=True)
class TimingStats:
    count: int = attrs.field(default=0, init=False)
    total: float = attrs.field(default=0.0, init=False)
    min: float = attrs.field(default=np.inf, init=False)
    max: float = attrs.field(default=0.0, init=False)
    _values: list[float] = attrs.field(factory=list, repr=False, init=False)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self._values.append(value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0

    @property
    def median(self) -> float:
        return statistics.median(self._values) if self._values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self._values) if len(self._values) > 1 else 0.0


@attrs.define(auto_attribs=True, slots=True)
class Timer(ContextDecorator):
    timers: ClassVar[defaultdict[str, TimingStats]] = defaultdict(TimingStats)
    name: str | None = None
    logger: Callable[[str], None] | None = None
    text: str | Callable[[float], str] = attrs.field(init=False)
    last: float = attrs.field(default=np.nan, init=False)
    _start_time: float | None = attrs.field(default=None, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self.name:
            self.text = f"{self.name} finished, Elapsed time: {{:.3f}} seconds"
        else:
            self.text = "Elapsed time: {:.3f} seconds"

    def start(self) -> None:
        """Start a new timer."""
        if self._start_time is not None:
            msg = f"Timer {self.name} is already running. Call stop() first."
            raise RuntimeError(msg)
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the current timer and return the elapsed time."""
        if self._start_time is None:
            msg = f"Timer {self.name} is not running. Call start() first."
            raise RuntimeError(msg)
        self.last = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            if callable(self.text):
                text = self.text(self.last)
            else:
                text = self.text.format(self.last)
            self.logger(text)
        if self.name:
            self.timers[self.name].add(self.last)
        return self.last

    def __enter__(self) -> Self:
        """Start a new timer as a context manager."""
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:  # noqa: ANN002
        """Stop the context manager timer."""
        self.stop()
