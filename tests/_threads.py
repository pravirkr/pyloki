"""Run code at a given numba thread count."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numba

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def num_threads(n: int) -> Iterator[None]:
    """Set numba's thread count to `n`, restoring the previous count on exit."""
    previous = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(previous)
