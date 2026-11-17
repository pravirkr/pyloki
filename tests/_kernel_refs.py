"""Grid, timing and phase helpers shared by the Taylor, Chebyshev and circular tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from collections.abc import Callable

NBINS = 64
LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])  # [accel, freq]
COUNTS = np.array([16, 64])

# T_ADD - T_INIT is not a whole number of cycles at f0, so a wrong proper time shows.
T_INIT, T_ADD = 10.0, 30.0037

# A leaf's velocity at t_init is its frequency offset: dv = (c / f0) df. One frequency
# cell of this grid (2/64 Hz at f0 = 100 Hz) is about 9.4e4 m/s.
CELL_DV = (2 / 64) * C_VAL / 100.0


def grid_cell(value: float, limits: np.ndarray, count: int) -> int:
    """Index of the grid cell holding `value` (clipped to the grid)."""
    lo, hi = limits
    return int(np.clip(np.floor(count * (value - lo) / (hi - lo)), 0, count - 1))


def expected_phase(dt: float, delay: float, f0: float) -> float:
    """Phase in bins of a pulse emitted `dt` after t_init and delayed by `delay`."""
    return ((dt - delay) * f0) % 1 * NBINS


def assert_phase(got: float, expected: float) -> None:
    """Compare phases on the circle of NBINS bins (63.999... and 0 are equal)."""
    diff = (got - expected + NBINS / 2) % NBINS - NBINS / 2
    assert abs(diff) < 1e-6, (got, expected)


def instantaneous_cell(
    resolve_at: Callable[[float], tuple[np.ndarray, np.ndarray]],
    t_add: float,
    f0: float,
) -> tuple[int, int]:
    """Return (picked freq cell, cell of the derivative of the returned phase).

    `resolve_at(t)` returns `(param_idx, phase)` for one leaf resolved at time `t`.
    The instantaneous frequency is the central difference of the phase, unwrapped
    around f0.
    """
    eps = 1e-3
    hi = resolve_at(t_add + eps)[1] / NBINS
    lo = resolve_at(t_add - eps)[1] / NBINS
    dcyc = (hi - lo + 0.5) % 1 - 0.5
    f_inst = (dcyc + round(f0 * 2 * eps - dcyc)) / (2 * eps)
    return resolve_at(t_add)[0][1], grid_cell(f_inst, LIMITS[1], COUNTS[1])
