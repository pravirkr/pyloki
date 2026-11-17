"""Grid, phase and branching references for the Taylor, Chebyshev, circular tests."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyloki.utils import psr_utils
from pyloki.utils.misc import C_VAL
from tests.jit_utils import python_impl

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


def expected_children(
    values: np.ndarray,
    steps: np.ndarray,
    dnew: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Children of each parent, in the kernels' order, from `psr_utils.branch_param`.

    `values`, `steps`, `dnew` and `shift` have shape ``(n_parents, n_params)``. A
    parameter whose shift is under one bin keeps its value and step; any other is
    split by the non-padded `branch_param`. Children are the Cartesian product of
    the per-parameter values, parent by parent.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Child values and steps, both ``(n_children, n_params)``, and their origins.
    """
    rows, row_steps, origins = [], [], []
    for i in range(len(values)):
        axes, dact = [], []
        for j in range(values.shape[1]):
            if shift[i, j] < 1.0 - 1e-12:
                axes.append([values[i, j]])
                dact.append(steps[i, j])
            else:
                vals, d = psr_utils.branch_param(values[i, j], steps[i, j], dnew[i, j])
                axes.append(list(vals))
                dact.append(d)
        for combo in itertools.product(*axes):
            rows.append(combo)
            row_steps.append(dact)
            origins.append(i)
    return np.array(rows), np.array(row_steps), np.array(origins)


def descendant_weighted_mean(singles: np.ndarray) -> np.ndarray:
    """Branching pattern of a frequency mix, from its single-frequency patterns.

    With single-frequency patterns ``n_i`` (rows of `singles`), level ``l`` of the
    mix is ``sum_i W_i n_i[l] / sum_i W_i`` with ``W_i = prod_{m < l} n_i[m]``.
    """
    weights = np.cumprod(
        np.hstack([np.ones((len(singles), 1)), singles[:, :-1]]),
        axis=1,
    )
    return (weights * singles).sum(axis=0) / weights.sum(axis=0)


def paired_jit_variants(func: Callable, other: Callable) -> list:
    """Parametrise ``(impl, other_impl)`` over compiled/compiled and py_func/py_func.

    For a test comparing two kernels: crossing a compiled kernel with the other's
    ``py_func`` covers nothing the matched pairs do not.
    """
    return [
        pytest.param(func, other, id="compiled"),
        pytest.param(python_impl(func), python_impl(other), id="py_func"),
    ]
