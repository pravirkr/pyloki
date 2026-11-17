"""The moving-grid resolvers pick the frequency cell of their own phase model.

A resolver returns the phase of the leaf at the added segment and the FFA grid cell to
read that segment from. The frequency of that cell should be the instantaneous
frequency of the returned phase at ``t_add``, i.e. its time derivative, whatever
convention the grid uses for ``f0``. On the moving grid a leaf's ``f0`` is never
updated, so a child's frequency offset from the seed is carried by its velocity at
``t_init``; the phase keeps it through the delay, and the frequency must keep it too.

The velocity offsets span a whole frequency cell (``CELL_DV``), so a resolver that
drops ``v(t_init)`` picks the neighbouring cell for some of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.core import chebyshev, circular, taylor
from pyloki.utils import transforms
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

NBINS = 64
LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])  # [accel, freq]
COUNTS = np.array([16, 64])
F0 = 100.0
# T_ADD - T_INIT is not a whole number of cycles at F0.
T_INIT, T_ADD = 10.0, 30.0037
CELL_DV = (2 / 64) * C_VAL / F0  # one frequency cell of velocity at F0
OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5)
ACCEL = 5.0


def _cell(value: float) -> int:
    lo, hi = LIMITS[1]
    return int(
        np.clip(np.floor(COUNTS[1] * (value - lo) / (hi - lo)), 0, COUNTS[1] - 1),
    )


def _picked_and_expected(resolve) -> tuple[int, int]:
    """Return (picked freq cell, cell of the derivative of the returned phase)."""
    eps = 1e-3
    hi = resolve(T_ADD + eps)[1] / NBINS
    lo = resolve(T_ADD - eps)[1] / NBINS
    dcyc = (hi - lo + 0.5) % 1 - 0.5
    f_inst = (dcyc + round(F0 * 2 * eps - dcyc)) / (2 * eps)
    return resolve(T_ADD)[0][1], _cell(f_inst)


def _taylor_state(vel_offset: float, t_cur: float) -> list[float]:
    """Return [jerk, accel, vel, pos] at t_cur of x = v dt + a dt^2 / 2 about t_init."""
    dt = t_cur - T_INIT
    return [0.0, ACCEL, vel_offset + ACCEL * dt, vel_offset * dt + ACCEL * dt**2 / 2]


@pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
@pytest.mark.parametrize("t_cur", [T_INIT, 18.3719])
@pytest.mark.parametrize("frac", OFFSETS)
def test_taylor(impl, t_cur: float, frac: float) -> None:
    leaf = np.zeros((1, 5, 2))
    leaf[0, :4, 0] = _taylor_state(frac * CELL_DV, t_cur)
    leaf[0, -1, 0] = F0

    def resolve(t: float) -> tuple[np.ndarray, float]:
        idx, ph = impl(
            leaf,
            (t, 1.0),
            (t_cur, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        return idx[0], ph[0]

    picked, expected = _picked_and_expected(resolve)
    assert picked == expected


@pytest.mark.parametrize("impl", jit_variants(chebyshev.poly_chebyshev_resolve_batch))
@pytest.mark.parametrize("t_cur", [T_INIT, 18.3719])
@pytest.mark.parametrize("frac", OFFSETS)
def test_chebyshev(impl, t_cur: float, frac: float) -> None:
    scale = 5.0
    leaf = np.zeros((1, 5, 2))
    state = np.array(_taylor_state(frac * CELL_DV, t_cur))
    leaf[0, :4, 0] = transforms.taylor_to_cheby(state, scale)
    leaf[0, -1, 0] = F0

    def resolve(t: float) -> tuple[np.ndarray, float]:
        idx, ph = impl(
            leaf,
            (t, 1.0),
            (t_cur, scale),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        return idx[0], ph[0]

    picked, expected = _picked_and_expected(resolve)
    assert picked == expected


def _circular_leaf(kind: str, vel_offset: float) -> np.ndarray:
    """Return a leaf the classifier treats as Taylor, or one on a circular orbit."""
    leaf = np.zeros((1, 7, 2))
    leaf[0, -1, 0] = F0
    if kind == "taylor":
        leaf[0, :6, 0] = [0.0, 0.0, 0.0, ACCEL, vel_offset, 0.0]
        leaf[0, :6, 1] = [1.0, 1.0, 1.0, 0.01, 10.0, 0.0]
        return leaf
    # x(t) = A sin(w (t - t_init) + phi) + vel_offset (t - t_init), at t_init.
    a, w, phi = 3e4, 2 * np.pi / 400.0, 0.7
    state = np.array(
        [
            a * w**5 * np.cos(phi),
            a * w**4 * np.sin(phi),
            -a * w**3 * np.cos(phi),
            -a * w**2 * np.sin(phi),
            a * w * np.cos(phi) + vel_offset,
            a * np.sin(phi),
        ],
    )
    leaf[0, :6, 0] = state
    leaf[0, :6, 1] = np.abs(state) * 1e-3 + 1e-12
    return leaf


@pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
@pytest.mark.parametrize("kind", ["taylor", "circular"])
@pytest.mark.parametrize("frac", OFFSETS)
def test_circular(impl, kind: str, frac: float) -> None:
    leaf = _circular_leaf(kind, frac * CELL_DV)

    def resolve(t: float) -> tuple[np.ndarray, float]:
        idx, ph = impl(
            leaf.copy(),
            (t, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            2.0,
        )
        return idx[0], ph[0]

    picked, expected = _picked_and_expected(resolve)
    assert picked == expected
