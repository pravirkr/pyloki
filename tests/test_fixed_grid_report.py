"""On the fixed grid, `report` re-centres the leaves to the report coordinate.

The fixed-grid leaves are expansions about the reference segment, ``coord_end``; the
report shifts them to ``coord_report`` before converting them. The moving grid's leaves
are already at the report coordinate and are reported as they are.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import typed

from pyloki.core import chebyshev, taylor
from pyloki.dynamic import dyn_circular_taylor, dyn_poly_cheby, dyn_poly_taylor
from pyloki.utils import transforms
from pyloki.utils.misc import C_VAL

NBINS = 32
T_INIT, T_REPORT = 10.0, 30.0
COORD_END, COORD_REPORT = (T_INIT, 5.0), (T_REPORT, 20.0)


def _funcs(init: object, order: int, *extra: float, moving: bool) -> object:
    higher = [np.array([0.0])] * (order - 2)
    param_arr = typed.List([*higher, np.array([-1.0, 1.0]), np.array([99.5, 100.5])])
    limits = np.array([*[[-1e-3, 1e-3]] * (order - 2), [-20.0, 20.0], [99.0, 101.0]])
    return init(
        param_arr,
        np.array([1e-4] * (order - 2) + [0.05, 0.01]),
        np.array([4, 8]),
        10.0,
        NBINS,
        1.0,
        limits,
        128,
        np.array([1, 2, 4]),
        order,
        64,
        "aggressive",
        *extra,
        moving,
    )


def _taylor(*, moving: bool) -> object:
    return _funcs(dyn_poly_taylor.prune_poly_taylor_dp_functs_init, 3, moving=moving)


def _cheby(*, moving: bool) -> object:
    return _funcs(dyn_poly_cheby.prune_chebyshev_dp_functs_init, 3, moving=moving)


def _circ(*, moving: bool) -> object:
    # p_orb_min, x_mass_const, propagator and validation significance
    extra = (100.0, 1e12, 2.0, 5.0)
    init = dyn_circular_taylor.prune_circ_taylor_dp_functs_init
    return _funcs(init, 5, *extra, moving=moving)


def _taylor_leaves(order: int) -> np.ndarray:
    """Three leaves with an acceleration and a velocity, [d_k .. d_0, (f0, flag)]."""
    leaves = np.zeros((3, order + 2, 2))
    leaves[:, order - 2, 0] = [5.0, -3.2, 1.7]
    leaves[:, order - 1, 0] = [0.0, 2.1e4, -1.3e4]
    leaves[:, :order, 1] = [*([1e-6] * (order - 2)), 0.05, 10.0]
    leaves[:, -1, 0] = [100.0, 100.1, 99.8]
    return leaves


def test_taylor_reports_the_frequency_at_the_report_time() -> None:
    """A leaf with accel 5 m/s^2 and f = 100 Hz at t_init, reported 20 s later."""
    leaves = _taylor_leaves(3)[:1]
    got = _taylor(moving=False).report(leaves.copy(), COORD_REPORT, COORD_END)
    expected = 100.0 * (1 - 5.0 * (T_REPORT - T_INIT) / C_VAL)
    assert got[0, 2, 0] == pytest.approx(expected, rel=1e-14)
    assert got[0, 2, 0] != pytest.approx(100.0, rel=1e-10)


def test_taylor_fixed_grid() -> None:
    leaves = _taylor_leaves(3)
    got = _taylor(moving=False).report(leaves.copy(), COORD_REPORT, COORD_END)
    recentred = taylor.poly_taylor_transform_batch(
        leaves,
        COORD_REPORT,
        COORD_END,
        "aggressive",
    )
    np.testing.assert_allclose(got, taylor.poly_taylor_report_batch(recentred))


def test_chebyshev_fixed_grid() -> None:
    leaves = _taylor_leaves(3)
    leaves[:, :-1] = transforms.taylor_to_cheby_full(leaves[:, :-1], COORD_END[1])
    got = _cheby(moving=False).report(leaves.copy(), COORD_REPORT, COORD_END)
    recentred = chebyshev.poly_chebyshev_transform_batch(
        leaves,
        COORD_REPORT,
        COORD_END,
        "aggressive",
    )
    np.testing.assert_allclose(
        got,
        chebyshev.poly_chebyshev_report_batch(recentred, COORD_REPORT),
    )


def test_circular_fixed_grid() -> None:
    leaves = _taylor_leaves(5)
    got = _circ(moving=False).report(leaves.copy(), COORD_REPORT, COORD_END)
    recentred = taylor.poly_taylor_transform_batch(
        leaves,
        COORD_REPORT,
        COORD_END,
        "aggressive",
    )
    np.testing.assert_allclose(got, taylor.poly_taylor_report_batch(recentred))


@pytest.mark.parametrize("make", [_taylor, _circ])
def test_moving_grid_reports_the_leaves_as_they_are(make: object) -> None:
    leaves = _taylor_leaves(3 if make is _taylor else 5)
    got = make(moving=True).report(leaves.copy(), COORD_REPORT, COORD_END)
    np.testing.assert_array_equal(got, taylor.poly_taylor_report_batch(leaves))
