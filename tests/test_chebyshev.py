"""Unit tests for `pyloki.core.chebyshev`, the Chebyshev-basis pruning kernels.

A leaf holds the line-of-sight position as a Chebyshev series on a domain
``(t_c, scale)``, ``leaf[:-1, 0] = [alpha_k, ..., alpha_0]``, with its grid steps in
``leaf[:-1, 1]`` and the frequency ``f0`` at the tree's initial reference time in
``leaf[-1, 0]``. The kernels mirror `core.taylor`: seed, branch, resolve onto the FFA
grid of the segment being added, move between domains, and report.

The references are these.

- The resolvers are checked against the leaf's own series, evaluated with
  `numpy.polynomial.Chebyshev` on its domain, whose derivatives account for the
  domain's scale. The phase is ``frac(((t_add - t_init) - delay) f0) nbins``, compared
  on the circle.
- The frequency cell each resolver picks must contain the instantaneous frequency of
  its own phase model: the time derivative of the phase it returns. The fixed grid
  meets that. The moving grid misses by the leaf's velocity at ``t_init``, the same
  defect as in `core.taylor`, and is pinned.
- Seeding and reporting are checked against `core.taylor`'s seed and gauge transform,
  through `transforms.taylor_to_cheby_full` and `cheby_to_taylor_full`.
- Branching is checked against the non-padded `psr_utils.branch_param` after the
  domain shift, and the branching-pattern generators against each other and against
  the closed-form frequency weighting.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numba import typed
from numpy import polynomial

from pyloki.core import chebyshev, taylor
from pyloki.utils import psr_utils, transforms
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.
MOVING_GRID_ISSUE = "#30: the moving-grid resolver can select the adjacent FFA frequency cell"

NBINS = 64
LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])  # [accel, freq]
COUNTS = np.array([16, 64])
SCALE = 5.0


def _cheby_leaf(d_vec: np.ndarray, f0: float, scale: float = SCALE) -> np.ndarray:
    """Return a one-leaf batch from a Taylor state [jerk, accel, vel, pos] at t_c."""
    leaf = np.zeros((1, 5, 2))
    leaf[0, :4, 0] = transforms.taylor_to_cheby(np.asarray(d_vec, dtype=float), scale)
    leaf[0, :4, 1] = [1e-6, 0.05, 10.0, 0.0]
    leaf[0, -1, 0] = f0
    return leaf


def _series(leaf_row: np.ndarray, t_c: float, scale: float) -> polynomial.Chebyshev:
    return polynomial.Chebyshev(leaf_row[::-1], domain=[t_c - scale, t_c + scale])


def _cell(value: float, limits: np.ndarray, count: int) -> int:
    lo, hi = limits
    return int(np.clip(np.floor(count * (value - lo) / (hi - lo)), 0, count - 1))


def _phase(dt: float, delay: float, f0: float) -> float:
    return ((dt - delay) * f0) % 1 * NBINS


def _assert_phase(got: float, expected: float) -> None:
    """Compare phases on the circle of NBINS bins (63.999... and 0 are equal)."""
    diff = (got - expected + NBINS / 2) % NBINS - NBINS / 2
    assert abs(diff) < 1e-6, (got, expected)


def _state_with_offset(accel: float, d1: float, t_init: float, t_c: float) -> list:
    """Taylor state at t_c of x(t) = d1 (t - t_init) + accel (t - t_init)^2 / 2."""
    dt = t_c - t_init
    return [0.0, accel, d1 + accel * dt, d1 * dt + accel * dt**2 / 2]


class TestSeed:
    @pytest.mark.parametrize("impl", jit_variants(chebyshev.poly_chebyshev_seed))
    def test_is_the_taylor_seed_in_chebyshev_form(self, impl) -> None:
        param_arr = typed.List(
            [np.array([0.0]), np.array([-1.0, 2.0]), np.array([99.5, 101.0])]
        )
        dparams = np.array([1e-4, 0.05, 0.01])
        coord = (5.0, 2.5)
        cheb = impl(param_arr, dparams, 3, coord)
        tay = taylor.poly_taylor_seed(param_arr, dparams, 3, coord)
        np.testing.assert_allclose(
            cheb[:, :-1],
            transforms.taylor_to_cheby_full(tay[:, :-1], coord[1]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(cheb[:, -1], tay[:, -1])


class TestResolve:
    # T_ADD - T_INIT is not a whole number of cycles at f0: a wrong proper time shows.
    T_INIT, T_ADD = 10.0, 30.0037
    CELL_DV = (2 / 64) * C_VAL / 100.0  # one frequency cell of velocity at f0 = 100

    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.poly_chebyshev_fixed_resolve_batch),
    )
    @pytest.mark.parametrize("d1", [0.0, -4.6e4, 3e4])
    def test_fixed_grid(self, impl, d1: float) -> None:
        """The leaf is a series about ``t_init`` on the fixed grid's scale."""
        leaf = _cheby_leaf([2e-3, 5.0, d1, 0.0], 100.0)
        idx, phase = impl(
            leaf,
            (self.T_ADD, 1.0),
            (0.0, SCALE),
            (self.T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _series(leaf[0, :4, 0], self.T_INIT, SCALE)
        freq = 100.0 * (1 - p.deriv(1)(self.T_ADD) / C_VAL)
        _assert_phase(
            phase[0], _phase(self.T_ADD - self.T_INIT, p(self.T_ADD) / C_VAL, 100.0)
        )
        assert idx[0, 0] == _cell(p.deriv(2)(self.T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == _cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_resolve_batch)
    )
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("x_init", [0.0, 1.3e5])
    def test_moving_grid(self, impl, t_cur: float, x_init: float) -> None:
        """Phase and accel cell in general; freq cell where ``v(t_init) = 0``."""
        state = _state_with_offset(5.0, 0.0, self.T_INIT, t_cur)
        state[3] += x_init
        leaf = _cheby_leaf(state, 100.0)
        idx, phase = impl(
            leaf,
            (self.T_ADD, 1.0),
            (t_cur, SCALE),
            (self.T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _series(leaf[0, :4, 0], t_cur, SCALE)
        delay = (p(self.T_ADD) - p(self.T_INIT)) / C_VAL
        freq = 100.0 * (1 - p.deriv(1)(self.T_ADD) / C_VAL)
        _assert_phase(phase[0], _phase(self.T_ADD - self.T_INIT, delay, 100.0))
        assert idx[0, 0] == _cell(p.deriv(2)(self.T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == _cell(freq, LIMITS[1], 64)

    @staticmethod
    def _instantaneous_cell(resolve, t_add: float, f0: float) -> tuple[int, int]:
        """Return (picked freq cell, cell of the derivative of the returned phase)."""
        eps = 1e-3
        hi = resolve(t_add + eps)[1] / NBINS
        lo = resolve(t_add - eps)[1] / NBINS
        dcyc = (hi - lo + 0.5) % 1 - 0.5
        f_inst = (dcyc + round(f0 * 2 * eps - dcyc)) / (2 * eps)
        return resolve(t_add)[0][1], _cell(f_inst, LIMITS[1], 64)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.poly_chebyshev_fixed_resolve_batch),
    )
    @pytest.mark.parametrize("frac", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_fixed_grid_cell_follows_its_own_phase(self, impl, frac: float) -> None:
        leaf = _cheby_leaf([0.0, 5.0, frac * self.CELL_DV, 0.0], 100.0)

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf,
                (t, 1.0),
                (0.0, SCALE),
                (self.T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
            )
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve, self.T_ADD, 100.0)
        assert picked == expected

    @pytest.mark.xfail(strict=True, reason=MOVING_GRID_ISSUE, raises=AssertionError)
    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_resolve_batch)
    )
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("frac", [-0.5, -0.25])
    def test_moving_grid_cell_follows_its_own_phase(
        self,
        impl,
        t_cur: float,
        frac: float,
    ) -> None:
        """Pin the moving grid dropping the velocity at t_init from the frequency.

        As in `core.taylor`: it computes ``f0 (1 - (v(t_add) - v(t_init)) / c)`` while
        its phase includes ``v(t_init)``, so at ``v(t_init) = -0.5`` cell it picks cell
        31 and the derivative of its own phase lies in cell 32.
        """
        state = _state_with_offset(5.0, frac * self.CELL_DV, self.T_INIT, t_cur)
        leaf = _cheby_leaf(state, 100.0)

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf,
                (t, 1.0),
                (t_cur, SCALE),
                (self.T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
            )
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve, self.T_ADD, 100.0)
        assert picked == expected

    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.poly_chebyshev_ascend_resolve_batch),
    )
    def test_ascend(self, impl) -> None:
        t_cur = 12.0
        leaves = np.concatenate(
            [
                # Velocities of a few frequency cells, so a wrong row changes the cell.
                _cheby_leaf([2e-3, 5.0, 2.3e5, 0.0], 100.0),
                _cheby_leaf([0.0, -3.0, -3.1e5, 0.0], 100.3),
            ],
        )
        seg_times = np.array([[4.137, 1.0], [12.0, 1.0], [26.3719, 1.0]])
        idx, phase = impl(leaves, seg_times, (t_cur, SCALE), COUNTS, LIMITS, NBINS)
        assert idx.shape == (2, 3, 2)
        for i in range(2):
            p = _series(leaves[i, :4, 0], t_cur, SCALE)
            f0 = leaves[i, -1, 0]
            for s, (t_seg, _) in enumerate(seg_times):
                _assert_phase(phase[i, s], _phase(t_seg - t_cur, p(t_seg) / C_VAL, f0))
                freq = f0 * (1 - p.deriv(1)(t_seg) / C_VAL)
                assert idx[i, s, 0] == _cell(p.deriv(2)(t_seg), LIMITS[0], 16)
                assert idx[i, s, 1] == _cell(freq, LIMITS[1], 64)


class TestBranch:
    COORD_PREV, COORD_CUR = (18.0, 6.0), (20.0, 12.0)

    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_branch_batch)
    )
    @pytest.mark.parametrize("strategy", ["aggressive", "conservative"])
    def test_children_tile_each_shifted_parent(self, impl, strategy: str) -> None:
        leaves = np.concatenate(
            [
                _cheby_leaf([0.0, 1.0, 0.0, 0.0], 100.0, scale=6.0),
                _cheby_leaf([1e-4, -2.0, 50.0, 3.0], 100.7, scale=6.0),
            ],
        )
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_cheb_step_vec(3, NBINS, 1.0, f0)
        # Parent steps a few times the new step once shifted, except the last.
        leaves[:, :3, 1] = dnew * np.array([2.5, 3.2, 0.1])
        out, origins = impl(
            leaves, self.COORD_CUR, self.COORD_PREV, NBINS, 1.0, 3, 64, strategy
        )
        shifted = transforms.shift_cheby_full(
            leaves[:, :-1],
            self.COORD_CUR,
            self.COORD_PREV,
            strategy,
        )
        shift = psr_utils.poly_cheb_shift_vec(shifted[:, :-1, 1], dnew, NBINS, f0)
        expected_rows, expected_origins = [], []
        for i in range(2):
            axes, steps = [], []
            for j in range(3):
                if shift[i, j] < 1.0 - 1e-12:
                    axes.append([shifted[i, j, 0]])
                    steps.append(shifted[i, j, 1])
                else:
                    vals, dact = psr_utils.branch_param(
                        shifted[i, j, 0],
                        shifted[i, j, 1],
                        dnew[i, j],
                    )
                    axes.append(list(vals))
                    steps.append(dact)
            for combo in itertools.product(*axes):
                expected_rows.append((combo, steps))
                expected_origins.append(i)
        np.testing.assert_array_equal(origins, expected_origins)
        assert len(out) == len(expected_rows) > 2
        for leaf, (combo, steps), i in zip(out, expected_rows, origins, strict=True):
            np.testing.assert_allclose(leaf[:3, 0], combo, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(leaf[:3, 1], steps, rtol=1e-12)
            assert leaf[3, 0] == pytest.approx(shifted[i, 3, 0], rel=1e-12)
            np.testing.assert_array_equal(leaf[-1], leaves[i, -1])


class TestTransformAndReport:
    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.poly_chebyshev_transform_batch),
    )
    @pytest.mark.parametrize("strategy", ["aggressive", "quadrature", "conservative"])
    def test_transform_is_shift_cheby_full(self, impl, strategy: str) -> None:
        leaves = np.concatenate(
            [
                _cheby_leaf([2e-3, 5.0, 1e4, 3.0], 100.0),
                _cheby_leaf([0.0, -3.0, -2e4, 0.0], 100.3),
            ],
        )
        leaves[:, -1, 1] = [0.0, 1.0]
        out = impl(leaves, (25.0, 7.0), (10.0, 5.0), strategy)
        np.testing.assert_allclose(
            out[:, :-1],
            transforms.shift_cheby_full(
                leaves[:, :-1], (25.0, 7.0), (10.0, 5.0), strategy
            ),
            rtol=1e-12,
        )
        np.testing.assert_array_equal(out[:, -1], leaves[:, -1])

    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_report_batch)
    )
    def test_report_is_taylor_report_of_the_taylor_form(self, impl) -> None:
        """Convert with the report scale, then the same gauge transform as Taylor."""
        leaves = np.concatenate(
            [
                _cheby_leaf([2e-3, 5.0, 1e4, 3.0], 100.0),
                _cheby_leaf([0.0, -3.0, -2e4, 0.0], 100.3),
            ],
        )
        out = impl(leaves, (99.0, SCALE))
        as_taylor = leaves.copy()
        as_taylor[:, :-1] = transforms.cheby_to_taylor_full(leaves[:, :-1], SCALE)
        np.testing.assert_allclose(
            out,
            taylor.poly_taylor_report_batch(as_taylor),
            rtol=1e-10,
            atol=1e-12,
        )


class TestBranchingPattern:
    PARAM_ARR = typed.List([np.array([0.0]), np.array([0.0]), np.array([100.0])])
    DPARAMS = np.array([2e-4, 2.0, 0.05])
    ARGS = (10.0, 16, NBINS, 1.0, 8)
    DPARAMS_F0 = np.array([1e-2, 200.0, 0.05])

    @pytest.mark.parametrize("impl", jit_variants(chebyshev.generate_bp_poly_chebyshev))
    @pytest.mark.parametrize(
        "approx_impl",
        jit_variants(chebyshev.generate_bp_poly_chebyshev_approx),
    )
    @pytest.mark.parametrize("moving", [True, False])
    @pytest.mark.parametrize("strategy", ["aggressive", "conservative"])
    def test_approx_matches_exact_for_one_frequency(
        self,
        impl,
        approx_impl,
        moving: bool,
        strategy: str,
    ) -> None:
        exact = impl(
            self.PARAM_ARR,
            self.DPARAMS,
            *self.ARGS,
            use_moving_grid=moving,
            tiling_strategy=strategy,
        )
        approx = approx_impl(
            self.PARAM_ARR,
            self.DPARAMS,
            *self.ARGS,
            use_moving_grid=moving,
            tiling_strategy=strategy,
            itree=0,
            branch_max=4096,
        )
        np.testing.assert_array_equal(exact, approx)
        assert exact.shape == (15,)
        assert (exact >= 1).all()
        assert exact.max() > 1

    @pytest.mark.parametrize("impl", jit_variants(chebyshev.generate_bp_poly_chebyshev))
    def test_exact_averages_over_frequencies(self, impl) -> None:
        """Each level is the descendant-weighted mean branching factor.

        Level ``l`` is ``sum_i W_i n_i[l] / sum_i W_i``, ``W_i = prod_{m<l} n_i[m]``.
        """
        freqs = np.array([20.0, 100.0, 400.0])
        kwargs = {"use_moving_grid": True, "tiling_strategy": "aggressive"}
        mixed = impl(
            typed.List([np.array([0.0]), np.array([0.0]), freqs]),
            self.DPARAMS_F0,
            *self.ARGS,
            **kwargs,
        )
        singles = np.array(
            [
                impl(
                    typed.List([np.array([0.0]), np.array([0.0]), np.array([f])]),
                    self.DPARAMS_F0,
                    *self.ARGS,
                    **kwargs,
                )
                for f in freqs
            ]
        )
        assert not np.allclose(singles[0], singles[-1])
        weights = np.cumprod(
            np.hstack([np.ones((len(freqs), 1)), singles[:, :-1]]),
            axis=1,
        )
        expected = (weights * singles).sum(axis=0) / weights.sum(axis=0)
        np.testing.assert_allclose(mixed, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.generate_bp_poly_chebyshev_approx),
    )
    def test_approx_flags_a_pattern_that_reaches_branch_max(self, impl) -> None:
        kwargs = {"use_moving_grid": True, "tiling_strategy": "aggressive", "itree": 0}
        full = impl(self.PARAM_ARR, self.DPARAMS, *self.ARGS, **kwargs, branch_max=4096)
        with pytest.raises(ValueError, match="truncated due to branch_max"):
            impl(
                self.PARAM_ARR,
                self.DPARAMS,
                *self.ARGS,
                **kwargs,
                branch_max=int(full.max()),
            )
