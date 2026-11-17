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
  its own phase model: the time derivative of the phase it returns. Both the fixed and
  moving grids meet that (#30).
- Seeding and reporting are checked against `core.taylor`'s seed and gauge transform,
  through `transforms.taylor_to_cheby_full` and `cheby_to_taylor_full`.
- Branching is checked against the non-padded `psr_utils.branch_param` after the
  domain shift, and the branching-pattern generators against each other and against
  the closed-form frequency weighting.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numba import typed
from numpy import polynomial

from pyloki.core import chebyshev, taylor
from pyloki.utils import psr_utils, transforms
from pyloki.utils.misc import C_VAL
from tests._kernel_refs import (
    CELL_DV,
    COUNTS,
    LIMITS,
    NBINS,
    T_ADD,
    T_INIT,
    assert_phase,
    descendant_weighted_mean,
    expected_children,
    expected_phase,
    grid_cell,
    instantaneous_cell,
    paired_jit_variants,
)
from tests.jit_utils import jit_variants

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
            (T_ADD, 1.0),
            (0.0, SCALE),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _series(leaf[0, :4, 0], T_INIT, SCALE)
        freq = 100.0 * (1 - p.deriv(1)(T_ADD) / C_VAL)
        assert_phase(phase[0], expected_phase(T_ADD - T_INIT, p(T_ADD) / C_VAL, 100.0))
        assert idx[0, 0] == grid_cell(p.deriv(2)(T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == grid_cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_resolve_batch)
    )
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("x_init", [0.0, 1.3e5])
    def test_moving_grid(self, impl, t_cur: float, x_init: float) -> None:
        """Phase and accel cell in general; freq cell where ``v(t_init) = 0``."""
        state = _state_with_offset(5.0, 0.0, T_INIT, t_cur)
        state[3] += x_init
        leaf = _cheby_leaf(state, 100.0)
        idx, phase = impl(
            leaf,
            (T_ADD, 1.0),
            (t_cur, SCALE),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _series(leaf[0, :4, 0], t_cur, SCALE)
        delay = (p(T_ADD) - p(T_INIT)) / C_VAL
        freq = 100.0 * (1 - p.deriv(1)(T_ADD) / C_VAL)
        assert_phase(phase[0], expected_phase(T_ADD - T_INIT, delay, 100.0))
        assert idx[0, 0] == grid_cell(p.deriv(2)(T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == grid_cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(chebyshev.poly_chebyshev_fixed_resolve_batch),
    )
    @pytest.mark.parametrize("frac", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_fixed_grid_cell_follows_its_own_phase(self, impl, frac: float) -> None:
        leaf = _cheby_leaf([0.0, 5.0, frac * CELL_DV, 0.0], 100.0)

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf,
                (t, 1.0),
                (0.0, SCALE),
                (T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
            )
            return idx[0], ph[0]

        picked, expected = instantaneous_cell(resolve, T_ADD, 100.0)
        assert picked == expected

    @pytest.mark.parametrize(
        "impl", jit_variants(chebyshev.poly_chebyshev_resolve_batch)
    )
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("frac", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_moving_grid_cell_follows_its_own_phase(
        self,
        impl,
        t_cur: float,
        frac: float,
    ) -> None:
        """The moving-grid resolver picks the cell of its own phase's derivative (#30).

        The frequency must come from the absolute ``v(t_add)``, which the phase also
        carries, so a nonzero ``v(t_init)`` must not move the picked cell.
        """
        state = _state_with_offset(5.0, frac * CELL_DV, T_INIT, t_cur)
        leaf = _cheby_leaf(state, 100.0)

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf,
                (t, 1.0),
                (t_cur, SCALE),
                (T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
            )
            return idx[0], ph[0]

        picked, expected = instantaneous_cell(resolve, T_ADD, 100.0)
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
                assert_phase(
                    phase[i, s], expected_phase(t_seg - t_cur, p(t_seg) / C_VAL, f0)
                )
                freq = f0 * (1 - p.deriv(1)(t_seg) / C_VAL)
                assert idx[i, s, 0] == grid_cell(p.deriv(2)(t_seg), LIMITS[0], 16)
                assert idx[i, s, 1] == grid_cell(freq, LIMITS[1], 64)


class TestBranch:
    COORD_PREV, COORD_CUR = (18.0, 6.0), (20.0, 12.0)
    # Parent steps in units of the new step. The domain shift maps them to
    # [8 m0, 4 m1, 2 m2] (aggressive), and with "conservative" to
    # [8 m0, 8 m0 + 4 m1, 20.7 m0 + 2.7 m1 + 2 m2], so there a branching jerk forces
    # velocity to branch: there jerk is the parameter that passes through.
    MULTIPLIERS: ClassVar = {
        "aggressive": (0.5, 0.6, 0.1),
        "conservative": (0.02, 0.6, 0.6),
    }

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
        leaves[:, :3, 1] = dnew * np.array(self.MULTIPLIERS[strategy])
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
        # Both the branching and the pass-through path are exercised.
        assert (shift < 1).any()
        assert (shift >= 1).any()
        values, steps, want_origins = expected_children(
            shifted[:, :3, 0], shifted[:, :3, 1], dnew, shift
        )
        np.testing.assert_array_equal(origins, want_origins)
        assert 2 < len(out) == len(values) <= 50
        np.testing.assert_allclose(out[:, :3, 0], values, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[:, :3, 1], steps, rtol=1e-12)
        np.testing.assert_allclose(out[:, 3, 0], shifted[origins, 3, 0], rtol=1e-12)
        np.testing.assert_array_equal(out[:, -1], leaves[origins, -1])


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

    @pytest.mark.parametrize(
        ("impl", "approx_impl"),
        paired_jit_variants(
            chebyshev.generate_bp_poly_chebyshev,
            chebyshev.generate_bp_poly_chebyshev_approx,
        ),
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
        """Each level is the descendant-weighted mean branching factor."""
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
        np.testing.assert_allclose(mixed, descendant_weighted_mean(singles), rtol=1e-12)

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
