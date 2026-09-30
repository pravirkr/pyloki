"""Unit tests for `pyloki.core.taylor`, the Taylor-basis pruning kernels.

A leaf holds a Taylor expansion of the line-of-sight position about a reference time,
``leaf[:-1, 0] = [d_k, ..., d_1, d_0]`` with ``d_j`` the j-th derivative, its grid step
in ``leaf[:-1, 1]``, and the frequency ``f0`` at the tree's initial reference time in
``leaf[-1, 0]``. These kernels seed the leaves, branch them, move them between
reference times, resolve them onto the FFA grid of the segment being added, and turn
them into reported parameters.

The references are these.

- `resolve` against the leaf's own polynomial, evaluated with `numpy.polynomial` at the
  segment's time: the phase is ``frac(((t_add - t_init) - delay) f0) nbins``, and the
  grid cell comes from the cell arithmetic of the FFA grid;
- `branch` against the non-padded `psr_utils.branch_param` and ``itertools.product``;
- `generate_bp_poly_taylor_approx`, which follows one leaf, against
  `generate_bp_poly_taylor`, which averages over frequencies: for one frequency they
  must agree.

On the moving grid, `poly_taylor_resolve_batch` takes the added segment's frequency
from the velocity change since ``t_init``, where the fixed-grid and ascend resolvers use
the absolute velocity. A test independent of either convention decides between them:
the frequency cell a resolver picks must contain the instantaneous frequency of its own
phase model at ``t_add``, i.e. the time derivative of the phase it returns. The fixed
grid meets that; the moving grid misses by the leaf's velocity at ``t_init``, so it is
pinned as a defect (`TestResolve.test_moving_grid_cell_follows_its_own_phase`).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numba import typed
from numpy import polynomial

from pyloki.core import taylor
from pyloki.utils import psr_utils, transforms
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.
MOVING_GRID_ISSUE = "#30: the moving-grid resolver can select the adjacent FFA frequency cell"

NBINS = 64
LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])  # [accel, freq]
COUNTS = np.array([16, 64])


def _leaf(jerk: float, accel: float, vel: float, pos: float, f0: float) -> np.ndarray:
    """Return a one-leaf batch with rows [d3, d2, d1, d0, (f0, flag)]."""
    leaf = np.zeros((1, 5, 2))
    leaf[0, :4, 0] = [jerk, accel, vel, pos]
    leaf[0, :4, 1] = [1e-6, 0.05, 10.0, 0.0]
    leaf[0, -1, 0] = f0
    return leaf


def _shifted(coeffs: np.ndarray, t_ref: float) -> polynomial.Polynomial:
    """Return p(t) = sum_j coeffs[j] (t - t_ref)**j, in absolute time."""
    base = polynomial.Polynomial([-t_ref, 1.0])
    out = polynomial.Polynomial([0.0])
    for j, c in enumerate(coeffs):
        out = out + c * base**j
    return out


def _cell(value: float, limits: np.ndarray, count: int) -> int:
    lo, hi = limits
    return int(np.clip(np.floor(count * (value - lo) / (hi - lo)), 0, count - 1))


def _phase(dt: float, delay: float, f0: float) -> float:
    return ((dt - delay) * f0) % 1 * NBINS


def _assert_phase(got: float, expected: float) -> None:
    """Compare phases on the circle of NBINS bins (63.999... and 0 are equal)."""
    diff = (got - expected + NBINS / 2) % NBINS - NBINS / 2
    assert abs(diff) < 1e-6, (got, expected)


class TestSeed:
    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_seed))
    def test_layout(self, impl) -> None:
        accels = np.array([-1.0, 2.0])
        freqs = np.array([99.5, 100.5, 101.0])
        param_arr = typed.List([np.array([0.0]), accels, freqs])
        dparams = np.array([1e-4, 0.05, 0.01])
        leaves = impl(param_arr, dparams, 3, (5.0, 1.0))
        assert leaves.shape == (6, 5, 2)
        combos = list(itertools.product([0.0], accels, freqs))
        for leaf, (jerk, accel, freq) in zip(leaves, combos, strict=True):
            np.testing.assert_array_equal(leaf[:2, 0], [jerk, accel])
            np.testing.assert_array_equal(leaf[:2, 1], dparams[:2])
            # Frequency uncertainty becomes a velocity step: dv = (c / f0) df.
            assert leaf[2, 0] == 0.0
            assert leaf[2, 1] == pytest.approx(dparams[-1] * C_VAL / freq)
            assert leaf[3, 0] == 0.0
            assert leaf[-1, 0] == freq
            assert leaf[-1, 1] == 0.0


class TestResolve:
    # T_ADD - T_INIT is not a whole number of cycles at f0: a wrong proper time shows.
    T_INIT, T_ADD = 10.0, 30.0037

    @pytest.mark.parametrize(
        "impl", jit_variants(taylor.poly_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("vel", [0.0, -4.6e4, 3e4])
    def test_fixed_grid(self, impl, vel: float) -> None:
        leaf = _leaf(2e-3, 5.0, vel, 0.0, 100.0)
        idx, phase = impl(
            leaf,
            (self.T_ADD, 1.0),
            (self.T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], self.T_INIT)
        accel = p.deriv(2)(self.T_ADD)
        freq = 100.0 * (1 - p.deriv(1)(self.T_ADD) / C_VAL)
        delay = p(self.T_ADD) / C_VAL
        _assert_phase(phase[0], _phase(self.T_ADD - self.T_INIT, delay, 100.0))
        assert idx[0, 0] == _cell(accel, LIMITS[0], 16)
        assert idx[0, 1] == _cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    # Non-round times, so that no interval is a whole number of cycles at f0.
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("x_init", [0.0, 1.3e5])
    def test_moving_grid(self, impl, t_cur: float, x_init: float) -> None:
        """Phase and accel cell in general; freq cell where ``v(t_init) = 0``.

        The leaf is built so that its velocity vanishes at ``t_init``, which is where
        the moving- and fixed-grid frequency conventions agree (module docstring).
        """
        jerk, accel_init = 2e-3, 5.0
        dt = t_cur - self.T_INIT
        # Taylor state at t_cur of a trajectory with v(t_init) = 0, x(t_init) = x_init.
        vel_cur = accel_init * dt + jerk * dt**2 / 2
        pos_cur = x_init + accel_init * dt**2 / 2 + jerk * dt**3 / 6
        accel_cur = accel_init + jerk * dt
        leaf = _leaf(jerk, accel_cur, vel_cur, pos_cur, 100.0)
        idx, phase = impl(
            leaf,
            (self.T_ADD, 1.0),
            (t_cur, 1.0),
            (self.T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], t_cur)
        assert p.deriv(1)(self.T_INIT) == pytest.approx(0.0, abs=1e-9)
        delay = (p(self.T_ADD) - p(self.T_INIT)) / C_VAL
        freq = 100.0 * (1 - p.deriv(1)(self.T_ADD) / C_VAL)
        _assert_phase(phase[0], _phase(self.T_ADD - self.T_INIT, delay, 100.0))
        assert idx[0, 0] == _cell(p.deriv(2)(self.T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == _cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    @pytest.mark.parametrize("vel", [-4.6e4, 3e4])
    def test_moving_grid_phase_carries_the_velocity(self, impl, vel: float) -> None:
        """With ``v(t_init) != 0`` the phase still includes it, through the delay."""
        leaf = _leaf(0.0, 5.0, vel, 0.0, 100.0)
        _, phase = impl(
            leaf,
            (self.T_ADD, 1.0),
            (self.T_INIT, 1.0),
            (self.T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], self.T_INIT)
        delay = (p(self.T_ADD) - p(self.T_INIT)) / C_VAL
        _assert_phase(phase[0], _phase(self.T_ADD - self.T_INIT, delay, 100.0))

    @staticmethod
    def _instantaneous_cell(resolve, leaf: np.ndarray, t_add: float) -> tuple[int, int]:
        """Return (picked freq cell, cell of the derivative of the returned phase)."""
        eps, f0 = 1e-3, leaf[0, -1, 0]
        hi = resolve(leaf, t_add + eps)[1] / NBINS
        lo = resolve(leaf, t_add - eps)[1] / NBINS
        dcyc = (hi - lo + 0.5) % 1 - 0.5
        f_inst = (dcyc + round(f0 * 2 * eps - dcyc)) / (2 * eps)
        return resolve(leaf, t_add)[0][1], _cell(f_inst, LIMITS[1], 64)

    # A leaf's velocity at t_init is its frequency offset: dv = (c / f0) df. One cell of
    # this grid (2/64 Hz at 100 Hz) is about 9.4e4 m/s.
    CELL_DV = (2 / 64) * C_VAL / 100.0
    OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5)

    @pytest.mark.parametrize(
        "impl", jit_variants(taylor.poly_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("frac", OFFSETS)
    def test_fixed_grid_cell_follows_its_own_phase(self, impl, frac: float) -> None:
        leaf = _leaf(0.0, 5.0, frac * self.CELL_DV, 0.0, 100.0)

        def resolve(lf: np.ndarray, t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(lf, (t, 1.0), (self.T_INIT, 1.0), COUNTS, LIMITS, NBINS)
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve, leaf, self.T_ADD)
        assert picked == expected

    @pytest.mark.xfail(strict=True, reason=MOVING_GRID_ISSUE, raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    @pytest.mark.parametrize("frac", [-0.5, -0.25])
    def test_moving_grid_cell_follows_its_own_phase(self, impl, frac: float) -> None:
        """Pin the moving grid dropping the velocity at t_init from the frequency.

        It computes ``f0 (1 - (v(t_add) - v(t_init)) / c)``. Its phase includes
        ``v(t_init)`` (through the delay), so the frequency it resolves to is not
        that phase's own rate: for ``v(t_init) = -0.5`` cell it picks cell 31, and
        the derivative of its phase lies in cell 32. The fixed grid picks 32.
        """
        leaf = _leaf(0.0, 5.0, frac * self.CELL_DV, 0.0, 100.0)

        def resolve(lf: np.ndarray, t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                lf,
                (t, 1.0),
                (self.T_INIT, 1.0),
                (self.T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
            )
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve, leaf, self.T_ADD)
        assert picked == expected

    @pytest.mark.parametrize(
        "impl", jit_variants(taylor.poly_taylor_ascend_resolve_batch)
    )
    def test_ascend(self, impl) -> None:
        leaf = np.concatenate(
            # Velocities of a few frequency cells (one cell is about 9.4e4 m/s here),
            # so a wrong velocity row lands in a different cell.
            [_leaf(2e-3, 5.0, 2.3e5, 0.0, 100.0), _leaf(0.0, -3.0, -3.1e5, 0.0, 100.3)],
        )
        t_cur = 12.0
        seg_times = np.array([[4.137, 1.0], [12.0, 1.0], [26.0131, 1.0]])
        idx, phase = impl(leaf, seg_times, (t_cur, 1.0), COUNTS, LIMITS, NBINS)
        assert idx.shape == (2, 3, 2)
        for i in range(2):
            p = _shifted(leaf[i, :4, 0][::-1] / [1, 1, 2, 6], t_cur)
            f0 = leaf[i, -1, 0]
            for s, (t_seg, _) in enumerate(seg_times):
                delay = p(t_seg) / C_VAL
                _assert_phase(phase[i, s], _phase(t_seg - t_cur, delay, f0))
                freq = f0 * (1 - p.deriv(1)(t_seg) / C_VAL)
                assert idx[i, s, 0] == _cell(p.deriv(2)(t_seg), LIMITS[0], 16)
                assert idx[i, s, 1] == _cell(freq, LIMITS[1], 64)


class TestBranch:
    COORD = (20.0, 12.0)

    def _leaves(self) -> np.ndarray:
        leaves = np.concatenate(
            [_leaf(0.0, 1.0, 0.0, 0.0, 100.0), _leaf(1e-4, -2.0, 50.0, 3.0, 100.7)],
        )
        # Parent steps a few times the new step, so that every parameter branches
        # except the last, which is kept just under it so it does not.
        dnew = psr_utils.poly_taylor_step_d_vec(
            3,
            self.COORD[1],
            NBINS,
            1.0,
            leaves[:, -1, 0],
            t_ref=0,
        )
        leaves[:, :3, 1] = dnew * np.array([2.5, 3.2, 0.9])
        return leaves

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_branch_batch))
    def test_children_tile_each_parent(self, impl) -> None:
        leaves = self._leaves()
        out, origins = impl(leaves, self.COORD, NBINS, 1.0, 3, 64)
        _, span = self.COORD
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_taylor_step_d_vec(3, span, NBINS, 1.0, f0, t_ref=0)
        shift = psr_utils.poly_taylor_shift_d_vec(
            leaves[:, :3, 1],
            dnew,
            span,
            NBINS,
            f0,
            t_ref=0,
        )
        expected_rows, expected_origins = [], []
        for i in range(2):
            axes, steps = [], []
            for j in range(3):
                if shift[i, j] < 1.0 - 1e-12:
                    axes.append([leaves[i, j, 0]])
                    steps.append(leaves[i, j, 1])
                else:
                    vals, dact = psr_utils.branch_param(
                        leaves[i, j, 0],
                        leaves[i, j, 1],
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
            np.testing.assert_allclose(leaf[:3, 0], combo, rtol=1e-12)
            np.testing.assert_allclose(leaf[:3, 1], steps, rtol=1e-12)
            assert leaf[3, 0] == leaves[i, 3, 0]
            np.testing.assert_array_equal(leaf[-1], leaves[i, -1])


class TestTransformAndReport:
    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_transform_batch))
    @pytest.mark.parametrize("strategy", ["aggressive", "quadrature", "conservative"])
    def test_transform_is_shift_taylor_full(self, impl, strategy: str) -> None:
        leaves = np.concatenate(
            [_leaf(2e-3, 5.0, 1e4, 3.0, 100.0), _leaf(0.0, -3.0, -2e4, 0.0, 100.3)],
        )
        leaves[:, -1, 1] = [0.0, 1.0]
        out = impl(leaves, (25.0, 1.0), (10.0, 1.0), strategy)
        np.testing.assert_allclose(
            out[:, :-1],
            transforms.shift_taylor_full(leaves[:, :-1], 15.0, strategy),
            rtol=1e-12,
        )
        np.testing.assert_array_equal(out[:, -1], leaves[:, -1])

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_report_batch))
    def test_report_gauge_transform(self, impl) -> None:
        """Divide by ``s = 1 - v/c``, turn ``v`` into ``f0 s``, propagate ``dv``."""
        leaves = np.concatenate(
            [_leaf(2e-3, 5.0, 1e4, 3.0, 100.0), _leaf(0.0, -3.0, -2e4, 0.0, 100.3)],
        )
        out = impl(leaves)
        for leaf, rep in zip(leaves, out, strict=True):
            vel, dvel, f0 = leaf[2, 0], leaf[2, 1], leaf[-1, 0]
            s = 1 - vel / C_VAL
            np.testing.assert_allclose(rep[:2, 0], leaf[:2, 0] / s, rtol=1e-12)
            expected_err = np.hypot(
                leaf[:2, 1] / s, leaf[:2, 0] / (C_VAL * s**2) * dvel
            )
            np.testing.assert_allclose(rep[:2, 1], expected_err, rtol=1e-12)
            assert rep[2, 0] == pytest.approx(f0 * s, rel=1e-14)
            assert rep[2, 1] == pytest.approx(f0 * dvel / C_VAL, rel=1e-14)
            np.testing.assert_array_equal(rep[3:], leaf[3:])


class TestBranchingPattern:
    PARAM_ARR = typed.List([np.array([0.0]), np.array([0.0]), np.array([100.0])])
    # A config that branches at several levels, and differently on the two grids.
    DPARAMS = np.array([2e-4, 2.0, 0.05])
    ARGS = (10.0, 16, NBINS, 1.0, 8)
    # Large enough accel/jerk steps that branching depends on f0, so that the
    # frequency weighting of `generate_bp_poly_taylor` is observable.
    DPARAMS_F0 = np.array([1e-2, 200.0, 0.05])

    @pytest.mark.parametrize("impl", jit_variants(taylor.generate_bp_poly_taylor))
    @pytest.mark.parametrize(
        "approx_impl",
        jit_variants(taylor.generate_bp_poly_taylor_approx),
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
            use_cheby_coarsening=False,
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

    @pytest.mark.parametrize("impl", jit_variants(taylor.generate_bp_poly_taylor))
    def test_exact_averages_over_frequencies(self, impl) -> None:
        """Each level is the mean branching factor, weighted by descendants.

        With single-frequency patterns ``n_i``, level ``l`` of the mix is
        ``sum_i W_i n_i[l] / sum_i W_i`` with ``W_i = prod_{m < l} n_i[m]``.
        """
        freqs = np.array([20.0, 100.0, 400.0])
        mixed = impl(
            typed.List([np.array([0.0]), np.array([0.0]), freqs]),
            self.DPARAMS_F0,
            *self.ARGS,
            use_moving_grid=True,
            tiling_strategy="aggressive",
            use_cheby_coarsening=False,
        )
        singles = np.array(
            [
                impl(
                    typed.List([np.array([0.0]), np.array([0.0]), np.array([f])]),
                    self.DPARAMS_F0,
                    *self.ARGS,
                    use_moving_grid=True,
                    tiling_strategy="aggressive",
                    use_cheby_coarsening=False,
                )
                for f in freqs
            ]
        )
        weights = np.cumprod(
            np.hstack([np.ones((len(freqs), 1)), singles[:, :-1]]),
            axis=1,
        )
        expected = (weights * singles).sum(axis=0) / weights.sum(axis=0)
        np.testing.assert_allclose(mixed, expected, rtol=1e-12)
        assert not np.allclose(singles[0], singles[-1])

    @pytest.mark.parametrize(
        "impl",
        jit_variants(taylor.generate_bp_poly_taylor_approx),
    )
    def test_approx_flags_a_pattern_that_reaches_branch_max(self, impl) -> None:
        """A level with exactly ``branch_max`` leaves may have been truncated."""
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
