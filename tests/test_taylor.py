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
from the absolute velocity at ``t_add``, as the fixed-grid and ascend resolvers do. The
frequency cell a resolver picks must contain the instantaneous frequency of its own
phase model at ``t_add``, i.e. the time derivative of the phase it returns
(`TestResolve.test_moving_grid_cell_follows_its_own_phase`, #30).
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
    @pytest.mark.parametrize(
        "impl", jit_variants(taylor.poly_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("vel", [0.0, -4.6e4, 3e4])
    def test_fixed_grid(self, impl, vel: float) -> None:
        leaf = _leaf(2e-3, 5.0, vel, 0.0, 100.0)
        idx, phase = impl(
            leaf,
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], T_INIT)
        accel = p.deriv(2)(T_ADD)
        freq = 100.0 * (1 - p.deriv(1)(T_ADD) / C_VAL)
        delay = p(T_ADD) / C_VAL
        assert_phase(phase[0], expected_phase(T_ADD - T_INIT, delay, 100.0))
        assert idx[0, 0] == grid_cell(accel, LIMITS[0], 16)
        assert idx[0, 1] == grid_cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    # Non-round times, so that no interval is a whole number of cycles at f0.
    @pytest.mark.parametrize("t_cur", [10.0, 18.3719])
    @pytest.mark.parametrize("x_init", [0.0, 1.3e5])
    def test_moving_grid(self, impl, t_cur: float, x_init: float) -> None:
        """Phase and accel cell in general; freq cell where ``v(t_init) = 0``.

        The leaf is built so that its velocity vanishes at ``t_init``, which makes the
        frequency cell independent of the velocity's reference time.
        """
        jerk, accel_init = 2e-3, 5.0
        dt = t_cur - T_INIT
        # Taylor state at t_cur of a trajectory with v(t_init) = 0, x(t_init) = x_init.
        vel_cur = accel_init * dt + jerk * dt**2 / 2
        pos_cur = x_init + accel_init * dt**2 / 2 + jerk * dt**3 / 6
        accel_cur = accel_init + jerk * dt
        leaf = _leaf(jerk, accel_cur, vel_cur, pos_cur, 100.0)
        idx, phase = impl(
            leaf,
            (T_ADD, 1.0),
            (t_cur, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], t_cur)
        assert p.deriv(1)(T_INIT) == pytest.approx(0.0, abs=1e-9)
        delay = (p(T_ADD) - p(T_INIT)) / C_VAL
        freq = 100.0 * (1 - p.deriv(1)(T_ADD) / C_VAL)
        assert_phase(phase[0], expected_phase(T_ADD - T_INIT, delay, 100.0))
        assert idx[0, 0] == grid_cell(p.deriv(2)(T_ADD), LIMITS[0], 16)
        assert idx[0, 1] == grid_cell(freq, LIMITS[1], 64)

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    @pytest.mark.parametrize("vel", [-4.6e4, 3e4])
    def test_moving_grid_phase_carries_the_velocity(self, impl, vel: float) -> None:
        """With ``v(t_init) != 0`` the phase still includes it, through the delay."""
        leaf = _leaf(0.0, 5.0, vel, 0.0, 100.0)
        _, phase = impl(
            leaf,
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
        )
        p = _shifted(leaf[0, :4, 0][::-1] / [1, 1, 2, 6], T_INIT)
        delay = (p(T_ADD) - p(T_INIT)) / C_VAL
        assert_phase(phase[0], expected_phase(T_ADD - T_INIT, delay, 100.0))

    OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5)

    @pytest.mark.parametrize(
        "impl", jit_variants(taylor.poly_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("frac", OFFSETS)
    def test_fixed_grid_cell_follows_its_own_phase(self, impl, frac: float) -> None:
        leaf = _leaf(0.0, 5.0, frac * CELL_DV, 0.0, 100.0)

        def resolve(lf: np.ndarray, t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(lf, (t, 1.0), (T_INIT, 1.0), COUNTS, LIMITS, NBINS)
            return idx[0], ph[0]

        picked, expected = instantaneous_cell(lambda t: resolve(leaf, t), T_ADD, 100.0)
        assert picked == expected

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_resolve_batch))
    @pytest.mark.parametrize("t_cur", [T_INIT, 18.3719])
    @pytest.mark.parametrize("frac", OFFSETS)
    def test_moving_grid_cell_follows_its_own_phase(
        self, impl, t_cur: float, frac: float
    ) -> None:
        """The moving-grid resolver picks the cell of its own phase's derivative (#30).

        The leaf is the trajectory ``x = v dt + a dt^2 / 2`` about t_init, expressed
        at t_cur, with ``v`` a fraction of a frequency cell.
        """
        accel, vel, dt = 5.0, frac * CELL_DV, t_cur - T_INIT
        leaf = _leaf(0.0, accel, vel + accel * dt, vel * dt + accel * dt**2 / 2, 100.0)

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

        picked, expected = instantaneous_cell(resolve, T_ADD, 100.0)
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
                assert_phase(phase[i, s], expected_phase(t_seg - t_cur, delay, f0))
                freq = f0 * (1 - p.deriv(1)(t_seg) / C_VAL)
                assert idx[i, s, 0] == grid_cell(p.deriv(2)(t_seg), LIMITS[0], 16)
                assert idx[i, s, 1] == grid_cell(freq, LIMITS[1], 64)


class TestBranch:
    COORD = (20.0, 12.0)

    def _leaves(self) -> np.ndarray:
        leaves = np.concatenate(
            [_leaf(0.0, 1.0, 0.0, 0.0, 100.0), _leaf(1e-4, -2.0, 50.0, 3.0, 100.7)],
        )
        # Parent steps a few times the new step for jerk and accel, so they branch,
        # and just under it for velocity, so it does not.
        dnew = psr_utils.poly_taylor_step_d_vec(
            3,
            self.COORD[1],
            NBINS,
            1.0,
            leaves[:, -1, 0],
            t_ref=0.0,
        )
        leaves[:, :3, 1] = dnew * np.array([2.5, 3.2, 0.9])
        return leaves

    @pytest.mark.parametrize("impl", jit_variants(taylor.poly_taylor_branch_batch))
    def test_children_tile_each_parent(self, impl) -> None:
        leaves = self._leaves()
        out, origins = impl(leaves, self.COORD, NBINS, 1.0, 3, 64)
        _, span = self.COORD
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_taylor_step_d_vec(3, span, NBINS, 1.0, f0, t_ref=0.0)
        shift = psr_utils.poly_taylor_shift_d_vec(
            leaves[:, :3, 1],
            dnew,
            span,
            NBINS,
            f0,
            t_ref=0.0,
        )
        # Both the branching and the pass-through path are exercised.
        assert (shift[:, :2] >= 1).all()
        assert (shift[:, 2] < 1).all()
        values, steps, want_origins = expected_children(
            leaves[:, :3, 0], leaves[:, :3, 1], dnew, shift
        )
        np.testing.assert_array_equal(origins, want_origins)
        assert len(out) == len(values) > 2
        np.testing.assert_allclose(out[:, :3, 0], values, rtol=1e-12)
        np.testing.assert_allclose(out[:, :3, 1], steps, rtol=1e-12)
        np.testing.assert_array_equal(out[:, 3, 0], leaves[origins, 3, 0])
        np.testing.assert_array_equal(out[:, -1], leaves[origins, -1])


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

    @pytest.mark.parametrize(
        ("impl", "approx_impl"),
        paired_jit_variants(
            taylor.generate_bp_poly_taylor, taylor.generate_bp_poly_taylor_approx
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
        """Each level is the mean branching factor, weighted by descendants."""
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
        assert not np.allclose(singles[0], singles[-1])
        np.testing.assert_allclose(mixed, descendant_weighted_mean(singles), rtol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(taylor.generate_bp_poly_taylor_approx),
    )
    def test_approx_level_count_above_branch_max_is_not_truncated(self, impl) -> None:
        """``branch_max`` caps children per parameter, not leaves per level.

        A level's leaf count is the product over parameters, so it can exceed
        ``branch_max`` without anything being truncated. The smallest
        ``branch_max`` that still runs must give the full pattern.
        """
        param_arr = typed.List([np.array([0.0]), np.array([100.0])])
        dparams = np.array([5000.0, 0.005])
        args = (10.0, 2, NBINS, 1.0, 0)
        kwargs = {"use_moving_grid": True, "tiling_strategy": "aggressive", "itree": 0}
        full = impl(param_arr, dparams, *args, **kwargs, branch_max=4096)
        smallest_ok = None
        for branch_max in range(1, int(full.max()) + 1):
            try:
                got = impl(param_arr, dparams, *args, **kwargs, branch_max=branch_max)
            except ValueError:
                continue
            smallest_ok = branch_max
            np.testing.assert_array_equal(got, full)
            break
        assert smallest_ok is not None
        assert full.max() > smallest_ok, "fixture must have a level above branch_max"

    @pytest.mark.parametrize(
        "impl",
        jit_variants(taylor.generate_bp_poly_taylor_approx),
    )
    def test_approx_raises_when_a_parameter_exceeds_branch_max(self, impl) -> None:
        kwargs = {"use_moving_grid": True, "tiling_strategy": "aggressive", "itree": 0}
        full = impl(self.PARAM_ARR, self.DPARAMS, *self.ARGS, **kwargs, branch_max=4096)
        assert full.max() > 1
        with pytest.raises(ValueError, match="increase branch_max"):
            impl(self.PARAM_ARR, self.DPARAMS, *self.ARGS, **kwargs, branch_max=1)
