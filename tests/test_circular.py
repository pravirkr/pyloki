"""Unit tests for `pyloki.core.circular`, the circular-orbit pruning kernels.

A leaf holds a fifth-order Taylor state ``[d5, d4, d3, d2, d1, d0]`` (crackle, snap,
jerk, accel, velocity, position) with its steps, and ``f0``. Each leaf is classified
as a circular orbit identified by snap and accel, as one identified by crackle and jerk
(the "hole", where snap and accel vanish), or as plain Taylor. Circular leaves are
moved with the exact circular propagator, and Taylor leaves with the Taylor shift.

The references are these.

- Classification: targeted cases for each class and the significance and sign rules,
  plus the partition property (each leaf in exactly one class).
- Resolvers: for each class, the state at the new time from the propagator that class
  uses (`transforms.shift_taylor_circular_params`, `shift_taylor_params`, both tested
  in `test_transforms.py`) and, for a circular orbit, from the analytic orbit. The
  phase is ``frac(((t_add - t_init) - delay) f0) nbins``, compared on the circle.
- The frequency cell each resolver picks must contain the instantaneous frequency of
  its own phase model (the time derivative of the phase it returns). The fixed grid
  meets that for every class; the moving grid does not (pinned, as in `core.taylor`).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numba import typed

from pyloki.core import circular
from pyloki.utils import psr_utils, transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.snail import MiddleOutScheme
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.
MOVING_GRID_ISSUE = "#30: the moving-grid resolver can select the adjacent FFA frequency cell"
FLAG_ISSUE = "#31: the basis flag is written to the wrong leaves"
UNUSED_ISSUE = "#36: two unused circular kernels do not work"

NBINS = 64
LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])  # [accel, freq]
COUNTS = np.array([16, 64])
SIG = 2.0
# T_ADD - T_INIT is not a whole number of cycles at f0, so a wrong proper time shows.
T_INIT, T_ADD = 10.0, 30.0037
CELL_DV = (2 / 64) * C_VAL / 100.0

AMP, OMEGA, PHASE = 3e4, 2 * np.pi / 400.0, 0.7


def _orbit_state(
    t: float, vel_offset: float = 0.0, t_ref: float = T_INIT
) -> np.ndarray:
    """Return [d5..d0] at t of the orbit plus a constant velocity offset.

    ``x(t) = AMP sin(OMEGA (t - t_ref) + PHASE) + vel_offset (t - t_ref)``.
    """
    a, w, arg = AMP, OMEGA, OMEGA * (t - t_ref) + PHASE
    return np.array(
        [
            a * w**5 * np.cos(arg),
            a * w**4 * np.sin(arg),
            -a * w**3 * np.cos(arg),
            -a * w**2 * np.sin(arg),
            a * w * np.cos(arg) + vel_offset,
            a * np.sin(arg) + vel_offset * (t - t_ref),
        ]
    )


def _leaf(state: np.ndarray, steps: np.ndarray, f0: float = 100.0) -> np.ndarray:
    leaf = np.zeros((1, 7, 2))
    leaf[0, :6, 0] = state
    leaf[0, :6, 1] = steps
    leaf[0, -1, 0] = f0
    return leaf


def _snap_leaf(vel_offset: float = 0.0) -> np.ndarray:
    state = _orbit_state(T_INIT, vel_offset)
    return _leaf(state, np.abs(state) * 1e-3 + 1e-12)


def _hole_leaf() -> np.ndarray:
    """Snap and accel vanish; crackle and jerk are significant and physical."""
    state = _orbit_state(T_INIT)
    state[1] = state[3] = 0.0
    steps = np.abs(_orbit_state(T_INIT)) * 1e-3 + np.array([0, 1, 0, 1, 0, 0])
    return _leaf(state, steps)


def _taylor_leaf(vel: float = 0.0, accel: float = 5.0) -> np.ndarray:
    state = np.array([0.0, 0.0, 0.0, accel, vel, 0.0])
    return _leaf(state, np.array([1.0, 1.0, 1.0, 0.01, 10.0, 0.0]))


def _cell(value: float, limits: np.ndarray, count: int) -> int:
    lo, hi = limits
    return int(np.clip(np.floor(count * (value - lo) / (hi - lo)), 0, count - 1))


def _phase(dt: float, delay: float, f0: float) -> float:
    return ((dt - delay) * f0) % 1 * NBINS


def _assert_phase(got: float, expected: float) -> None:
    diff = (got - expected + NBINS / 2) % NBINS - NBINS / 2
    assert abs(diff) < 1e-6, (got, expected)


class TestClassification:
    @pytest.mark.parametrize("impl", jit_variants(circular.get_circ_taylor_mask))
    def test_three_classes_and_partition(self, impl) -> None:
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf()])
        snap, crackle, tay = impl(leaves, SIG)
        np.testing.assert_array_equal(snap, [0])
        np.testing.assert_array_equal(crackle, [1])
        np.testing.assert_array_equal(tay, [2])

    @pytest.mark.parametrize("impl", jit_variants(circular.get_circ_mask))
    def test_rules(self, impl) -> None:
        # Rows: significant & physical snap -> snap; significant but unphysical
        # sign -> not snap; insignificant snap, significant jerk, physical crackle ->
        # crackle; insignificant everything -> taylor; physical crackle but
        # insignificant jerk -> taylor; physical snap below significance
        # (0.4 < 2 * 0.5) -> taylor; significant jerk but unphysical crackle -> taylor.
        crackle = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        snap = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.4, 0.0])
        accel = np.array([-3.0, 3.0, 0.0, 0.0, 0.0, -3.0, 0.0])
        jerk = np.array([0.0, 0.0, -5.0, 0.0, -0.1, 0.0, 5.0])
        d = np.full(7, 0.5)
        idx_snap, idx_crackle, idx_taylor = impl(
            crackle, snap, d, jerk, d, accel, d, SIG
        )
        np.testing.assert_array_equal(idx_snap, [0])
        np.testing.assert_array_equal(idx_crackle, [2])
        np.testing.assert_array_equal(idx_taylor, [1, 3, 4, 5, 6])

    @pytest.mark.parametrize("impl", jit_variants(circular.get_circ_mask))
    def test_partition_random(self, impl) -> None:
        rng = np.random.default_rng(3)
        vals = [rng.standard_normal(200) for _ in range(7)]
        crackle, snap, dsnap, jerk, djerk, accel, daccel = vals
        out = impl(
            crackle,
            snap,
            np.abs(dsnap),
            jerk,
            np.abs(djerk),
            accel,
            np.abs(daccel),
            SIG,
        )
        together = np.sort(np.concatenate(out))
        np.testing.assert_array_equal(together, np.arange(200))

    @pytest.mark.parametrize("impl", jit_variants(circular.get_circ_taylor_mask_branch))
    def test_branch_mask_expands_only_hole_leaves_that_need_it(self, impl) -> None:
        leaves = np.concatenate(
            [_snap_leaf(), _hole_leaf(), _hole_leaf(), _taylor_leaf()]
        )
        needs = np.array([True, True, False, True])
        expand, keep = impl(leaves[:, :-2, 0], leaves[:, :-2, 1], needs, SIG)
        np.testing.assert_array_equal(expand, [1])
        np.testing.assert_array_equal(keep, [0, 2, 3])


class TestValidate:
    P_ORB_MIN, X_MASS = 100.0, 1e6

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_validate_batch))
    def test_keeps_physical_and_hole_drops_the_rest(self, impl) -> None:
        w_max_sq = (2 * np.pi / self.P_ORB_MIN) ** 2
        #          ok     wrong sign  too fast          too large accel   hole
        snap = np.array(
            [0.5 * w_max_sq, 0.5 * w_max_sq, 4 * w_max_sq, 0.5 * w_max_sq, 0.0]
        )
        accel = np.array([-1.0, 1.0, -1.0, -1e9, 0.0])
        d = np.full(5, 1e-9)
        d[-1] = 1.0
        kept = impl(snap, d, accel, d, self.P_ORB_MIN, self.X_MASS, 5.0)
        np.testing.assert_array_equal(kept, [0, 4])

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_validate_batch))
    def test_leaf_wrapper_reads_snap_and_accel_rows(self, impl) -> None:
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf()])
        origins = np.array([7, 8, 9])
        kept, kept_origins = impl(leaves, origins, 100.0, 1e12, 5.0)
        expected = circular.circ_validate_batch(
            leaves[:, 1, 0],
            leaves[:, 1, 1],
            leaves[:, 3, 0],
            leaves[:, 3, 1],
            100.0,
            1e12,
            5.0,
        )
        np.testing.assert_array_equal(kept, leaves[expected])
        np.testing.assert_array_equal(kept_origins, origins[expected])


def _propagate(leaf: np.ndarray, dt: float) -> np.ndarray:
    """State at t + dt by the propagator the classifier assigns to the leaf."""
    snap, crackle, _ = circular.get_circ_taylor_mask(leaf, SIG)
    vec = leaf[:, :-1, 0]
    if snap.size:
        return transforms.shift_taylor_circular_params(vec, dt)[0]
    if crackle.size:
        return transforms.shift_taylor_circular_params(vec, dt, in_hole=True)[0]
    return transforms.shift_taylor_params(vec, dt)[0]


LEAVES = {
    "snap": _snap_leaf,
    "hole": _hole_leaf,
    "taylor": _taylor_leaf,
}


class TestResolve:
    @pytest.mark.parametrize(
        "impl", jit_variants(circular.circ_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("kind", list(LEAVES))
    def test_fixed_grid(self, impl, kind: str) -> None:
        leaf = LEAVES[kind]()
        idx, phase = impl(
            leaf.copy(),
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            SIG,
        )
        state = _propagate(leaf, T_ADD - T_INIT)
        _assert_phase(phase[0], _phase(T_ADD - T_INIT, state[-1] / C_VAL, 100.0))
        assert idx[0, 0] == _cell(state[-3], LIMITS[0], 16)
        assert idx[0, 1] == _cell(100.0 * (1 - state[-2] / C_VAL), LIMITS[1], 64)

    def test_circular_propagator_matches_the_analytic_orbit(self) -> None:
        """Guard the reference: the propagator reproduces the orbit it models."""
        np.testing.assert_allclose(
            _propagate(_snap_leaf(), 13.7),
            _orbit_state(T_INIT + 13.7),
            rtol=1e-8,
            atol=1e-9,
        )

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
    @pytest.mark.parametrize("kind", ["taylor", "snap"])
    def test_moving_grid_where_velocity_at_t_init_is_zero(
        self, impl, kind: str
    ) -> None:
        """Phase and cells, for a leaf whose velocity vanishes at ``t_init``."""
        leaf = LEAVES[kind]()
        leaf[0, 4, 0] = 0.0  # v(t_init) = 0 (for the orbit: a zero-velocity epoch)
        t_cur = 18.3719
        moved = leaf.copy()
        moved[0, :6, 0] = _propagate(leaf, t_cur - T_INIT)
        idx, phase = impl(
            moved.copy(),
            (T_ADD, 1.0),
            (t_cur, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            SIG,
        )
        at_add = _propagate(moved, T_ADD - t_cur)
        at_init = _propagate(moved, T_INIT - t_cur)
        delay = (at_add[-1] - at_init[-1]) / C_VAL
        _assert_phase(phase[0], _phase(T_ADD - T_INIT, delay, 100.0))
        assert idx[0, 0] == _cell(at_add[-3], LIMITS[0], 16)
        assert idx[0, 1] == _cell(100.0 * (1 - at_add[-2] / C_VAL), LIMITS[1], 64)

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
    def test_moving_grid_hole_leaf_uses_the_crackle_propagator(self, impl) -> None:
        """Phase and accel cell for a hole leaf; its freq cell has the #30 defect."""
        leaf = _hole_leaf()
        idx, phase = impl(
            leaf.copy(),
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            SIG,
        )
        at_add = _propagate(leaf, T_ADD - T_INIT)
        delay = (at_add[-1] - leaf[0, 5, 0]) / C_VAL
        _assert_phase(phase[0], _phase(T_ADD - T_INIT, delay, 100.0))
        assert idx[0, 0] == _cell(at_add[-3], LIMITS[0], 16)

    @staticmethod
    def _instantaneous_cell(resolve) -> tuple[int, int]:
        eps = 1e-3
        hi = resolve(T_ADD + eps)[1] / NBINS
        lo = resolve(T_ADD - eps)[1] / NBINS
        dcyc = (hi - lo + 0.5) % 1 - 0.5
        f_inst = (dcyc + round(100.0 * 2 * eps - dcyc)) / (2 * eps)
        return resolve(T_ADD)[0][1], _cell(f_inst, LIMITS[1], 64)

    @pytest.mark.parametrize(
        "impl", jit_variants(circular.circ_taylor_fixed_resolve_batch)
    )
    @pytest.mark.parametrize("kind", ["taylor", "snap"])
    @pytest.mark.parametrize("frac", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_fixed_grid_cell_follows_its_own_phase(
        self,
        impl,
        kind: str,
        frac: float,
    ) -> None:
        leaf = (
            _taylor_leaf(frac * CELL_DV)
            if kind == "taylor"
            else _snap_leaf(frac * CELL_DV)
        )

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf.copy(),
                (t, 1.0),
                (T_INIT, 1.0),
                (T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
                SIG,
            )
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve)
        assert picked == expected

    @pytest.mark.xfail(strict=True, reason=MOVING_GRID_ISSUE, raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
    @pytest.mark.parametrize(
        ("kind", "frac"),
        [("taylor", -0.5), ("taylor", -0.25), ("snap", 0.25), ("snap", 0.5)],
    )
    def test_moving_grid_cell_follows_its_own_phase(
        self,
        impl,
        kind: str,
        frac: float,
    ) -> None:
        """Pin the moving grid dropping the velocity at t_init from the frequency.

        As in `core.taylor`: the subtraction of the leaf's own ``v(t_init)`` follows
        the per-class propagation, so it applies to every class. A Taylor-class leaf
        at -0.5 cell picks 31 against 32; a circular-orbit leaf with a +0.25 or +0.5
        cell velocity offset picks 32 against 31.
        """
        leaf = (
            _taylor_leaf(frac * CELL_DV)
            if kind == "taylor"
            else _snap_leaf(frac * CELL_DV)
        )

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(
                leaf.copy(),
                (t, 1.0),
                (T_INIT, 1.0),
                (T_INIT, 1.0),
                COUNTS,
                LIMITS,
                NBINS,
                SIG,
            )
            return idx[0], ph[0]

        picked, expected = self._instantaneous_cell(resolve)
        assert picked == expected

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
    def test_moving_grid_flags_snap_class(self, impl) -> None:
        leaves = np.concatenate([_snap_leaf(), _taylor_leaf()])
        leaves[:, -1, 1] = 9.0
        impl(
            leaves,
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            SIG,
        )
        assert leaves[0, -1, 1] == 1

    @pytest.mark.xfail(strict=True, reason=FLAG_ISSUE, raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
    def test_moving_grid_flags_every_class(self, impl) -> None:
        """Pin the Taylor branch writing the flag to the crackle indices.

        Inside ``if idx_taylor.size > 0`` the kernel sets
        ``leaves_batch[idx_circ_crackle, -1, 1] = 0``: the crackle leaves lose their
        flag (2) whenever a Taylor leaf is present, and the Taylor leaves keep
        whatever flag they had. The flags reach the result file through the report.
        """
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf()])
        leaves[:, -1, 1] = 9.0
        impl(
            leaves,
            (T_ADD, 1.0),
            (T_INIT, 1.0),
            (T_INIT, 1.0),
            COUNTS,
            LIMITS,
            NBINS,
            SIG,
        )
        np.testing.assert_array_equal(leaves[:, -1, 1], [1, 2, 0])

    @pytest.mark.parametrize(
        "impl", jit_variants(circular.circ_taylor_ascend_resolve_batch)
    )
    def test_ascend(self, impl) -> None:
        leaves = np.concatenate(
            [
                _snap_leaf(),
                _hole_leaf(),
                _taylor_leaf(2.3e5),
                _taylor_leaf(-3.1e5, -3.0),
            ],
        )
        t_cur = T_INIT
        segs = np.array([[4.137, 1.0], [12.0071, 1.0], [26.3719, 1.0]])
        idx, phase = impl(leaves, segs, (t_cur, 1.0), COUNTS, LIMITS, NBINS, SIG)
        assert idx.shape == (4, 3, 2)
        for i in range(4):
            for s, (t_seg, _) in enumerate(segs):
                state = _propagate(leaves[i : i + 1], t_seg - t_cur)
                _assert_phase(
                    phase[i, s], _phase(t_seg - t_cur, state[-1] / C_VAL, 100.0)
                )
                assert idx[i, s, 0] == _cell(state[-3], LIMITS[0], 16)
                assert idx[i, s, 1] == _cell(
                    100.0 * (1 - state[-2] / C_VAL), LIMITS[1], 64
                )


class TestTransform:
    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_transform_batch))
    @pytest.mark.parametrize("strategy", ["aggressive", "conservative"])
    def test_each_class_uses_its_propagator(self, impl, strategy: str) -> None:
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf(1e4)])
        out = impl(leaves, (25.0, 1.0), (T_INIT, 1.0), strategy, SIG)
        np.testing.assert_allclose(
            out[0, :-1],
            transforms.shift_taylor_circular_full(leaves[0:1, :-1], 15.0, strategy)[0],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            out[1, :-1],
            transforms.shift_taylor_circular_full(
                leaves[1:2, :-1], 15.0, strategy, in_hole=True
            )[0],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            out[2, :-1],
            transforms.shift_taylor_full(leaves[2:3, :-1], 15.0, strategy)[0],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(out[:, -1], leaves[:, -1])


class TestBranch:
    COORD = (20.0, 12.0)

    @staticmethod
    def _reference(
        leaves: np.ndarray,
        span: float,
        branch_max: int,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Build the expected children from psr_utils pieces, in the kernel's order."""
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_taylor_step_d_vec(5, span, NBINS, 1.0, f0, t_ref=0)
        shift = psr_utils.poly_taylor_shift_d_vec(
            leaves[:, :5, 1], dnew, span, NBINS, f0, t_ref=0
        )
        needs = shift >= 1.0 - 1e-12
        first = []  # (params, steps, origin)
        for i, leaf in enumerate(leaves):
            axes, steps = [[leaf[0, 0]]], []
            steps.append(
                psr_utils.branch_dparam_crackle(leaf[0, 1], dnew[i, 0], branch_max)
                if needs[i, 0]
                else leaf[0, 1],
            )
            for j in range(1, 5):
                if needs[i, j]:
                    vals, dact = psr_utils.branch_param(
                        leaf[j, 0], leaf[j, 1], dnew[i, j]
                    )
                    axes.append(list(vals))
                    steps.append(dact)
                else:
                    axes.append([leaf[j, 0]])
                    steps.append(leaf[j, 1])
            first += [
                (np.array(c), np.array(steps), i) for c in itertools.product(*axes)
            ]
        params = np.array([p for p, _, _ in first])
        dparams = np.array([d for _, d, _ in first])
        origins = np.array([o for _, _, o in first])
        expand, keep = circular.get_circ_taylor_mask_branch(
            params, dparams, needs[origins, 0], SIG
        )
        out = [(params[k], dparams[k], origins[k]) for k in keep]
        for k in expand:
            i = origins[k]
            vals, dact = psr_utils.branch_param(
                params[k, 0], leaves[i, 0, 1], dnew[i, 0]
            )
            for v in vals:
                p, d = params[k].copy(), dparams[k].copy()
                p[0], d[0] = v, dact
                out.append((p, d, i))
        return out

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_branch_batch))
    def test_children_match_the_reference(self, impl) -> None:
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf(2e3)])
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_taylor_step_d_vec(
            5, self.COORD[1], NBINS, 1.0, f0, t_ref=0
        )
        # Coarse steps on crackle, jerk and velocity so that the leaves branch; the
        # hole leaf keeps a fine jerk step so its jerk stays significant (it is then
        # in the hole) and expands crackle.
        leaves[:, :5, 1] = dnew * np.array([3.3, 2.1, 2.4, 0.5, 2.2])
        leaves[1, 2, 1] = abs(leaves[1, 2, 0]) / 10
        out, origins = impl(leaves, self.COORD, NBINS, 1.0, 5, 64, SIG)
        expected = self._reference(leaves, self.COORD[1], 64)
        assert len(out) == len(expected)
        assert any(p[0] != leaves[o, 0, 0] for p, _, o in expected)  # crackle expanded
        for leaf, origin, (p, d, o) in zip(out, origins, expected, strict=True):
            assert origin == o
            np.testing.assert_allclose(leaf[:5, 0], p, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(leaf[:5, 1], d, rtol=1e-12)
            assert leaf[5, 0] == leaves[o, 5, 0]
            np.testing.assert_array_equal(leaf[-1], leaves[o, -1])

    @pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_branch_batch))
    def test_no_crackle_expansion_fast_path(self, impl) -> None:
        leaves = np.concatenate([_snap_leaf(), _taylor_leaf(2e3)])
        f0 = leaves[:, -1, 0]
        dnew = psr_utils.poly_taylor_step_d_vec(
            5, self.COORD[1], NBINS, 1.0, f0, t_ref=0
        )
        leaves[:, :5, 1] = dnew * np.array([0.5, 0.5, 2.4, 0.5, 2.2])
        out, origins = impl(leaves, self.COORD, NBINS, 1.0, 5, 64, SIG)
        expected = self._reference(leaves, self.COORD[1], 64)
        assert len(out) == len(expected) > 2
        for leaf, origin, (p, d, o) in zip(out, origins, expected, strict=True):
            assert origin == o
            np.testing.assert_allclose(leaf[:5, 0], p, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(leaf[:5, 1], d, rtol=1e-12)


class TestUnusedKernels:
    """Kernels nothing in ``src/pyloki`` calls; their contracts, as far as stated."""

    @pytest.mark.parametrize("impl", jit_variants(circular.get_circ_chebyshev_mask))
    # A long domain (omega ts > sqrt(6)) makes the snap term in alpha_2 decide
    # the sign of the accel the mask derives.
    @pytest.mark.parametrize("ts", [7.0, 300.0])
    def test_chebyshev_mask_classifies_the_taylor_form(self, impl, ts: float) -> None:
        leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf()])
        cheb = leaves.copy()
        cheb[:, :-1] = transforms.taylor_to_cheby_full(leaves[:, :-1], ts)
        got = impl(cheb, ts, SIG)
        want = circular.get_circ_taylor_mask(leaves, SIG)
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)

    @pytest.mark.parametrize("impl", jit_variants(circular.poly_circular_resolve_batch))
    def test_physical_resolve_phase_and_accel(self, impl) -> None:
        """``[omega, x cos nu, x sin nu, f]`` with ``x`` in light-seconds."""
        # x = 0.1 light-seconds keeps both frequencies on the grid (no clamping).
        w, x, nu, f = OMEGA, 0.1, 0.7, 100.0
        leaf = np.zeros((1, 6, 2))
        leaf[0, :4, 0] = [w, x * np.cos(nu), x * np.sin(nu), f]
        idx, phase = impl(leaf, (T_ADD, 1.0), (T_INIT, 1.0), COUNTS, LIMITS, NBINS)
        nu_new = nu + w * (T_ADD - T_INIT)
        delay = x * np.sin(nu_new) - x * np.sin(nu)
        _assert_phase(phase[0], _phase(T_ADD - T_INIT, delay, f))
        assert idx[0, 0] == _cell(-C_VAL * w**2 * x * np.sin(nu_new), LIMITS[0], 16)

    @pytest.mark.xfail(strict=True, reason=UNUSED_ISSUE, raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(circular.poly_circular_resolve_batch))
    def test_physical_resolve_cell_follows_its_own_phase(self, impl) -> None:
        """Pin the frequency sign: ``f (1 + v)`` where its phase gives ``f (1 - v)``.

        Its delay is ``x sin(nu)``, so the instantaneous frequency of its own phase
        is ``f (1 - omega x cos(nu))``; the kernel uses ``f (1 + ...)``, the opposite
        of every other resolver here. On this leaf it picks cell 34 against 29.
        """
        # x = 0.1 light-seconds keeps both frequencies on the grid (no clamping).
        w, x, nu, f = OMEGA, 0.1, 0.7, 100.0
        leaf = np.zeros((1, 6, 2))
        leaf[0, :4, 0] = [w, x * np.cos(nu), x * np.sin(nu), f]

        def resolve(t: float) -> tuple[np.ndarray, float]:
            idx, ph = impl(leaf, (t, 1.0), (T_INIT, 1.0), COUNTS, LIMITS, NBINS)
            return idx[0], ph[0]

        picked, expected = TestResolve._instantaneous_cell(resolve)  # noqa: SLF001
        assert picked == expected

    @pytest.mark.xfail(strict=True, reason=UNUSED_ISSUE, raises=ValueError)
    @pytest.mark.parametrize("impl", jit_variants(circular.cir_physical_branch_batch))
    def test_physical_branch_runs(self, impl) -> None:
        """Pin a shape error that makes it fail on any input.

        ``psr_utils.poly_taylor_step_d_vec(2, ...)`` returns shape ``(n_batch, 2)``,
        which the kernel assigns into the single column ``dparam_opt_batch[:, 0]``.
        """
        leaves = np.zeros((2, 7, 2))
        leaves[:, :4, 0] = 1.0
        leaves[:, :4, 1] = 0.1
        leaves[:, -2, 0] = 100.0
        impl(leaves, (20.0, 12.0), NBINS, 1.0, 5, 16)


class TestBranchingPattern:
    DPARAMS = np.array([1e-9, 1e-6, 2e-4, 2.0, 0.05])
    ARGS = (10.0, 16, NBINS, 1.0, 8)

    @pytest.mark.parametrize("impl", jit_variants(circular.generate_bp_circ_taylor))
    def test_requires_five_parameters(self, impl) -> None:
        with pytest.raises(ValueError, match="exactly 5 parameters"):
            impl(
                typed.List([np.array([0.0]), np.array([0.0]), np.array([100.0])]),
                np.array([2e-4, 2.0, 0.05]),
                *self.ARGS,
                use_moving_grid=True,
                tiling_strategy="aggressive",
            )

    @staticmethod
    def _param_arr(freqs: np.ndarray) -> typed.List:
        return typed.List([np.array([0.0])] * 4 + [freqs])

    @pytest.mark.parametrize("impl", jit_variants(circular.generate_bp_circ_taylor))
    @pytest.mark.parametrize("moving", [True, False])
    def test_exact_averages_over_frequencies(self, impl, moving: bool) -> None:
        freqs = np.array([20.0, 100.0, 400.0])
        kwargs = {"use_moving_grid": moving, "tiling_strategy": "aggressive"}
        dparams = np.array([1e-9, 1e-3, 1e-2, 200.0, 0.05])
        mixed = impl(self._param_arr(freqs), dparams, *self.ARGS, **kwargs)
        singles = np.array(
            [
                impl(self._param_arr(np.array([f])), dparams, *self.ARGS, **kwargs)
                for f in freqs
            ]
        )
        assert not np.allclose(singles[0], singles[-1])
        weights = np.cumprod(np.hstack([np.ones((3, 1)), singles[:, :-1]]), axis=1)
        expected = (weights * singles).sum(axis=0) / weights.sum(axis=0)
        np.testing.assert_allclose(mixed, expected, rtol=1e-12)

    @staticmethod
    def _reference_one_frequency(
        dparams: np.ndarray,
        f0: float,
        moving: bool,
    ) -> np.ndarray:
        """Apply the documented rule for one frequency, with psr_utils and transforms.

        Each level multiplies the ceil(step ratio) of every parameter except crackle
        that needs branching. The first level at which snap splits is halved, because
        half of those children fail validation. On the moving grid the steps are
        carried to the next reference time.
        """
        tseg, nsegments, nbins, eta, ref = TestBranchingPattern.ARGS
        scheme = MiddleOutScheme(nsegments, ref, tseg, 1)
        cur = dparams.astype(float).copy()
        cur[-1] *= C_VAL / f0
        snap_seen, pattern = False, []
        for level in range(1, nsegments):
            _, span = scheme.get_current_coord(level, moving)
            new = psr_utils.poly_taylor_step_d_vec(
                5, span, nbins, eta, np.array([f0]), t_ref=0
            )[0]
            shift = psr_utils.poly_taylor_shift_d_vec(
                cur[None],
                new[None],
                span,
                nbins,
                np.array([f0]),
                t_ref=0,
            )[0]
            factor, snap_split = 1.0, False
            nxt = cur.copy()
            for j in range(1, 5):
                if shift[j] < eta - 1e-12:
                    continue
                n = max(1, int(np.ceil(cur[j] / new[j] - 1e-12)))
                factor *= n
                nxt[j] = cur[j] / n
                snap_split |= j == 1 and n > 1
            if snap_split and not snap_seen:
                factor *= 0.5
                snap_seen = True
            pattern.append(factor)
            if moving:
                dt = (
                    scheme.get_coord(level)[0]
                    - scheme.get_current_coord(level, moving)[0]
                )
                full = np.append(nxt, 0.0)
                nxt = transforms.shift_taylor_errors(full[None], dt, "aggressive")[
                    0, :-1
                ]
            cur = nxt
        return np.array(pattern)

    @pytest.mark.parametrize("impl", jit_variants(circular.generate_bp_circ_taylor))
    @pytest.mark.parametrize("moving", [True, False])
    def test_one_frequency_follows_the_documented_rule(
        self, impl, moving: bool
    ) -> None:
        # A snap step large enough to split snap (a halved level, 2.5, appears).
        dparams = np.array([1e-9, 100.0, 1e-2, 200.0, 0.05])
        got = impl(
            self._param_arr(np.array([100.0])),
            dparams,
            *self.ARGS,
            use_moving_grid=moving,
            tiling_strategy="aggressive",
        )
        expected = self._reference_one_frequency(dparams, 100.0, moving)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        # The config does split snap, so the halving is exercised.
        assert not np.allclose(
            got,
            self._reference_one_frequency(dparams * [1, 1e-12, 1, 1, 1], 100.0, moving),
        )
