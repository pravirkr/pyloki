"""Unit tests for `pyloki.utils.transforms`, the parameter-basis kernels.

These kernels move a candidate's parameters between reference times (`shift_taylor_*`,
`shift_taylor_circular_*`), between the Taylor and Chebyshev bases (`taylor_to_cheby*`,
`cheby_to_taylor*`), and between Chebyshev domains (`shift_cheby_*`). The pruning
search calls them at every stage, and a wrong transport does not raise: the search
keeps running and quietly looks in the wrong place.

Every contract is checked against an independent reference, not a golden number:

- parameter transports against the analytic derivatives of a polynomial or of a
  circular orbit, evaluated at the new time;
- basis changes against `numpy.polynomial`;
- error propagation against the exact image of the error box. The maps are linear, so
  "conservative" must be exactly the bounding box of the mapped box corners,
  "quadrature" exactly ``sqrt(diag(M Sigma M^T))``, and "aggressive" the diagonal;
- the box limits for Chebyshev coefficients against enumeration of the Taylor box's
  corners, both containing them and tight on them;
- each hand-unrolled kernel against the generic one it unrolls.

Tests run over ``jit_variants(...)`` (see `tests/jit_utils`), so the same assertion
runs against the compiled kernel and its ``py_func``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy import polynomial

from pyloki.utils import maths, transforms
from pyloki.utils.misc import C_VAL
from tests.jit_utils import jit_variants

# Library defects found while writing these tests. Each is pinned with a strict xfail
# so that the fix, when it lands, turns the pin into a failure and forces its removal.

STRATEGIES = ("conservative", "quadrature", "aggressive")


def _taylor_matrix(n_params: int, delta_t: float) -> np.ndarray:
    """Return M with ``new = old @ M.T`` for descending ``[d_k, ..., d_0]``."""
    mat = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i + 1):
            mat[i, j] = delta_t ** (i - j) / math.factorial(i - j)
    return mat


def _box_corners(half_widths: np.ndarray) -> np.ndarray:
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(half_widths))))
    return signs * half_widths


def _propagate(errors: np.ndarray, mat: np.ndarray, strategy: str) -> np.ndarray:
    """Exact references for ``new = old @ mat.T`` applied to an error box."""
    if strategy == "conservative":
        return np.abs(_box_corners(errors) @ mat.T).max(axis=0)
    if strategy == "quadrature":
        return np.sqrt(np.diag(mat @ np.diag(errors**2) @ mat.T))
    return errors * np.abs(np.diag(mat))


class TestShiftTaylorParams:
    POLY = polynomial.Polynomial([0.3, -1.1, 0.7, 2.0, -0.4, 0.05])

    def _state(self, t: float) -> np.ndarray:
        """Return ``[d_5, ..., d_0]``: the derivatives of POLY at `t`, descending."""
        return np.array([self.POLY.deriv(k)(t) for k in range(6)])[::-1]

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_params))
    @pytest.mark.parametrize("delta_t", [-2.1, 0.0, 0.9, 7.5])
    def test_transports_polynomial(self, impl, delta_t: float) -> None:
        np.testing.assert_allclose(
            impl(self._state(0.2), delta_t),
            self._state(0.2 + delta_t),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_params))
    @pytest.mark.parametrize("n_out", [-1, 0, 1, 3, 6, 9])
    def test_n_out_keeps_lowest_orders(self, impl, n_out: int) -> None:
        full = impl(self._state(0.2), 1.3)
        kept = 6 if n_out <= 0 else min(n_out, 6)
        np.testing.assert_allclose(impl(self._state(0.2), 1.3, n_out), full[-kept:])

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_params))
    def test_batch_and_non_contiguous_view(self, impl) -> None:
        """The docstring allows ``(..., n_params)`` and a non-contiguous view."""
        states = np.stack([self._state(t) for t in (0.1, 0.5, 2.0)])
        wide = np.zeros((3, 12))
        wide[:, ::2] = states
        view = wide[:, ::2]
        assert not view.flags.c_contiguous
        expected = np.stack([self._state(t + 0.7) for t in (0.1, 0.5, 2.0)])
        np.testing.assert_allclose(impl(view, 0.7), expected, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_params))
    def test_group_law(self, impl) -> None:
        start = self._state(0.2)
        np.testing.assert_allclose(
            impl(impl(start, 0.4), 1.1),
            impl(start, 1.5),
            rtol=1e-12,
        )


class TestShiftTaylorErrors:
    ERRORS = np.array([0.1, 0.5, 2.0, 3.0, 0.25])

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_errors))
    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("delta_t", [-1.3, 0.6, 4.0])
    def test_matches_exact_reference(self, impl, strategy: str, delta_t: float) -> None:
        mat = _taylor_matrix(len(self.ERRORS), delta_t)
        np.testing.assert_allclose(
            impl(self.ERRORS, delta_t, strategy),
            _propagate(self.ERRORS, mat, strategy),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_errors)
        + jit_variants(transforms.shift_taylor_full),
    )
    def test_rejects_unknown_strategy(self, impl) -> None:
        vec = np.ones((4, 2)) if "full" in impl.__name__ else np.ones(4)
        with pytest.raises(ValueError, match="Invalid tiling strategy"):
            impl(vec, 1.0, "bogus")

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_full))
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_full_matches_params_and_errors(self, impl, strategy: str) -> None:
        values = np.array([[0.2, -1.0, 3.0, 0.5, 7.0], [1.0, 2.0, -3.0, 4.0, 0.0]])
        errors = np.array([[0.1, 0.5, 2.0, 3.0, 0.25], [1.0, 1.0, 1.0, 1.0, 1.0]])
        out = impl(np.stack([values, errors], axis=-1), 2.3, strategy)
        np.testing.assert_allclose(
            out[..., 0],
            transforms.shift_taylor_params(values, 2.3),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            out[..., 1],
            transforms.shift_taylor_errors(errors, 2.3, strategy),
            rtol=1e-12,
        )


class TestShiftTaylorParamsDF:
    """``[..., j, a, f]`` shifted with zero initial velocity and delay."""

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_taylor_params_d_f))
    def test_closed_form(self, impl) -> None:
        jerk, accel, freq, dt = 2e-6, 3.0, 150.0, 2.5
        new, delay = impl(np.array([jerk, accel, freq]), dt)
        velocity = accel * dt + jerk * dt**2 / 2
        np.testing.assert_allclose(new[0], jerk, rtol=1e-12)
        np.testing.assert_allclose(new[1], accel + jerk * dt, rtol=1e-12)
        np.testing.assert_allclose(new[2], freq * (1 - velocity / C_VAL), rtol=1e-12)
        position = accel * dt**2 / 2 + jerk * dt**3 / 6
        np.testing.assert_allclose(delay, position / C_VAL, rtol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_params_d_f_batch),
    )
    def test_batch_matches_scalar(self, impl) -> None:
        batch = np.array([[2e-6, 3.0, 150.0], [0.0, -1.0, 90.0], [1e-4, 0.0, 400.0]])
        new, delay = impl(batch, 2.5)
        for row, new_row, d in zip(batch, new, delay, strict=True):
            ref_new, ref_d = transforms.shift_taylor_params_d_f(row, 2.5)
            np.testing.assert_allclose(new_row, ref_new, rtol=1e-12)
            np.testing.assert_allclose(d, ref_d, rtol=1e-12)


class CircularOrbit:
    """``x(t) = A sin(w t + phi) + v t + x0`` and its derivatives."""

    AMP, OMEGA, PHASE, VEL, X0 = 1.3e-3, 2 * np.pi / 3.0, 0.4, 0.7, 0.2

    def state(self, t: float) -> np.ndarray:
        """Return ``[d5, d4, d3, d2, d1, d0]`` at `t`."""
        a, w, arg = self.AMP, self.OMEGA, self.OMEGA * t + self.PHASE
        return np.array(
            [
                a * w**5 * np.cos(arg),
                a * w**4 * np.sin(arg),
                -a * w**3 * np.cos(arg),
                -a * w**2 * np.sin(arg),
                a * w * np.cos(arg) + self.VEL,
                a * np.sin(arg) + self.VEL * t + self.X0,
            ]
        )


class TestShiftTaylorCircular(CircularOrbit):
    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_params),
    )
    @pytest.mark.parametrize("in_hole", [False, True])
    @pytest.mark.parametrize("delta_t", [-0.8, 0.0, 1.37, 11.0])
    def test_params_transport_orbit(
        self,
        impl,
        in_hole: bool,
        delta_t: float,
    ) -> None:
        got = impl(self.state(0.2)[None], delta_t, in_hole)[0]
        expected = self.state(0.2 + delta_t)
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-15)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_full),
    )
    @pytest.mark.parametrize("in_hole", [False, True])
    def test_full_transports_values_keeps_errors(self, impl, in_hole: bool) -> None:
        errors = np.linspace(0.01, 0.06, 6)
        full = np.stack([self.state(0.2), errors], axis=-1)[None]
        out = impl(full, 1.37, "aggressive", in_hole)[0]
        np.testing.assert_allclose(out[:, 0], self.state(1.57), rtol=1e-9, atol=1e-15)
        np.testing.assert_array_equal(out[:, 1], errors)

    @pytest.mark.parametrize(
        ("impl", "shape"),
        [
            (p.values[0], (1, 5))
            for p in jit_variants(transforms.shift_taylor_circular_params)
        ]
        + [
            (p.values[0], (1, 5, 2))
            for p in jit_variants(transforms.shift_taylor_circular_full)
        ],
    )
    def test_requires_six_coefficients(self, impl, shape) -> None:
        with pytest.raises(ValueError, match="5 parameters"):
            impl(np.ones(shape), 1.0)


class TestShiftTaylorCircularErrors:
    P_ORB_MIN = 10.0
    ERRORS = np.array([[0.0, 0.0, 0.3, 0.2, 0.1, 0.05]])

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    @pytest.mark.parametrize("strategy", ["quadrature", "aggressive"])
    def test_higher_orders_follow_circular_constraint(
        self,
        impl,
        strategy: str,
    ) -> None:
        out = impl(self.ERRORS, 2.0, self.P_ORB_MIN, strategy)[0]
        w_sq = (2 * np.pi / self.P_ORB_MIN) ** 2
        np.testing.assert_allclose(out[0], w_sq * out[2])
        np.testing.assert_allclose(out[1], w_sq * out[3])
        assert out[5] == 0

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    def test_aggressive_keeps_errors(self, impl) -> None:
        out = impl(self.ERRORS, 2.0, self.P_ORB_MIN, "aggressive")[0]
        np.testing.assert_array_equal(out[2:5], self.ERRORS[0, 2:5])

    def _worst_accel_jerk(
        self,
        sig3: float,
        sig2: float,
        delta_t: float,
    ) -> tuple[float, float]:
        """Worst linear propagation of d2 and d3 over every allowed omega.

        ``d2_j = d2 cos + d3 sin / w`` and ``d3_j = d3 cos - d2 w sin`` for any
        ``w <= 2 pi / p_orb_min``.
        """
        w = np.linspace(1e-6, 2 * np.pi / self.P_ORB_MIN, 20001)
        c, s = np.cos(w * delta_t), np.sin(w * delta_t)
        worst_d2 = np.hypot(c * sig2, s / w * sig3).max()
        worst_d3 = np.hypot(c * sig3, w * s * sig2).max()
        return worst_d2, worst_d3

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    @pytest.mark.parametrize(("sig3", "sig2"), [(0.3, 0.2), (1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("delta_t", [0.5, 2.5, 5.0])
    def test_quadrature_accel_and_jerk_bound_the_transport(
        self,
        impl,
        sig3: float,
        sig2: float,
        delta_t: float,
    ) -> None:
        errors = np.array([[0.0, 0.0, sig3, sig2, 0.0, 0.0]])
        out = impl(errors, delta_t, self.P_ORB_MIN, "quadrature")[0]
        worst_d2, worst_d3 = self._worst_accel_jerk(sig3, sig2, delta_t)
        assert out[3] >= worst_d2 * (1 - 1e-12)
        assert out[2] >= worst_d3 * (1 - 1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    def test_quadrature_accel_and_jerk_bounds_are_tight(self, impl) -> None:
        """With one error at a time each bound is attained, so it is not loose.

        A jerk error alone reaches the acceleration as ``sin(w dt) / w``, which
        tends to ``dt`` as ``w -> 0``. An acceleration error alone reaches the jerk
        as ``w |sin(w dt)|``, which is ``w_max`` at ``dt = pi / (2 w_max)``.
        """
        delta_t = np.pi / 2 / (2 * np.pi / self.P_ORB_MIN)
        for sig3, sig2, row, index in [(1.0, 0.0, 3, 0), (0.0, 1.0, 2, 1)]:
            errors = np.array([[0.0, 0.0, sig3, sig2, 0.0, 0.0]])
            out = impl(errors, delta_t, self.P_ORB_MIN, "quadrature")[0]
            worst = self._worst_accel_jerk(sig3, sig2, delta_t)[index]
            np.testing.assert_allclose(out[row], worst, rtol=1e-6)

    @pytest.mark.xfail(strict=True, reason="#27", raises=AssertionError)
    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    @pytest.mark.parametrize("delta_t", [2.0, 5.0])
    def test_quadrature_velocity_bounds_the_transport(
        self,
        impl,
        delta_t: float,
    ) -> None:
        """Pin the velocity term scaling as ``dt`` instead of ``dt**2``.

        ``d1_j = d1 + d3 (1 - cos) / w**2 + d2 sin / w``, so a jerk error reaches the
        velocity with coefficient up to ``dt**2 / 2``. The kernel uses
        ``delta_t * sig_d3 / 2``, which has units of acceleration, and so falls
        short of the transport for ``dt > 1`` (in whatever time unit is used).
        """
        sig_jerk_only = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        out = impl(
            sig_jerk_only,
            delta_t,
            self.P_ORB_MIN,
            "quadrature",
        )[0]
        w_max = 2 * np.pi / self.P_ORB_MIN
        worst = max(
            (1 - np.cos(w * delta_t)) / w**2 for w in np.linspace(1e-4, w_max, 501)
        )
        assert out[4] >= worst * (1 - 1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.shift_taylor_circular_errors),
    )
    @pytest.mark.parametrize(
        ("vec", "strategy", "exc", "match"),
        [
            (np.ones((1, 5)), "aggressive", ValueError, "6 parameters"),
            (np.ones((1, 6)), "conservative", NotImplementedError, "Conservative"),
            (np.ones((1, 6)), "bogus", ValueError, "Invalid tiling strategy"),
        ],
    )
    def test_rejects_bad_input(self, impl, vec, strategy, exc, match) -> None:
        with pytest.raises(exc, match=match):
            impl(vec, 1.0, self.P_ORB_MIN, strategy)


class TestTaylorToCircular:
    """``[snap, jerk, accel, freq] -> [omega, freq, x_cos_phi, x_sin_phi]``."""

    AMP_LS, OMEGA, PHASE, FREQ = 2.0, 2 * np.pi / 3600.0, 0.9, 200.0

    def _inputs(self) -> tuple[np.ndarray, np.ndarray]:
        amp = self.AMP_LS * C_VAL
        w, arg = self.OMEGA, self.PHASE
        values = np.array(
            [
                amp * w**4 * np.sin(arg),
                -amp * w**3 * np.cos(arg),
                -amp * w**2 * np.sin(arg),
                self.FREQ,
            ]
        )
        errors = np.abs(values) * 1e-3
        errors[3] = 1e-6
        return values, errors

    @staticmethod
    def _forward(q: np.ndarray) -> np.ndarray:
        snap, jerk, accel, freq = q
        w_sq = -snap / accel
        return np.array(
            [
                np.sqrt(w_sq),
                freq * (1 + jerk / (w_sq * C_VAL)),
                -accel / (w_sq * C_VAL),
                -jerk / (w_sq**1.5 * C_VAL),
            ]
        )

    def _linear_errors(self) -> np.ndarray:
        values, errors = self._inputs()
        jac = np.zeros((4, 4))
        for i in range(4):
            step = np.zeros(4)
            step[i] = abs(values[i]) * 1e-6
            jac[:, i] = (
                self._forward(values + step) - self._forward(values - step)
            ) / (2 * step[i])
        return np.sqrt((jac**2 * errors**2).sum(axis=1))

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_circular_full))
    def test_values_recover_the_orbit(self, impl) -> None:
        values, errors = self._inputs()
        out = impl(np.stack([values, errors], axis=-1)[None])[0]
        velocity = self.AMP_LS * C_VAL * self.OMEGA * np.cos(self.PHASE)
        np.testing.assert_allclose(out[0, 0], self.OMEGA, rtol=1e-12)
        np.testing.assert_allclose(
            out[1, 0],
            self.FREQ * (1 - velocity / C_VAL),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            out[2, 0],
            self.AMP_LS * np.sin(self.PHASE),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            out[3, 0],
            self.AMP_LS * np.cos(self.PHASE),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_circular_full))
    @pytest.mark.parametrize("row", [0, 1, 3], ids=["omega", "freq", "x_sin_phi"])
    def test_errors_are_first_order(self, impl, row: int) -> None:
        values, errors = self._inputs()
        out = impl(np.stack([values, errors], axis=-1)[None])[0]
        np.testing.assert_allclose(out[row, 1], self._linear_errors()[row], rtol=1e-5)

    @pytest.mark.xfail(strict=True, reason="#28", raises=AssertionError)
    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_circular_full))
    def test_x_cos_phi_error_is_first_order(self, impl) -> None:
        """Pin the accel/omega correlation dropped from ``dx_cos_phi``.

        ``x_cos_phi = -accel / (omega**2 C)`` with ``omega**2 = -snap / accel``, so it
        is ``accel**2 / (snap C)``, and accel enters twice. The kernel adds the two
        routes in quadrature as if independent; at equal relative errors on snap and
        accel it reports sqrt(3/5) = 0.775 of the first-order error.
        """
        values, errors = self._inputs()
        out = impl(np.stack([values, errors], -1)[None])
        np.testing.assert_allclose(out[0, 2, 1], self._linear_errors()[2], rtol=1e-5)


class TestTaylorChebyshevBasis:
    TS = 1.7
    D_VEC = np.array([0.5, 2.3, -1.5, 1.1, 0.4])  # [d4, ..., d0]
    ERRORS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def _alpha(self) -> np.ndarray:
        return transforms.taylor_to_cheby(self.D_VEC, self.TS)

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_cheby))
    def test_same_polynomial(self, impl) -> None:
        alpha = impl(self.D_VEC, self.TS)
        t = np.linspace(-self.TS, self.TS, 11)
        coeffs = self.D_VEC[::-1] / special_factorials(len(self.D_VEC))
        np.testing.assert_allclose(
            polynomial.Chebyshev(alpha[::-1])(t / self.TS),
            polynomial.Polynomial(coeffs)(t),
            atol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(transforms.cheby_to_taylor))
    def test_inverse(self, impl) -> None:
        np.testing.assert_allclose(impl(self._alpha(), self.TS), self.D_VEC, atol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_cheby))
    def test_batch_rows_independent(self, impl) -> None:
        batch = np.stack([self.D_VEC, 2 * self.D_VEC, self.D_VEC[::-1]])
        out = impl(batch, self.TS)
        for row, got in zip(batch, out, strict=True):
            np.testing.assert_allclose(got, transforms.taylor_to_cheby(row, self.TS))

    @staticmethod
    def _linear_map(func, n: int, ts: float) -> np.ndarray:
        """Row i is the image of basis vector i."""
        return np.array([func(np.eye(n)[i], ts) for i in range(n)])

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_cheby_errors))
    def test_errors_are_quadrature_of_the_map(self, impl) -> None:
        jac = self._linear_map(transforms.taylor_to_cheby, 5, self.TS)
        expected = np.sqrt((self.ERRORS[:, None] ** 2 * jac**2).sum(axis=0))
        np.testing.assert_allclose(impl(self.ERRORS, self.TS), expected, rtol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(transforms.taylor_to_cheby_full))
    def test_full_to_cheby(self, impl) -> None:
        out = impl(np.stack([self.D_VEC, self.ERRORS], axis=-1), self.TS)
        np.testing.assert_allclose(out[:, 0], self._alpha(), rtol=1e-12)
        np.testing.assert_allclose(
            out[:, 1],
            transforms.taylor_to_cheby_errors(self.ERRORS, self.TS),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(transforms.cheby_to_taylor_full))
    def test_full_to_taylor(self, impl) -> None:
        out = impl(np.stack([self._alpha(), self.ERRORS], axis=-1), self.TS)
        jac = self._linear_map(transforms.cheby_to_taylor, 5, self.TS)
        np.testing.assert_allclose(out[:, 0], self.D_VEC, atol=1e-12)
        np.testing.assert_allclose(
            out[:, 1],
            np.sqrt((self.ERRORS[:, None] ** 2 * jac**2).sum(axis=0)),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        ("impl", "ndim"),
        [
            (p.values[0], ndim)
            for func, ndim in [
                (transforms.taylor_to_cheby, 1),
                (transforms.taylor_to_cheby_errors, 1),
                (transforms.cheby_to_taylor, 1),
                (transforms.taylor_to_cheby_full, 2),
                (transforms.cheby_to_taylor_full, 2),
            ]
            for p in jit_variants(func)
        ],
    )
    @pytest.mark.parametrize("t_s", [0.0, -1.0])
    def test_rejects_non_positive_scale(self, impl, ndim: int, t_s: float) -> None:
        vec = np.ones((3, 2)) if ndim == 2 else np.ones(3)
        with pytest.raises(ValueError, match="t_s must be a positive"):
            impl(vec, t_s)


def special_factorials(n: int) -> np.ndarray:
    return np.array([math.factorial(k) for k in range(n)], dtype=float)


# Module level so that the parametrize comprehensions below can see them.
CHEBY_ALPHA = np.array([[0.3, -0.2, 1.4, 0.8, -0.5], [1.0, 0.0, -2.0, 0.5, 0.1]])
CHEBY_ERRORS = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [1.0, 0.5, 0.25, 0.125, 0.0625]])


class TestShiftCheby:
    COORD_CUR, COORD_NEXT = (0.0, 2.0), (0.5, 1.0)
    ALPHA = CHEBY_ALPHA
    ERRORS = CHEBY_ERRORS

    def _c_mat(self) -> np.ndarray:
        (tc1, ts1), (tc2, ts2) = self.COORD_CUR, self.COORD_NEXT
        return maths.poly_chebyshev_transform_matrix(4, tc1, ts1, tc2, ts2, 1)

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_cheby_full))
    def test_values_same_function_on_new_domain(self, impl) -> None:
        full = np.stack([self.ALPHA, self.ERRORS], axis=-1)
        out = impl(full, self.COORD_NEXT, self.COORD_CUR, "aggressive")
        (tc1, ts1), (tc2, ts2) = self.COORD_CUR, self.COORD_NEXT
        t = np.linspace(tc2 - ts2, tc2 + ts2, 13)
        for old, new in zip(self.ALPHA, out[..., 0], strict=True):
            np.testing.assert_allclose(
                polynomial.Chebyshev(new[::-1])((t - tc2) / ts2),
                polynomial.Chebyshev(old[::-1])((t - tc1) / ts1),
                atol=1e-12,
            )

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_cheby_errors))
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_errors_match_exact_reference(self, impl, strategy: str) -> None:
        # Descending coefficients as row vectors: new = old @ c_mat.
        mat = self._c_mat().T
        out = impl(self.ERRORS, self.COORD_NEXT, self.COORD_CUR, strategy)
        for errors, got in zip(self.ERRORS, out, strict=True):
            np.testing.assert_allclose(
                got,
                _propagate(errors, mat, strategy),
                rtol=1e-12,
                atol=1e-15,
            )

    @pytest.mark.parametrize("impl", jit_variants(transforms.shift_cheby_full))
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_full_errors_match_errors(self, impl, strategy: str) -> None:
        full = np.stack([self.ALPHA, self.ERRORS], axis=-1)
        out = impl(full, self.COORD_NEXT, self.COORD_CUR, strategy)
        np.testing.assert_allclose(
            out[..., 1],
            transforms.shift_cheby_errors(
                self.ERRORS,
                self.COORD_NEXT,
                self.COORD_CUR,
                strategy,
            ),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        ("impl", "vec"),
        [
            (p.values[0], CHEBY_ERRORS)
            for p in jit_variants(transforms.shift_cheby_errors)
        ]
        + [
            (p.values[0], np.stack([CHEBY_ALPHA, CHEBY_ERRORS], axis=-1))
            for p in jit_variants(transforms.shift_cheby_full)
        ],
    )
    def test_rejects_unknown_strategy(self, impl, vec) -> None:
        with pytest.raises(ValueError, match="Invalid tiling strategy"):
            impl(vec, self.COORD_NEXT, self.COORD_CUR, "bogus")


class TestChebyToTaylorParamShift:
    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.cheby_to_taylor_param_shift),
    )
    @pytest.mark.parametrize("t_eval", [0.4, 1.1, -1.0])
    def test_derivatives_at_t_eval(self, impl, t_eval: float) -> None:
        t0, ts = 0.4, 1.7
        alpha = np.array([[0.3, -0.2, 1.4, 0.8, -0.5], [0.0, 1.0, 0.0, -1.0, 2.0]])
        out = impl(alpha, t0, ts, t_eval)
        for row, got in zip(alpha, out, strict=True):
            series = polynomial.Chebyshev(row[::-1], domain=[t0 - ts, t0 + ts])
            expected = [
                series.deriv(k)(t_eval) if k else series(t_eval) for k in range(5)
            ]
            np.testing.assert_allclose(got, expected[::-1], rtol=1e-11, atol=1e-12)


class TestChebyshevLimits:
    """Box limits on ``[d_k, ..., d_1]`` mapped to ``[alpha_k, ..., alpha_1]``.

    Every connection coefficient is non-negative, so the map is monotone in each input:
    the limits must contain the image of every corner of the Taylor box and be
    attained by two of them.
    """

    TS = 1.3

    @staticmethod
    def _limits(n_params: int, n_batch: int = 3, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed + n_params)
        return np.sort(rng.normal(size=(n_batch, n_params, 2)), axis=-1)

    def _corner_extremes(self, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_params = box.shape[0]
        corners = np.array(
            [
                [box[i, side[i]] for i in range(n_params)]
                for side in itertools.product([0, 1], repeat=n_params)
            ]
        )
        with_d0 = np.hstack([corners, np.zeros((len(corners), 1))])
        alpha = transforms.taylor_to_cheby(with_d0, self.TS)[:, :-1]
        return alpha.min(axis=0), alpha.max(axis=0)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.taylor_to_chebyshev_limits_generic),
    )
    @pytest.mark.parametrize("n_params", [1, 2, 3, 4, 5, 6])
    def test_generic_is_tight(self, impl, n_params: int) -> None:
        limits = self._limits(n_params)
        out = impl(limits, self.TS)
        for box, got in zip(limits, out, strict=True):
            lo, hi = self._corner_extremes(box)
            np.testing.assert_allclose(got[:, 0], lo, atol=1e-12)
            np.testing.assert_allclose(got[:, 1], hi, atol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.taylor_to_chebyshev_limits_full),
    )
    @pytest.mark.parametrize("n_params", [2, 3, 4, 5])
    def test_full_unrolls_generic(self, impl, n_params: int) -> None:
        limits = self._limits(n_params)
        np.testing.assert_allclose(
            impl(limits, self.TS),
            transforms.taylor_to_chebyshev_limits_generic(limits, self.TS),
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.taylor_to_chebyshev_limits_d1),
    )
    @pytest.mark.parametrize("n_higher", [1, 2, 3, 4])
    def test_d1_unrolls_generic(self, impl, n_higher: int) -> None:
        """``taylor_limits`` holds ``[d_k, ..., d_2]`` and is shared by the batch."""
        limits = self._limits(n_higher + 1, n_batch=4)
        limits[:, :-1] = limits[0, :-1]
        generic = transforms.taylor_to_chebyshev_limits_generic(limits, self.TS)
        out = impl(limits[:, -1], limits[0, :-1], self.TS)
        np.testing.assert_allclose(out, generic[:, -1], atol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.taylor_to_chebyshev_limits_upto_d2),
    )
    def test_upto_d2_single_param(self, impl) -> None:
        limits = self._limits(2, n_batch=1)
        generic = transforms.taylor_to_chebyshev_limits_generic(limits, self.TS)
        np.testing.assert_allclose(impl(limits[0, :-1], self.TS), generic[0, :-1])

    @pytest.mark.xfail(strict=True, reason="#26", raises=AssertionError)
    @pytest.mark.parametrize(
        "impl",
        jit_variants(transforms.taylor_to_chebyshev_limits_upto_d2),
    )
    @pytest.mark.parametrize("n_higher", [2, 3, 4])
    def test_upto_d2_unrolls_generic(self, impl, n_higher: int) -> None:
        """Pin the unrolled branches disagreeing with the docstring and each other.

        The docstring orders the output ``[alpha_kmax, ..., alpha_2]``. The 2- and
        3-parameter branches return it ascending, and the 4-parameter branch returns
        it descending but writes ``alpha_2`` to row 0 and then overwrites that row
        with ``alpha_5``, so ``alpha_2`` is lost and row 3 stays zero.
        """
        limits = self._limits(n_higher + 1, n_batch=1)
        generic = transforms.taylor_to_chebyshev_limits_generic(limits, self.TS)
        np.testing.assert_allclose(
            impl(limits[0, :-1], self.TS),
            generic[0, :-1],
        )

    @pytest.mark.parametrize(
        ("impl", "args"),
        [
            (p.values[0], (np.ones((1, 6, 2)), 1.0))
            for p in jit_variants(transforms.taylor_to_chebyshev_limits_full)
        ]
        + [
            (p.values[0], (np.ones((1, 2)), np.ones((5, 2)), 1.0))
            for p in jit_variants(transforms.taylor_to_chebyshev_limits_d1)
        ]
        + [
            (p.values[0], (np.ones((5, 2)), 1.0))
            for p in jit_variants(transforms.taylor_to_chebyshev_limits_upto_d2)
        ],
    )
    def test_unrolled_reject_too_many_params(self, impl, args) -> None:
        with pytest.raises(ValueError, match="not supported"):
            impl(*args)
