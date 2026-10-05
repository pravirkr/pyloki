import math

import numpy as np
import pytest
from numpy import polynomial
from scipy import special, stats

from pyloki.utils import maths, transforms
from tests.jit_utils import jit_variants, python_impl

# ``maths.norm_isf_func`` linearly interpolates a table of ``norm.isf(exp(-x))``
# sampled every ``maths.minus_logsf_res`` (= 0.1). The tabulated function is
# steepest as x -> 0 (it diverges to -inf at x = 0), so the interpolation error
# peaks in the first table cells and decays quickly outwards. The tolerances below
# are the accuracy the function actually delivers, measured on a 10**6-point grid
# over [0, 10]:
#     [0.1, 0.2)  max |err| 2.7e-2   <- exceeds the usual decimal=2 bound (1.5e-2)
#     [0.2, 0.3)  max |err| 1.0e-2
#     [0.3, 1.0)  max |err| 5.4e-3
#     [1.0, 10]   max |err| 7.5e-4
# Below the first node the table cannot follow the divergence to -inf at x = 0;
# the values there are finite and monotonic but only qualitative, see
# ``TestMaths.test_norm_isf_func_below_first_node_is_approximate``.
# The points are deliberately off the 0.1-spaced table nodes, where the error is
# identically zero and the test would prove nothing.
NORM_ISF_POINTS = [
    (0.125, 3e-2),
    (0.145, 3e-2),
    (0.175, 3e-2),
    (0.230, 1.5e-2),
    (0.270, 1.5e-2),
    (0.350, 6e-3),
    (0.470, 6e-3),
    (0.660, 6e-3),
    (1.050, 1e-3),
    (1.630, 1e-3),
    (2.440, 1e-3),
    (3.870, 1e-3),
    (5.550, 1e-3),
    (7.310, 1e-3),
    (9.420, 1e-3),
]

# Same idea for ``maths.chi_sq_minus_logsf_func``, whose table is sampled every
# ``maths.chi_sq_res`` (= 0.5): points off the table nodes, spanning [0, 10].
CHI_SQ_POINTS = [0.25, 0.75, 1.6, 3.3, 6.2, 9.8]


class TestMaths:
    @pytest.mark.parametrize(
        ("n", "k"),
        [(5, 2), (6, 7), (2, 3), (20, 12), (5, 0), (5, 5), (0, 0)],
    )
    def test_nbinom(self, n: int, k: int) -> None:
        np.testing.assert_almost_equal(maths.nbinom(n, k), special.binom(n, k))

    @pytest.mark.parametrize("n", [0, 1, 5, 20, np.array([0, 1, 5, 20])])
    def test_fact(self, n: int | np.ndarray) -> None:
        np.testing.assert_almost_equal(maths.fact(n), special.factorial(n))

    @pytest.mark.parametrize(("minus_logsf", "atol"), NORM_ISF_POINTS)
    def test_norm_isf_func(self, minus_logsf: float, atol: float) -> None:
        expected = stats.norm.isf(np.exp(-minus_logsf))
        np.testing.assert_allclose(
            maths.norm_isf_func(minus_logsf),
            expected,
            rtol=0,
            atol=atol,
        )

    def test_norm_isf_table_is_finite(self) -> None:
        """Entry 0 used to be ``norm.isf(exp(-0)) == norm.isf(1) == -inf``."""
        assert np.isfinite(maths.norm_isf_table).all()

    @pytest.mark.parametrize("minus_logsf", [0.0, 1e-9, 0.05, 0.099])
    def test_norm_isf_func_first_cell_is_finite(self, minus_logsf: float) -> None:
        """The first cell [0, ``minus_logsf_res``) used to come back non-finite.

        ``gen_norm_isf_table`` starts at x = 0, where the tabulated function is
        ``-inf``, and that infinity propagated through every interpolation in the
        first cell. Entry 0 now carries the extrapolation of the first two finite
        nodes, so the cell is finite. It is not accurate -- see
        ``test_norm_isf_func_below_first_node_is_approximate``.
        """
        result = maths.norm_isf_func(minus_logsf)
        assert np.isfinite(result), f"expected finite, got {result}"
        assert result < 0, f"expected a non-detection, got {result}"

    @pytest.mark.parametrize("minus_logsf", [-1e-9, -0.5, -1.0, -10.0])
    def test_norm_isf_func_out_of_domain_is_negative(self, minus_logsf: float) -> None:
        """Negative input is out of domain and must not read as a detection.

        ``sf = exp(-minus_logsf) > 1`` here, so the honest answer is "very
        negative". ``pos`` went negative, ``int(pos)`` truncated towards zero and
        the negative index wrapped to the tail of the table, so these inputs came
        back at ~+28 sigma -- a non-detection reported as a maximal detection.
        Reachable from ``scoring.py`` via ``chi_sq_minus_logsf_func(...) -
        lee_penalty``, which is routinely negative on noise.
        """
        result = maths.norm_isf_func(minus_logsf)
        assert np.isfinite(result), f"expected finite, got {result}"
        assert result < maths.norm_isf_func(0.0), (
            f"out-of-domain input scored {result}, at or above the x = 0 value"
        )

    def test_norm_isf_func_is_monotonic(self) -> None:
        """Increasing evidence must never decrease the score, and vice versa.

        Spans the out-of-domain region, the extrapolated first cell, the table and
        the ``max_minus_logsf`` tail extrapolation in one sweep, which is what
        catches an index that wraps or a seam that steps the wrong way.
        """
        x = np.linspace(-10, 500, 100_001)
        got = np.array([maths.norm_isf_func(val) for val in x])
        assert np.isfinite(got).all()
        np.testing.assert_array_less(
            -1e-9,
            np.diff(got),
            err_msg="norm_isf_func decreased as minus_logsf increased",
        )

    @pytest.mark.parametrize("minus_logsf", [399.91, 399.95, 399.99])
    def test_norm_isf_func_last_cell_stays_inside_the_table(
        self,
        minus_logsf: float,
    ) -> None:
        """The top cell used to read one entry past the end of the table.

        ``np.arange(0, max_minus_logsf, minus_logsf_res)`` stops one step short, so
        the top node is at 399.9, not 400. The guard tested ``minus_logsf <
        max_minus_logsf``, so [399.9, 400) still interpolated and reached
        ``norm_isf_table[4000]`` in a 4000-entry table. These functions are njit'd
        and unchecked, so that read returned whatever was in memory -- ~1.6e185.
        """
        result = maths.norm_isf_func(minus_logsf)
        assert np.isfinite(result)
        np.testing.assert_allclose(result, maths.norm_isf_table[-1], atol=1e-2)

    @pytest.mark.parametrize("df", [1, 2, 3, 32, 64])
    @pytest.mark.parametrize("chi_sq", [299.6, 299.9, 299.99])
    def test_chi_sq_minus_logsf_func_last_cell_stays_inside_its_row(
        self,
        chi_sq: float,
        df: int,
    ) -> None:
        """The top cell used to read across into the next ``df`` row.

        Same off-by-one-cell as ``norm_isf_func``, but the table is 2-D and C
        contiguous, so ``[df, 600]`` in a (65, 600) table silently returned row
        ``df + 1``'s first entry: ``chi_sq_minus_logsf_func(299.9, 2)`` gave 29.95
        where the true value is ~149.95. At ``df = 64``, the last row, the read
        left the array entirely.
        """
        result = maths.chi_sq_minus_logsf_func(chi_sq, df)
        assert np.isfinite(result)
        # Measured: the table's own worst relative error over these points is
        # 1.05e-3, at df = 64 on the tail extrapolation. The cross-row read this
        # guards against was wrong by 0.80 relative, so 2e-3 clears the accuracy
        # floor by 2x and still catches the defect with ~400x to spare.
        np.testing.assert_allclose(result, -stats.chi2.logsf(chi_sq, df), rtol=2e-3)

    @pytest.mark.parametrize("df", [1, 2, 3, 32, 64])
    def test_chi_sq_minus_logsf_func_is_monotonic(self, df: int) -> None:
        """Sweeps the table, the top seam and the tail extrapolation together."""
        x = np.linspace(0, 600, 50_001)
        got = np.array([maths.chi_sq_minus_logsf_func(val, df) for val in x])
        assert np.isfinite(got).all()
        np.testing.assert_array_less(
            -1e-9,
            np.diff(got),
            err_msg=f"chi_sq_minus_logsf_func(df={df}) decreased as chi_sq increased",
        )

    def test_norm_isf_func_below_first_node_is_approximate(self) -> None:
        """Pin how wrong the first cell still is, so "finite" is not read as "right".

        ``norm.isf(exp(-x))`` diverges to -inf at x = 0, so no finite entry on a
        uniform ``minus_logsf_res`` grid can follow it. The extrapolated entry 0
        keeps the cell finite and monotonic and is right to ~0.5 sigma over most of
        it, but reads high without bound as x -> 0. A finer table below x = 1 is
        what would fix this; if one lands, this test is the one to update.
        """
        x = np.linspace(1e-12, maths.minus_logsf_res, 2001, endpoint=False)
        err = np.array([maths.norm_isf_func(val) for val in x]) - stats.norm.isf(
            np.exp(-x),
        )
        assert np.isfinite(err).all()
        # High, not low: the approximation overstates significance here.
        np.testing.assert_array_less(-1e-9, err)
        np.testing.assert_array_less(1.0, err.max(), err_msg="first cell now accurate")
        assert err.max() < 6.0
        # ...but it is usable over most of the cell.
        assert err[x >= 0.02].max() < 0.5

    def test_norm_isf_func_first_cell_accuracy_limit(self) -> None:
        """Characterise the error spike in the first *finite* table cell.

        With ``minus_logsf_res = 0.1`` the table cannot follow the curvature of
        ``norm.isf(exp(-x))`` just above zero: the error peaks at ~2.7e-2 near
        x = 0.145, i.e. the function is not accurate to two decimals there. This
        is why ``NORM_ISF_POINTS`` carries a looser tolerance below x = 0.2.
        """
        x = np.linspace(maths.minus_logsf_res, 2 * maths.minus_logsf_res, 201)
        got = np.array([maths.norm_isf_func(val) for val in x])
        err = np.abs(got - stats.norm.isf(np.exp(-x)))
        np.testing.assert_array_less(
            1.5e-2,
            err.max(),
            err_msg="first cell now meets decimal=2; tighten NORM_ISF_POINTS",
        )
        np.testing.assert_array_less(err.max(), 3e-2)

    @pytest.mark.parametrize("df", [2, 3, 5, 10, 32])
    @pytest.mark.parametrize("chi_sq", CHI_SQ_POINTS)
    def test_chi_sq_minus_logsf_func(self, chi_sq: float, df: int) -> None:
        expected = -stats.chi2.logsf(chi_sq, df)
        np.testing.assert_allclose(
            maths.chi_sq_minus_logsf_func(chi_sq, df),
            expected,
            rtol=0,
            atol=1.5e-2,
        )

    def test_chi_sq_minus_logsf_func_df1_accuracy_limit(self) -> None:
        """Characterise the one df where the chi2 table misses two decimals.

        ``-chi2.logsf(x, 1)`` has an infinite slope at x = 0, and with
        ``chi_sq_res = 0.5`` the first cell is far too coarse for it: the error
        reaches ~1.4e-1 near x = 0.12 and stays above 1.5e-2 out to x ~ 0.47.
        Every other df tested is within 1.1e-2 over [0, 10] (worst: df = 3).
        """
        x = np.linspace(0, maths.chi_sq_res, 201)
        got = np.array([maths.chi_sq_minus_logsf_func(val, 1) for val in x])
        err = np.abs(got + stats.chi2.logsf(x, 1))
        np.testing.assert_array_less(
            1.5e-2,
            err.max(),
            err_msg="df=1 now meets decimal=2; fold df=1 into CHI_SQ_POINTS",
        )
        np.testing.assert_array_less(err.max(), 1.5e-1)

    @pytest.mark.parametrize(("order_max", "n_derivs"), [(3, 1), (5, 3), (10, 10)])
    def test_gen_chebyshev_polys_table(self, order_max: int, n_derivs: int) -> None:
        expected = maths.gen_chebyshev_polys_table_np(order_max, n_derivs)
        np.testing.assert_equal(
            expected.shape,
            (n_derivs + 1, order_max + 1, order_max + 1),
        )
        np.testing.assert_almost_equal(
            maths.gen_chebyshev_polys_table(order_max, n_derivs),
            expected,
            decimal=2,
        )


class TestChebyshevTransform:
    def test_connection_coefficients_s(self) -> None:
        # S_{0,0} = 1
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_s(0, 0),
            1.0,
        )
        # S_{2,0} = 1/2, S_{2,2} = 1/2
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_s(2, 0),
            0.5,
        )
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_s(2, 2),
            0.5,
        )
        # S_{3,1} = 3/4, S_{3,3} = 1/4
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_s(3, 1),
            0.75,
        )
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_s(3, 3),
            0.25,
        )

    def test_connection_coefficients_r(self) -> None:
        # R_{2,0} = 1, R_{2,2} = 2
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_r(2, 0),
            -1.0,
        )
        np.testing.assert_almost_equal(
            maths.compute_connection_coefficient_r(2, 2),
            2.0,
        )

    def test_taylor_to_cheby_manual(self) -> None:
        # Snap parameters
        d_vec = np.array([0.5, 2.3, 1500.0, 1e6, 1e4])
        t_s = 4.2
        # Expected coefficients
        alpha_4 = d_vec[0] * t_s**4 / (8 * maths.fact(4))
        alpha_3 = d_vec[1] * t_s**3 / (4 * maths.fact(3))
        alpha_2 = 0.5 * (
            (d_vec[2] * t_s**2 / maths.fact(2)) + (d_vec[0] * t_s**4 / (maths.fact(4)))
        )
        alpha_1 = d_vec[3] * t_s + (0.75 * d_vec[1] * t_s**3 / maths.fact(3))
        alpha_0 = (
            d_vec[4]
            + d_vec[2] * t_s**2 / (2 * maths.fact(2))
            + 3 * d_vec[0] * t_s**4 / (8 * maths.fact(4))
        )
        alpha_expected = np.array([alpha_4, alpha_3, alpha_2, alpha_1, alpha_0])
        alpha = transforms.taylor_to_cheby(d_vec, t_s)
        np.testing.assert_almost_equal(alpha, alpha_expected, decimal=12)

    def test_cheby_to_taylor_manual(self) -> None:
        alpha_vec = np.array([0.5, 2.3, 1500.0, 1e6, 1e4])
        t_s = 4.2
        # Expected coefficients
        d_4 = 192 * alpha_vec[0] / t_s**4
        d_3 = 24 * alpha_vec[1] / t_s**3
        d_2 = 4 * (alpha_vec[2] - 4 * alpha_vec[0]) / t_s**2
        d_1 = 1 * (alpha_vec[3] - 3 * alpha_vec[1]) / t_s
        d_0 = alpha_vec[4] - alpha_vec[2] + alpha_vec[0]
        d_expected = np.array([d_4, d_3, d_2, d_1, d_0])
        d = transforms.cheby_to_taylor(alpha_vec, t_s)
        np.testing.assert_almost_equal(d, d_expected, decimal=12)

    @pytest.mark.parametrize("k_max", [2, 4, 6])
    def test_roundtrip_identity(self, k_max: int) -> None:
        rng = np.random.default_rng(42)
        d_vec = rng.random(k_max + 1)
        t_s = 1.5
        alpha = transforms.taylor_to_cheby(d_vec, t_s)
        d_reconstructed = transforms.cheby_to_taylor(alpha, t_s)
        np.testing.assert_almost_equal(d_vec, d_reconstructed, decimal=12)

    def test_polynomial_evaluation(self) -> None:
        d_vec = np.array([0.5, 2.3, 1500.0, 1e6, 1e4])
        k_max = len(d_vec) - 1
        t_c, t_s = 4.2, 2.6
        alpha_vec = transforms.taylor_to_cheby(d_vec, t_s)
        t_test = np.linspace(t_c - t_s, t_c + t_s, 11)
        x = (t_test - t_c) / t_s
        k_range = np.arange(k_max + 1)
        c_power = d_vec[::-1] / maths.fact(k_range)
        taylor_poly = polynomial.Polynomial(c_power)
        val_taylor = taylor_poly(t_test - t_c)
        cheby_poly = polynomial.Chebyshev(alpha_vec[::-1], domain=[-1, 1])
        val_cheby = cheby_poly(x)
        np.testing.assert_almost_equal(val_taylor, val_cheby, decimal=8)


# --------------------------------------------------------------------------------------
# Kernel tests toward #12. Each contract runs on both the compiled kernel and its Python
# source via ``jit_variants`` (see tests/jit_utils.py). `norm_isf_func` and
# `chi_sq_minus_logsf_func` are deliberately not covered here: their table edges are
# fixed and tested in #21.
#
# Library defects found while writing these are pinned with a strict xfail, so that the
# fix, when it lands, turns the pin into a failure and forces its removal.


class TestFactKernel:
    @pytest.mark.parametrize("impl", jit_variants(maths.fact))
    @pytest.mark.parametrize("n", range(21))
    def test_exact_through_20(self, impl, n: int) -> None:
        assert impl(n) == math.factorial(n)

    def test_broadcasts(self) -> None:
        n = np.arange(21).reshape(3, 7)
        expected = np.array([math.factorial(k) for k in range(21)], dtype=float)
        np.testing.assert_array_equal(maths.fact(n), expected.reshape(3, 7))

    @pytest.mark.xfail(strict=True, reason="#22", raises=AssertionError)
    @pytest.mark.parametrize("n", [21, 30, 66, 119])
    def test_above_20(self, n: int) -> None:
        """Pin the int64 overflow in the factorial table.

        `fact_factory` fills its table with ``_fact(ii, 0)``, an int64 product in
        compiled code, which overflows from 21! on: ``fact(21)`` is -4.2e18 and
        ``fact(n)`` is 0 for every n >= 66. The table is built at import, so both paths
        see it.
        """
        np.testing.assert_allclose(maths.fact(n), float(math.factorial(n)), rtol=1e-12)


class TestNbinomKernel:
    @pytest.mark.parametrize("impl", jit_variants(maths.nbinom))
    def test_exact_through_20(self, impl) -> None:
        for n in range(21):
            for k in range(n + 1):
                assert impl(n, k) == math.comb(n, k), (n, k)

    @pytest.mark.parametrize("impl", jit_variants(maths.nbinom))
    @pytest.mark.parametrize(("n", "k"), [(5, -1), (5, 6), (0, 1), (130, 131)])
    def test_outside_triangle_is_zero(self, impl, n: int, k: int) -> None:
        assert impl(n, k) == 0

    @pytest.mark.parametrize("impl", jit_variants(maths.nbinom))
    @pytest.mark.parametrize(("n", "k"), [(121, 1), (130, 5), (200, 3), (1000, 2)])
    def test_multiplicative_branch(self, impl, n: int, k: int) -> None:
        """Above n = 120 the product formula is used; exact while it fits in int64."""
        assert impl(n, k) == math.comb(n, k)
        assert impl(n, n - k) == math.comb(n, k)

    # Wrong below n = 67, ZeroDivisionError from there (compiled): both are the defect.
    @pytest.mark.xfail(
        strict=True,
        reason="#22",
        raises=(AssertionError, ZeroDivisionError),
    )
    @pytest.mark.parametrize(("n", "k"), [(21, 1), (30, 15), (100, 50)])
    def test_factorial_branch_above_20(self, n: int, k: int) -> None:
        """Pin the factorial branch inheriting the broken `fact`.

        For 21 <= n <= 120 the result divides values of the broken `fact`
        (`TestFactKernel.test_above_20`): wrong, or ZeroDivisionError from n = 67.
        """
        assert maths.nbinom(n, k) == math.comb(n, k)

    @pytest.mark.xfail(strict=True, reason="#22", raises=AssertionError)
    def test_multiplicative_branch_compiled_overflows(self) -> None:
        """Pin the compiled and Python paths diverging above int64.

        Compiled, the product formula wraps in int64; ``.py_func`` uses Python ints
        and is exact, so the two paths diverge: C(200, 100) is 9.2e16 vs 9.1e58.
        """
        assert maths.nbinom(200, 100) == maths.nbinom.py_func(200, 100)


class TestIsPowerOfTwo:
    @pytest.mark.parametrize("impl", jit_variants(maths.is_power_of_two))
    def test_matches_brute_force(self, impl) -> None:
        powers = {2**i for i in range(12)}
        for n in range(-8, 3000):
            assert impl(n) == (n in powers), n


class TestChebyshevPolysTable:
    @pytest.mark.parametrize("impl", jit_variants(maths.gen_chebyshev_polys_table))
    @pytest.mark.parametrize(("order_max", "n_derivs"), [(1, 0), (3, 1), (8, 3)])
    def test_matches_numpy(self, impl, order_max: int, n_derivs: int) -> None:
        np.testing.assert_allclose(
            impl(order_max, n_derivs),
            maths.gen_chebyshev_polys_table_np(order_max, n_derivs),
            rtol=1e-6,
        )

    @pytest.mark.parametrize("impl", jit_variants(maths.generalized_cheb_pols))
    @pytest.mark.parametrize(("t0", "scale"), [(0.0, 1.0), (0.0, 1.7), (0.3, 1.0)])
    def test_generalized_where_correct(self, impl, t0: float, scale: float) -> None:
        """Correct whenever ``t0 == 0`` or ``scale == 1``; see the xfail below."""
        self._check_generalized(impl, t0, scale)

    @pytest.mark.xfail(strict=True, reason="#24", raises=AssertionError)
    def test_generalized_shifted_and_scaled(self) -> None:
        """Pin the shift being divided by ``scale`` twice.

        The docstring defines row n as ``T_n((x - t0) / scale)``. The kernel returns
        ``T_n((x - t0 / scale) / scale)``: the shift is applied after scaling. The
        two agree only at ``t0 = 0`` or ``scale = 1``.
        """
        self._check_generalized(maths.generalized_cheb_pols, 0.3, 1.7)

    @staticmethod
    def _check_generalized(impl, t0: float, scale: float) -> None:
        order = 4
        table = impl(order, t0, scale)
        x = np.linspace(t0 - scale, t0 + scale, 17)
        for n in range(order + 1):
            expected = polynomial.Chebyshev.basis(n)((x - t0) / scale)
            got = polynomial.polynomial.polyval(x, table[n])
            np.testing.assert_allclose(got, expected, atol=1e-5)


class TestDesignMatrixTaylor:
    @pytest.mark.parametrize("impl", jit_variants(maths.gen_design_matrix_taylor))
    def test_entries(self, impl) -> None:
        t_vals = np.array([-1.3, 0.0, 0.4, 2.2])
        mat = impl(t_vals, 5)
        assert mat.shape == (4, 6)
        expected = t_vals[:, None] ** np.arange(6) / special.factorial(np.arange(6))
        np.testing.assert_allclose(mat, expected, rtol=1e-6)

    def test_is_float32(self) -> None:
        assert maths.gen_design_matrix_taylor(np.array([0.5]), 3).dtype == np.float32


class TestPolyTaylorTransformMatrix:
    """`alpha_new = alpha_old @ T` moves a Taylor state by ``delta_t`` (docstring)."""

    POLY = polynomial.Polynomial([0.3, -1.1, 0.7, 2.0, -0.4])

    def _state(self, t: float) -> np.ndarray:
        """Return the polynomial and its first four derivatives at `t`, ascending."""
        return np.array([self.POLY.deriv(k)(t) for k in range(5)])

    @pytest.mark.parametrize("impl", jit_variants(maths.poly_taylor_transform_matrix))
    @pytest.mark.parametrize("delta_t", [-0.7, 0.0, 0.9, 3.0])
    def test_transports_polynomial_state(self, impl, delta_t: float) -> None:
        t0 = 0.2
        np.testing.assert_allclose(
            self._state(t0) @ impl(4, delta_t, 0),
            self._state(t0 + delta_t),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("impl", jit_variants(maths.poly_taylor_transform_matrix))
    def test_descending_is_reversed_ascending(self, impl) -> None:
        np.testing.assert_array_equal(impl(4, 0.7, 1), impl(4, 0.7, 0)[::-1, ::-1])

    @pytest.mark.parametrize("impl", jit_variants(maths.poly_taylor_transform_matrix))
    def test_group_law(self, impl) -> None:
        np.testing.assert_allclose(
            impl(5, 0.3, 0) @ impl(5, 0.5, 0),
            impl(5, 0.8, 0),
            atol=1e-14,
        )
        np.testing.assert_array_equal(impl(5, 0.0, 0), np.eye(6))


class TestCircTaylorTransformMatrix:
    """Transport of ``(d1, ..., d5)`` along a circular orbit.

    The convention is ``state_new = L @ state``, column vectors, unlike the row vectors
    of `poly_taylor_transform_matrix`. A circular orbit obeys ``d4 = -omega**2 d2`` and
    ``d5 = -omega**2 d3``, and the matrix imposes that: at ``delta_t = 0`` it is a
    projection onto that manifold, not the identity.
    """

    P_ORB, AMP, PHASE, VEL = 2.0, 1.3, 0.4, 0.7

    def _state(self, t: float) -> np.ndarray:
        """Return derivatives 1..5 of ``AMP sin(omega t + PHASE) + VEL t``."""
        w = 2 * np.pi / self.P_ORB
        arg = w * t + self.PHASE
        return np.array(
            [
                self.AMP * w * np.cos(arg) + self.VEL,
                -self.AMP * w**2 * np.sin(arg),
                -self.AMP * w**3 * np.cos(arg),
                self.AMP * w**4 * np.sin(arg),
                self.AMP * w**5 * np.cos(arg),
            ]
        )

    @pytest.mark.parametrize("impl", jit_variants(maths.circ_taylor_transform_matrix))
    @pytest.mark.parametrize("delta_t", [-0.6, 0.0, 0.37, 5.1])
    def test_transports_orbit_state(self, impl, delta_t: float) -> None:
        t0 = 0.2
        np.testing.assert_allclose(
            impl(delta_t, self.P_ORB, 0) @ self._state(t0),
            self._state(t0 + delta_t),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("impl", jit_variants(maths.circ_taylor_transform_matrix))
    def test_group_law_and_projection(self, impl) -> None:
        a, b = 0.37, 1.21
        np.testing.assert_allclose(
            impl(b, self.P_ORB, 0) @ impl(a, self.P_ORB, 0),
            impl(a + b, self.P_ORB, 0),
            atol=1e-12,
        )
        l0 = impl(0.0, self.P_ORB, 0)
        np.testing.assert_allclose(l0 @ l0, l0, atol=1e-12)

    @pytest.mark.parametrize("impl", jit_variants(maths.circ_taylor_transform_matrix))
    def test_descending_is_reversed_ascending(self, impl) -> None:
        np.testing.assert_array_equal(
            impl(0.37, self.P_ORB, 1),
            impl(0.37, self.P_ORB, 0)[::-1, ::-1],
        )

    @pytest.mark.parametrize("impl", jit_variants(maths.circ_taylor_transform_matrix_n))
    def test_omega_variant_matches(self, impl) -> None:
        omega = 2 * np.pi / self.P_ORB
        np.testing.assert_allclose(
            impl(0.37, omega),
            maths.circ_taylor_transform_matrix(0.37, self.P_ORB, 1),
            rtol=1e-14,
        )


class TestConnectionCoefficients:
    """``x^k = sum_m S[k, m] T_m(x)`` and ``T_k(x) = sum_m R[k, m] x^m``."""

    K_MAX = 7

    @pytest.mark.parametrize("impl", jit_variants(maths.compute_connection_matrix_s))
    def test_s_matches_numpy(self, impl) -> None:
        s_mat = impl(self.K_MAX)
        for k in range(self.K_MAX + 1):
            expected = polynomial.chebyshev.poly2cheb(
                np.eye(self.K_MAX + 1)[k][: k + 1]
            )
            np.testing.assert_allclose(s_mat[k, : k + 1], expected, atol=1e-14)
            np.testing.assert_array_equal(s_mat[k, k + 1 :], 0.0)

    @pytest.mark.parametrize("impl", jit_variants(maths.compute_connection_matrix_r))
    def test_r_matches_numpy(self, impl) -> None:
        r_mat = impl(self.K_MAX)
        for k in range(self.K_MAX + 1):
            expected = polynomial.chebyshev.cheb2poly(
                np.eye(self.K_MAX + 1)[k][: k + 1]
            )
            np.testing.assert_allclose(r_mat[k, : k + 1], expected, atol=1e-12)

    @pytest.mark.parametrize(
        ("impl", "matrix"),
        [
            (impl.values[0], maths.compute_connection_matrix_s)
            for impl in jit_variants(maths.compute_connection_coefficient_s)
        ]
        + [
            (impl.values[0], maths.compute_connection_matrix_r)
            for impl in jit_variants(maths.compute_connection_coefficient_r)
        ],
        ids=["s-compiled", "s-py_func", "r-compiled", "r-py_func"],
    )
    def test_scalar_matches_matrix(self, impl, matrix) -> None:
        mat = matrix(self.K_MAX)
        for k in range(self.K_MAX + 1):
            for m in range(k + 1):
                assert impl(k, m) == mat[k, m], (k, m)

    def test_s_and_r_are_inverse(self) -> None:
        s_mat = maths.compute_connection_matrix_s(self.K_MAX)
        r_mat = maths.compute_connection_matrix_r(self.K_MAX)
        np.testing.assert_allclose(s_mat @ r_mat, np.eye(self.K_MAX + 1), atol=1e-12)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(maths.compute_connection_coefficient_s)
        + jit_variants(maths.compute_connection_coefficient_r),
    )
    @pytest.mark.parametrize(("k", "m"), [(-1, 0), (2, -1), (2, 3), (4, 1)])
    def test_zero_off_support(self, impl, k: int, m: int) -> None:
        assert impl(k, m) == 0.0

    @pytest.mark.parametrize("impl", jit_variants(maths.compute_connection_matrix_s))
    def test_s_rejects_negative_order(self, impl) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            impl(-1)


class TestPolyChebyshevTransformMatrix:
    """``b = a @ C`` re-expands a Chebyshev series from domain 1 onto domain 2."""

    DOM1, DOM2, DOM3 = (0.0, 2.0), (0.5, 1.0), (0.7, 0.4)

    @pytest.mark.parametrize(
        "impl", jit_variants(maths.poly_chebyshev_transform_matrix)
    )
    def test_same_function_on_new_domain(self, impl) -> None:
        (tc1, ts1), (tc2, ts2) = self.DOM1, self.DOM2
        a = np.array([0.2, -0.5, 1.1, 0.3, -0.8])
        b = a @ impl(4, tc1, ts1, tc2, ts2, 0)
        t = np.linspace(tc2 - ts2, tc2 + ts2, 13)
        np.testing.assert_allclose(
            polynomial.Chebyshev(b)((t - tc2) / ts2),
            polynomial.Chebyshev(a)((t - tc1) / ts1),
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "impl", jit_variants(maths.poly_chebyshev_transform_matrix)
    )
    def test_group_law_and_identity(self, impl) -> None:
        c12 = impl(4, *self.DOM1, *self.DOM2, 0)
        c23 = impl(4, *self.DOM2, *self.DOM3, 0)
        c13 = impl(4, *self.DOM1, *self.DOM3, 0)
        np.testing.assert_allclose(c12 @ c23, c13, atol=1e-12)
        np.testing.assert_allclose(
            impl(4, *self.DOM2, *self.DOM2, 0),
            np.eye(5),
            atol=1e-14,
        )

    @pytest.mark.parametrize(
        "impl", jit_variants(maths.poly_chebyshev_transform_matrix)
    )
    def test_descending_is_reversed_ascending(self, impl) -> None:
        np.testing.assert_array_equal(
            impl(4, *self.DOM1, *self.DOM2, 1),
            impl(4, *self.DOM1, *self.DOM2, 0)[::-1, ::-1],
        )

    @pytest.mark.parametrize(
        "impl", jit_variants(maths.poly_chebyshev_transform_matrix)
    )
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((-1, 0, 1, 0, 1), "non-negative"),
            ((2, 0, 0, 0, 1), "positive"),
            ((2, 0, 1, 0, -1), "positive"),
        ],
    )
    def test_rejects_bad_input(self, impl, args, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            impl(*args)

    @pytest.mark.parametrize(
        "impl",
        jit_variants(maths.compute_transformation_coefficient_c),
    )
    def test_coefficient_zero_above_diagonal(self, impl) -> None:
        assert impl(2, 3, 0.5, 0.1) == 0.0

    @pytest.mark.parametrize(
        "impl",
        jit_variants(maths.compute_transformation_coefficient_c),
    )
    def test_coefficient_matches_matrix(self, impl) -> None:
        (tc1, ts1), (tc2, ts2) = self.DOM1, self.DOM2
        mat = maths.poly_chebyshev_transform_matrix(4, tc1, ts1, tc2, ts2, 0)
        p, q = ts2 / ts1, (tc2 - tc1) / ts1
        for n in range(5):
            for k in range(5):
                np.testing.assert_allclose(impl(n, k, p, q), mat[n, k], atol=1e-14)


class TestPowerSeriesTable:
    @pytest.mark.parametrize(("order_max", "n_derivs"), [(0, 0), (3, 1), (6, 4)])
    def test_derivatives_of_scaled_monomials(
        self,
        order_max: int,
        n_derivs: int,
    ) -> None:
        """Row ``[d, n]`` holds the d-th derivative of ``x**n / n!``.

        That is ``x**(n - d) / (n - d)!`` for ``d <= n``, and zero otherwise.
        """
        tab = maths.gen_power_series_table_np(order_max, n_derivs)
        assert tab.shape == (n_derivs + 1, order_max + 1, order_max + 1)
        for d in range(n_derivs + 1):
            for n in range(order_max + 1):
                expected = np.zeros(order_max + 1)
                if d <= n:
                    expected[n - d] = 1 / math.factorial(n - d)
                np.testing.assert_allclose(tab[d, n], expected, rtol=1e-14)


class TestFindSmallPolys:
    @pytest.mark.parametrize("impl", jit_variants(maths.find_small_polys))
    @pytest.mark.parametrize(("degree", "error_bound"), [(2, 1), (3, 1), (3, 2)])
    def test_returned_polys_meet_the_bound(
        self,
        impl,
        degree: int,
        error_bound: int,
    ) -> None:
        good, point_volume, volume_factor = impl(degree, error_bound)
        x = np.linspace(-1, 1, 128)
        values = polynomial.polynomial.polyval(x, good.T, tensor=True)
        violations = (np.abs(values) > error_bound).sum(axis=-1)
        assert len(good) > 0
        assert (violations < 12).all()
        n_grid = (2 * (degree - 1) * 4) ** degree
        np.testing.assert_allclose(point_volume * n_grid, (2 * (degree - 1)) ** degree)
        np.testing.assert_allclose(volume_factor, point_volume * len(good) / 2**degree)

    def test_paths_agree(self) -> None:
        compiled = maths.find_small_polys(3, 1)[0]
        python = python_impl(maths.find_small_polys)(3, 1)[0]
        np.testing.assert_allclose(compiled, python, rtol=0, atol=1e-12)
