import numpy as np
import pytest
from numpy import polynomial
from scipy import special, stats

from pyloki.utils import maths, transforms

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
