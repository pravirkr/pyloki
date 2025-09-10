import numpy as np
import pytest
from numpy import polynomial
from scipy import special, stats

from pyloki.utils import maths


class TestMaths:
    rng = np.random.default_rng()

    @pytest.mark.parametrize(
        ("n", "k"),
        [(5, 2), (6, 7), (2, 3), (20, 12), (5, 0), (5, 5), (0, 0)],
    )
    def test_nbinom(self, n: int, k: int) -> None:
        np.testing.assert_almost_equal(maths.nbinom(n, k), special.binom(n, k))

    @pytest.mark.parametrize("n", [0, 1, 5, 20, np.array([0, 1, 5, 20])])
    def test_fact(self, n: int | np.ndarray) -> None:
        np.testing.assert_almost_equal(maths.fact(n), special.factorial(n))

    def test_norm_isf_func(self) -> None:
        minus_logsf = self.rng.uniform(0, 10)
        expected = stats.norm.isf(np.exp(-minus_logsf))
        np.testing.assert_almost_equal(
            maths.norm_isf_func(minus_logsf),
            expected,
            decimal=2,
        )

    def test_chi_sq_minus_logsf_func(self) -> None:
        df = 5  # Example degrees of freedom
        chi_sq = self.rng.uniform(0, 10)
        expected = -stats.chi2.logsf(chi_sq, df)
        np.testing.assert_almost_equal(
            maths.chi_sq_minus_logsf_func(chi_sq, df),
            expected,
            decimal=2,
        )

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
    rng = np.random.default_rng()

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
        alpha = maths.taylor_to_cheby(d_vec, t_s)
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
        d = maths.cheby_to_taylor(alpha_vec, t_s)
        np.testing.assert_almost_equal(d, d_expected, decimal=12)

    @pytest.mark.parametrize("k_max", [2, 4, 6])
    def test_roundtrip_identity(self, k_max: int) -> None:
        d_vec = self.rng.random(k_max + 1)
        t_s = 1.5
        alpha = maths.taylor_to_cheby(d_vec, t_s)
        d_reconstructed = maths.cheby_to_taylor(alpha, t_s)
        np.testing.assert_almost_equal(d_vec, d_reconstructed, decimal=12)

    def test_polynomial_evaluation(self) -> None:
        d_vec = np.array([0.5, 2.3, 1500.0, 1e6, 1e4])
        k_max = len(d_vec) - 1
        t_c, t_s = 4.2, 2.6
        alpha_vec = maths.taylor_to_cheby(d_vec, t_s)
        t_test = np.linspace(t_c - t_s, t_c + t_s, 11)
        x = (t_test - t_c) / t_s
        k_range = np.arange(k_max + 1)
        c_power = d_vec[::-1] / maths.fact(k_range)
        taylor_poly = polynomial.Polynomial(c_power)
        val_taylor = taylor_poly(t_test - t_c)
        cheby_poly = polynomial.Chebyshev(alpha_vec[::-1], domain=[-1, 1])
        val_cheby = cheby_poly(x)
        np.testing.assert_almost_equal(val_taylor, val_cheby, decimal=8)
