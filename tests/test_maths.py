import numpy as np
import pytest
from scipy import special, stats

from pyloki.utils import maths


class TestMaths:
    rng = np.random.default_rng()

    @pytest.mark.parametrize(
        ("n", "k"),
        [(5, 2), (6, 7), (2, 3), (25, 12), (5, 0), (5, 5), (0, 0)],
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
