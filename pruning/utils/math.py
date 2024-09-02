from __future__ import annotations

import ctypes

import numpy as np
from numba import njit, vectorize
from numba.extending import get_cython_function_address
from numpy import polynomial
from scipy import stats

addr = get_cython_function_address("scipy.special.cython_special", "binom")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
cbinom_func = functype(addr)


@vectorize("f8(f8, f8)")
def nbinom(xx: float, yy: float) -> float:
    return cbinom_func(xx, yy)


def fact_factory(n_tab_out: int = 100) -> np.ufunc:
    fact_tab = np.ones(n_tab_out)

    @njit(cache=True)
    def _fact(num: int, n_tab: int = n_tab_out) -> int:
        if num < n_tab:
            return fact_tab[num]
        ret = 1
        for nn in range(1, num + 1):
            ret *= nn
        return ret

    for ii in range(n_tab_out):
        fact_tab[ii] = _fact(ii, 0)

    @vectorize(cache=True)
    def fact_vec(num: int) -> int:
        return _fact(num)

    return fact_vec


fact = fact_factory(120)


def gen_norm_isf_table(max_minus_logsf: float, minus_logsf_res: float) -> np.ndarray:
    x_arr = np.arange(0, max_minus_logsf, minus_logsf_res)
    return stats.norm.isf(np.exp(-x_arr))


def gen_chi_sq_minus_logsf_table(
    df_max: int,
    chi_sq_max: float,
    chi_sq_res: float,
) -> np.ndarray:
    x_arr = np.arange(0, chi_sq_max, chi_sq_res)
    table = np.zeros((df_max + 1, len(x_arr)))
    for i in range(1, df_max + 1):
        table[i] = -stats.chi2.logsf(x_arr, i)
    return table


chi_sq_res = 0.5
chi_sq_max = 300
max_minus_logsf = 400
minus_logsf_res = 0.1
chi_sq_minus_logsf_table = gen_chi_sq_minus_logsf_table(64, chi_sq_max, chi_sq_res)
norm_isf_table = gen_norm_isf_table(max_minus_logsf, minus_logsf_res)


@njit(cache=True)
def norm_isf_func(minus_logsf: float) -> float:
    pos = minus_logsf / minus_logsf_res
    frac_pos = pos % 1
    if minus_logsf < max_minus_logsf:
        return (
            norm_isf_table[int(pos)] * (1 - frac_pos)
            + norm_isf_table[int(pos) + 1] * frac_pos
        )
    return norm_isf_table[-1] * (minus_logsf / max_minus_logsf) ** 0.5


@njit(cache=True)
def chi_sq_minus_logsf_func(chi_sq_score: float, df: int) -> float:
    tab_pos = chi_sq_score / chi_sq_res
    frac_pos = tab_pos % 1
    if chi_sq_score < chi_sq_max:
        return (
            chi_sq_minus_logsf_table[df, int(tab_pos)] * (1 - frac_pos)
            + chi_sq_minus_logsf_table[df, int(tab_pos) + 1] * frac_pos
        )
    return chi_sq_minus_logsf_table[df, -1] * chi_sq_score / chi_sq_max


def gen_chebyshev_polys_table_np(order: int, n_derivs: int) -> np.ndarray:
    tab = np.zeros((n_derivs + 1, order + 1, order + 1))
    basis = [polynomial.chebyshev.Chebyshev.basis(i) for i in range(order + 1)]

    for ideriv in range(n_derivs + 1):
        for iorder, poly in enumerate(basis):
            poly_coeffs = polynomial.chebyshev.cheb2poly(poly.deriv(ideriv).coef)
            tab[ideriv, iorder, : len(poly_coeffs)] = poly_coeffs
    return tab


@njit(cache=True, fastmath=True)
def gen_chebyshev_polys_table(order_max: int, n_derivs: int) -> np.ndarray:
    """Generate table of Chebyshev polynomials of the first kind and their derivatives.

    Parameters
    ----------
    order_max : int
        The maximum order of the polynomials (T_0, T_1, ..., T_order_max)
    n_derivs : int
        The number of derivatives to generate for each polynomial (0th derivative
        is the polynomial itself)

    Returns
    -------
    np.ndarray
        A 3D array of shape (n_derivs + 1, order_max + 1, order_max + 1) containing the
        coefficients of the polynomials.
    """
    tab = np.zeros((n_derivs + 1, order_max + 1, order_max + 1), dtype=np.float32)
    tab[0, 0, 0] = 1.0
    tab[0, 1, 1] = 1.0

    for jorder in range(2, order_max + 1):
        tab[0, jorder] = 2 * np.roll(tab[0, jorder - 1], 1) - tab[0, jorder - 2]

    factor = np.arange(1, order_max + 2, dtype=np.float32)
    for ideriv in range(1, n_derivs + 1):
        for jorder in range(1, order_max + 1):
            tab[ideriv, jorder] = np.roll(tab[ideriv - 1, jorder], -1) * factor
            tab[ideriv, jorder, -1] = 0

    return tab


@njit(fastmath=True)
def generalized_cheb_pols(poly_order: int, t0: float, scale: float) -> np.ndarray:
    """Generate table of generalized Chebyshev polynomials.

    Parameters
    ----------
    poly_order : int
        The maximum order of the polynomials (upto T_poly_order)
    t0 : float
        Shifted origin of the polynomials.
    scale : float
        Scaling factor for the polynomials.

    Returns
    -------
    np.ndarray
        A 2D array of shape (poly_order + 1, poly_order + 1) containing the coefficients
        of the polynomials.

    Notes
    -----
    The generalized Chebyshev polynomials are defined as:
    T*_n(y) = T_n((x - t0) / scale)
    where :math: y in [-1, 1] and x in [t0 - scale, t0 + scale].
    """
    cheb_pols = gen_chebyshev_polys_table(poly_order, 0)[0]

    # Shift the origin to t0
    cheb_pols_shifted = np.zeros_like(cheb_pols)
    for iorder in range(poly_order + 1):
        iterms = np.arange(iorder + 1, dtype=np.float32)
        shifted = nbinom(iorder, iterms) * (-t0 / scale) ** (iorder - iterms)
        cheb_pols_shifted[iorder, : len(shifted)] = shifted
    pols = np.dot(cheb_pols, cheb_pols_shifted)
    # scale the polynomials
    pols *= scale ** (-np.arange(poly_order + 1, dtype=np.float32))
    return pols


def gen_power_series_table_np(order_max: int, n_derivs: int) -> np.ndarray:
    tab = np.zeros((n_derivs + 1, order_max + 1, order_max + 1))

    for order in range(order_max + 1):
        # Create power series polynomial
        coeffs = np.zeros(order + 1)
        coeffs[order] = 1 / np.math.factorial(order)
        poly = polynomial.Polynomial(coeffs)

        for deriv in range(n_derivs + 1):
            # Get coefficients of the derivative
            deriv_coeffs = poly.deriv(deriv).coef
            # Pad with zeros if necessary
            tab[deriv, order, : len(deriv_coeffs)] = deriv_coeffs

    return tab
