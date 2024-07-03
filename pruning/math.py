from __future__ import annotations

import ctypes

import numpy as np
from numba import njit, vectorize
from numba.extending import get_cython_function_address
from scipy import stats

addr = get_cython_function_address("scipy.special.cython_special", "binom")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
cbinom_func = functype(addr)


@vectorize("float64(float64, float64)")
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
