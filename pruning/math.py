from __future__ import annotations

import ctypes
from typing import Callable

import numpy as np
from numba import njit, vectorize
from numba.extending import get_cython_function_address
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


@njit(cache=True, fastmath=True)
def find_nearest_sorted_idx(array: np.ndarray, value: float) -> int:
    """Find the index of the closest value in a sorted array.

    Parameters
    ----------
    array : np.ndarray
        Sorted array
    value : float
        Value to search for

    Returns
    -------
    int
        Index of the closest value in the array
    """
    idx = int(np.searchsorted(array, value, side="left"))
    if idx > 0 and (
        idx == len(array) or abs(value - array[idx - 1]) <= abs(value - array[idx])
    ):
        return idx - 1
    return idx


@njit
def np_apply_along_axis(func1d: Callable, axis: int, arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        msg = "arr must be 2D"
        raise ValueError(msg)
    if axis not in {0, 1}:
        msg = "axis must be 0 or 1"
        raise ValueError(msg)
    if axis == 0:
        res_len = arr.shape[1]
        result = np.empty(res_len)
        for ii in range(res_len):
            result[ii] = func1d(arr[:, ii])
    else:
        res_len = arr.shape[0]
        result = np.empty(res_len)
        for jj in range(res_len):
            result[jj] = func1d(arr[jj, :])
    return result


@njit
def nb_max(array: np.ndarray, axis: int) -> np.ndarray:
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_mean(array: np.ndarray, axis: int) -> np.ndarray:
    return np_apply_along_axis(np.mean, axis, array)


@njit
def downsample_1d(array: np.ndarray, factor: int) -> np.ndarray:
    reshaped_ar = np.reshape(array, (array.size // factor, factor))
    return np_mean(reshaped_ar, 1)


@njit
def nb_roll2d(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll the 2D array along the second axis (axis=1).

    Parameters
    ----------
    arr : np.ndarray
        2D array to roll
    shift : int
        Number of bins to shift

    Returns
    -------
    np.ndarray
        Rolled array
    """
    axis = 1
    res = np.empty_like(arr)
    arr_size = arr.shape[axis]
    shift %= arr_size
    for irow in range(arr.shape[0]):
        res[irow, shift:] = arr[irow, : arr_size - shift]
        res[irow, :shift] = arr[irow, arr_size - shift :]
    return res


@njit
def nb_roll3d(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll the 3D array along the last axis (axis=-1).

    Parameters
    ----------
    arr : np.ndarray
        3D array to roll
    shift : int
        Number of bins to shift

    Returns
    -------
    np.ndarray
        Rolled array
    """
    axis = -1
    res = np.empty_like(arr)
    arr_size = arr.shape[axis]
    shift %= arr_size
    for irow in range(arr.shape[0]):
        res[irow, :, shift:] = arr[irow, :, : arr_size - shift]
        res[irow, :, :shift] = arr[irow, :, arr_size - shift :]
    return res


@njit
def cartesian_prod(arrays: np.ndarray) -> np.ndarray:
    nn = 1
    for array in arrays:
        nn *= array.size
    out = np.zeros((nn, len(arrays)))

    for iarr, arr in enumerate(arrays):
        mm = nn // arr.size
        out[:nn, iarr] = np.repeat(arr, mm)
        nn //= arr.size

    nn = arrays[-1].size
    for kk in range(len(arrays) - 2, -1, -1):
        nn *= arrays[kk].size
        mm = nn // arrays[kk].size
        for jj in range(1, arrays[kk].size):
            out[jj * mm : (jj + 1) * mm, kk + 1 :] = out[0:mm, kk + 1 :]
    return out


@njit
def numba_bf_row_vector_unique(arr: np.ndarray) -> np.ndarray:
    ret = np.zeros(arr.shape, arr.dtype)
    ret[0] = arr[0]
    ind = 1
    for ii in range(1, len(arr)):
        good = True
        for ind_vec in range(ind):
            if numba_bypass_all_close(ret[ind_vec], arr[ii]):
                good = False
        if good:
            ret[ind] = arr[ii]
            ind += 1
    return ret[:ind]


@njit
def numba_bypass_all_close(
    arr1: np.ndarray,
    arr2: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    return bool(np.all(np.abs(arr1 - arr2) < tol))
