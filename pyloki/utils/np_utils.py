from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit


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


# Note: no cache=True here, as it is not supported by numba
@njit(fastmath=True)
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


@njit(fastmath=True)
def nb_max(array: np.ndarray, axis: int) -> np.ndarray:
    return np_apply_along_axis(np.max, axis, array)


@njit(cache=True, fastmath=True)
def np_mean(array: np.ndarray, axis: int) -> np.ndarray:
    return np_apply_along_axis(np.mean, axis, array)


@njit(cache=True, fastmath=True)
def downsample_1d(array: np.ndarray, factor: int) -> np.ndarray:
    reshaped_ar = np.reshape(array, (array.size // factor, factor))
    return np_mean(reshaped_ar, 1)


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
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


def cartesian_prod_np(arr_list: list[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*arr_list, indexing="ij")
    flattened_mesh = [arr.ravel() for arr in mesh]
    return np.vstack(flattened_mesh).T


def cartesian_prod_st(arr_list: list[np.ndarray]) -> np.ndarray:
    """Twice as fast as cartesian_prod_np."""
    la = len(arr_list)
    dtype = np.result_type(*arr_list)
    cart = np.empty([la] + [len(arr) for arr in arr_list], dtype=dtype)
    for iarr, arr in enumerate(np.ix_(*arr_list)):
        cart[iarr, ...] = arr
    return cart.reshape(la, -1).T


@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def numba_bypass_all_close(
    arr1: np.ndarray,
    arr2: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    return bool(np.all(np.abs(arr1 - arr2) < tol))


def pad_with_inf(param_list: list[np.ndarray]) -> np.ndarray:
    """Pad a list of arrays with inf to make them all the same length.

    Parameters
    ----------
    param_list : list[np.ndarray]
        List of arrays to pad.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    maxlen = np.max(list(map(len, param_list)))
    output = np.zeros([len(param_list), maxlen])
    output += np.inf
    for iarr, arr in enumerate(param_list):
        output[iarr][: len(arr)] = arr
    return output


@njit(cache=True, fastmath=True)
def cpadpow2(arr: np.ndarray) -> np.ndarray:
    """Circularly pad the last dimension to power of 2.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    nbins = arr.shape[-1]
    padded_length = 2 ** int(np.ceil(np.log2(nbins)))
    padding_needed = padded_length - nbins
    return np.concatenate((arr, arr[..., :padding_needed]), axis=-1)


@njit(cache=True, fastmath=True)
def cpad2len(arr: np.ndarray, size: int) -> np.ndarray:
    """Circularly pad the last dimension of ndarray 'arr' to given length with zeros."""
    padding_needed = size - arr.shape[-1]
    zero_arr = np.zeros(arr.shape[:-1] + (padding_needed,))
    return np.concatenate((arr, zero_arr), axis=-1)


@njit(cache=True)
def interpolate_missing(profile: np.ndarray, count: np.ndarray) -> np.ndarray:
    """
    Interpolate missing values in a profile.

    Parameters
    ----------
    profile : np.ndarray
        Profile to interpolate missing values
    count : np.ndarray
        Array of bin counts

    """
    empty_idx = np.where(count == 0)[0]
    non_empty_idx = np.where(count > 0)[0]
    if len(non_empty_idx) != 0:
        profile[empty_idx] = np.interp(empty_idx, non_empty_idx, profile[non_empty_idx])
    return profile
