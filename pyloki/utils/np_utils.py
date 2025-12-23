from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from collections.abc import Callable


@njit(cache=True, fastmath=True)
def find_nearest_sorted_idx(
    array: np.ndarray,
    value: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> int:
    """Find the index of the closest value in a sorted array.

    In case of a tie, the index of the smaller value is returned.
    Behaviour is undefined if the array is not sorted.

    Parameters
    ----------
    array : np.ndarray
        Sorted array. Must contain at least one element.
    value : float
        Value to search for.
    rtol : float, optional
        Relative tolerance for floating-point comparison, by default 1e-5.
    atol : float, optional
        Absolute tolerance for floating-point comparison, by default 1e-8.

    Returns
    -------
    int
        Index of the closest value in the array.

    Raises
    ------
    ValueError
        If the array is empty
    """
    if len(array) == 0:
        msg = "Array must not be empty"
        raise ValueError(msg)
    idx = int(np.searchsorted(array, value, side="left"))
    if idx > 0:
        diff_prev = abs(value - array[idx - 1])
        diff_curr = abs(value - array[idx]) if idx < len(array) else np.inf
        if diff_prev <= diff_curr * (1 + rtol) + atol:
            return idx - 1
    return idx


@njit(cache=True, fastmath=True)
def find_nearest_sorted_idx_vect(
    array: np.ndarray,
    values: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> np.ndarray:
    """Find the indices of the closest values in a sorted array (vectorized)."""
    if len(array) == 0:
        msg = "Array must not be empty"
        raise ValueError(msg)

    n = len(array)
    idxs = np.searchsorted(array, values, side="left")
    # Prepare output array
    out = np.empty(len(values), dtype=np.int64)
    for i in range(len(values)):
        idx = idxs[i]
        if idx > 0:
            diff_prev = abs(values[i] - array[idx - 1])
            diff_curr = abs(values[i] - array[idx]) if idx < n else np.inf
            if diff_prev <= diff_curr * (1 + rtol) + atol:
                out[i] = idx - 1
            else:
                out[i] = idx
        else:
            out[i] = idx
    return out


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


@njit(cache=True, nogil=True, fastmath=True)
def nb_roll(
    arr: np.ndarray,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Roll array elements along a given axis.

    Implemented in rocket-fft. This function is a njit-compiled wrapper
    around np.roll.

    Parameters
    ----------
    arr : ndarray
        Input array
    shift : int | tuple[int, ...]
        Number of bins to shift
    axis : int | tuple[int, ...] | None, optional
        Axis or axes along which to roll, by default None

    Returns
    -------
    np.ndarray
        Rolled array with the same shape as `arr`
    """
    return np.roll(arr, shift, axis)


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


@njit(cache=True, fastmath=True)
def cartesian_prod_padded(
    padded_arrays: np.ndarray,
    actual_counts: np.ndarray,
    n_batch: int,
    nparams: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian product of padded arrays with actual counts.

    Parameters
    ----------
    padded_arrays : np.ndarray
        Padded arrays. Shape: (n_batch, nparams, MAX_BRANCH_VALS).
    actual_counts : np.ndarray
        Actual counts. Shape: (n_batch, nparams).
    n_batch : int
        Number of batches.
    nparams : int
        Number of parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The Cartesian product of the padded values with the actual counts.
    """
    items_per_batch = np.zeros(n_batch, dtype=np.int64)

    # First pass: Calculate total items and items per batch
    total_items = 0
    for i in range(n_batch):
        count_i = 1
        for j in range(nparams):
            count_i *= actual_counts[i, j]
        items_per_batch[i] = count_i
        total_items += count_i
    cart_prod = np.empty((total_items, nparams), dtype=padded_arrays.dtype)
    origins = np.empty(total_items, dtype=np.int64)

    # Second pass: Generate combinations and fill arrays
    current_row_idx = 0
    for i in range(n_batch):
        num_items_i = items_per_batch[i]
        origins[current_row_idx : current_row_idx + num_items_i] = i
        # Generate Cartesian product for item 'i'
        indices = np.zeros(nparams, dtype=np.int64)
        item_row_idx = 0
        while item_row_idx < num_items_i:
            for k in range(nparams):
                param_idx = indices[k]
                cart_prod[current_row_idx + item_row_idx, k] = padded_arrays[
                    i,
                    k,
                    param_idx,
                ]
            item_row_idx += 1

            # Odometer increment
            for k in range(nparams - 1, -1, -1):
                max_idx_k = actual_counts[i, k] - 1
                if indices[k] < max_idx_k:
                    indices[k] += 1
                    break  # Move to next combination
                indices[k] = 0  # Reset current param index and carry over
        current_row_idx += num_items_i
    return cart_prod, origins


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
    zero_arr = np.zeros((*arr.shape[:-1], padding_needed))
    return np.concatenate((arr, zero_arr), axis=-1)


@njit(cache=True)
def interpolate_missing(profile: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Interpolate missing values in a profile.

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


@njit(cache=True, fastmath=True)
def lstsq_weighted(
    design_matrix: np.ndarray,
    data: np.ndarray,
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform weighted least squares estimation.

    Parameters
    ----------
    design_matrix : np.ndarray
        The design matrix A.
    data : np.ndarray
        The observed data points.
    errors : np.ndarray
        The errors of the observed data points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - x_hat: The estimated parameters.
        - cov_x_hat: The covariance matrix of the estimated parameters.
        - phi_t_estimated: The estimated data points.
    """
    weights = 1.0 / (errors**2)
    w_sqrt = np.sqrt(weights)
    a_w = design_matrix * w_sqrt[:, np.newaxis]
    b_w = data * w_sqrt

    x_hat, _, rank, _ = np.linalg.lstsq(a_w, b_w, rcond=None)

    # Calculate the covariance matrix
    if rank == design_matrix.shape[1]:  # Full rank
        cov_x_hat = np.linalg.inv(design_matrix.T @ np.diag(weights) @ design_matrix)
    else:
        # Use pseudoinverse if not full rank
        cov_x_hat = np.linalg.pinv(design_matrix.T @ np.diag(weights) @ design_matrix)

    phi_t_estimated = design_matrix @ x_hat

    return x_hat, cov_x_hat, phi_t_estimated
