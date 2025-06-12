from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, types

from pyloki.utils import np_utils, psr_utils

if TYPE_CHECKING:
    from collections.abc import Callable


@njit(cache=True, fastmath=True)
def brutefold_single(
    ts: np.ndarray,
    proper_time: np.ndarray,
    freq: float,
    nsegments: int,
    nbins: int,
) -> np.ndarray:
    """Fold a time series for a given set of frequencies.

    Parameters
    ----------
    ts : np.ndarray
        Time series
    proper_time : np.ndarray
        Proper time of the signal in time units
    freq : float
        Frequency to fold the time series
    nsegments : int
        Number of segments to fold
    nbins : int
        Number of bins in the folded profile

    Returns
    -------
    np.ndarray
        Folded time series with shape (nsegments, nfreqs, 2, nbins)

    Raises
    ------
    ValueError
        if freq_arr is empty or if the nsamples is not a multiple of segment_len
    """
    nsamples = len(ts)
    if len(proper_time) != nsamples:
        msg = "ts and proper_time must have the same length."
        raise ValueError(msg)
    phase_map = psr_utils.get_phase_idx_int(proper_time, freq, nbins, 0)
    segment_len = nsamples // nsegments
    nsamps_fold = nsegments * segment_len
    segments_idxs = np.arange(nsamps_fold) // segment_len
    fold = np.zeros((nsegments, nbins), dtype=np.float32)
    for isamp in range(nsamps_fold):
        fold[segments_idxs[isamp], phase_map[isamp]] += ts[isamp]
    return fold


@njit(["f4[:,:,:](f4[:],f4[:],f8[:],f8,i8,i8)"], cache=True, fastmath=True)
def brutefold(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    proper_time: np.ndarray,
    freq: float,
    nsegments: int,
    nbins: int,
) -> np.ndarray:
    """Fold a time series for a given set of frequencies.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series signal (intensity)
    ts_v : np.ndarray
        Time series variance
    proper_time : np.ndarray
        Proper time of the signal in time units
    freq : float
        Frequency to fold the time series
    nsegments : int
        Number of segments to fold
    nbins : int
        Number of bins in the folded profile

    Returns
    -------
    np.ndarray
        Folded time series with shape (nsegments, nfreqs, 2, nbins)

    Raises
    ------
    ValueError
        if freq_arr is empty or if the nsamples is not a multiple of segment_len
    """
    nsamples = len(ts_e)
    if len(ts_v) != nsamples or len(proper_time) != nsamples:
        msg = "ts_e, ts_v, and proper_time must have the same length."
        raise ValueError(msg)
    phase_map = psr_utils.get_phase_idx_int(proper_time, freq, nbins, 0)
    segment_len = nsamples // nsegments
    nsamps_fold = nsegments * segment_len
    segments_idxs = np.arange(nsamps_fold) // segment_len

    fold = np.zeros((nsegments, 2, nbins), dtype=np.float32)
    for isamp in range(nsamps_fold):
        fold[segments_idxs[isamp], 0, phase_map[isamp]] += ts_e[isamp]
        fold[segments_idxs[isamp], 1, phase_map[isamp]] += ts_v[isamp]
    return fold


@njit(
    ["f4[:,:,:,::1](f4[::1],f4[::1],f8[::1],i8,i8,f8,f8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def brutefold_start(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    segment_len: int,
    nbins: int,
    tsamp: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Fold a time series for a given set of frequencies.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series signal (intensity).
    ts_v : np.ndarray
        Time series variance.
    freq_arr : np.ndarray
        Array of frequencies to fold the time series.
    segment_len : int
        Length of the segment (in samples) to fold.
    nbins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the time series.
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0).

    Returns
    -------
    np.ndarray
        Folded time series with shape (nsegments, nfreqs, 2, nbins).
    """
    nfreqs = len(freq_arr)
    nsamples = len(ts_e)
    nsegments = int(np.ceil(nsamples / segment_len))
    proper_time = np.arange(segment_len) * tsamp - t_ref
    phase_map = np.zeros((nfreqs, segment_len), dtype=np.int32)
    for ifreq in range(nfreqs):
        phase_map[ifreq] = psr_utils.get_phase_idx_int(
            proper_time,
            freq_arr[ifreq],
            nbins,
            0,
        )

    fold = np.zeros((nsegments, nfreqs, 2, nbins), dtype=np.float32)
    for iseg in prange(nsegments):
        segment_start = iseg * segment_len
        ts_e_seg = ts_e[segment_start : segment_start + segment_len]
        ts_v_seg = ts_v[segment_start : segment_start + segment_len]
        segment_len_actual = len(ts_e_seg)
        for ifreq in range(nfreqs):
            for isamp in range(segment_len_actual):
                iphase = phase_map[ifreq, isamp]
                fold[iseg, ifreq, 0, iphase] += ts_e_seg[isamp]
                fold[iseg, ifreq, 1, iphase] += ts_v_seg[isamp]
    return fold


@njit(cache=True, fastmath=True)
def get_leaves(param_arr: types.ListType, dparams: np.ndarray) -> np.ndarray:
    """Get the leaf parameter sets for pruning.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    param_cart = np_utils.cartesian_prod(param_arr)
    param_mat = np.expand_dims(param_cart, axis=2)
    dparams_set = np.broadcast_to(np.expand_dims(dparams, 1), param_mat.shape)
    return np.concatenate((param_mat, dparams_set), axis=2)


@njit(cache=True, fastmath=True)
def get_leaves_opt(
    param_arr: types.ListType,
    dparams: np.ndarray,
) -> np.ndarray:
    nparams = len(param_arr)
    shapes = np.empty(nparams, dtype=np.int64)
    for i in range(nparams):
        shapes[i] = len(param_arr[i])
    total_size = np.prod(shapes)
    leaves_taylor = np.empty((total_size, nparams, 2), dtype=np.float64)

    if total_size == 0:
        return leaves_taylor  # Return empty array if no leaves

    # Fill column 1 (dparams) - this is constant for each parameter across leaves
    for j in range(nparams):
        leaves_taylor[:, j, 1] = dparams[j]

    # Fill column 0 (parameter values) using Cartesian product logic
    # Similar logic to the optimized cartesian_prod, but fills directly
    elements_per_cycle = np.empty(nparams, dtype=np.int64)
    elements_in_block = total_size

    for i in range(nparams - 1, -1, -1):
        elements_in_block //= shapes[i]
        elements_per_cycle[i] = elements_in_block

    for i in range(total_size):
        for j in range(nparams):
            arr = param_arr[j]
            # Calculate index within the specific parameter's array
            idx = (i // elements_per_cycle[j]) % shapes[j]
            leaves_taylor[i, j, 0] = arr[idx]

    return leaves_taylor


@njit(cache=True, fastmath=True)
def load_folds_1d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Load the fold from the input structure (1D-case).

    Parameters
    ----------
    fold : np.ndarray
        Input fold structure with shape (nsegments, nfreqs, 2, nbins).
    iseg : int
        Index of the segment.
    param_idx : np.ndarray
        Index of the parameter with shape [ifreq].

    Returns
    -------
    np.ndarray
        Fold with shape (2, nbins).
    """
    return fold[iseg, param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_2d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Load the fold from the input structure (2D-case).

    Parameters
    ----------
    fold : np.ndarray
        Input fold structure with shape (nsegments, naccels, nfreqs, 2, nbins).
    iseg : int
        Index of the segment.
    param_idx : np.ndarray
        Index of the parameter with shape [iacc, ifreq].

    Returns
    -------
    np.ndarray
        Fold with shape (2, nbins).
    """
    return fold[iseg, param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_3d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsegments, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[iseg, param_idx[-3], param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_4d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsegments, nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[iseg, param_idx[-4], param_idx[-3], param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_prune_folds_1d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[freq_idx, j, k]
    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_2d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[accel_idx, freq_idx, j, k]

    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_3d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (njerks, naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[0, param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[0, accel_idx, freq_idx, j, k]
    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_4d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[0, 0, param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[0, 0, accel_idx, freq_idx, j, k]
    return result


def set_ffa_load_func(
    nparams: int,
) -> Callable[[np.ndarray, int, np.ndarray], np.ndarray]:
    """Set the appropriate load function based on the number of parameters.

    Parameters
    ----------
    nparams : int
        Number of search parameters (dimensions).

    Returns
    -------
    Callable[[np.ndarray, int, np.ndarray], np.ndarray]
        The appropriate load function for the given number of parameters.
    """
    nparams_to_load_func = {
        1: load_folds_1d,
        2: load_folds_2d,
        3: load_folds_3d,
        4: load_folds_4d,
    }
    return nparams_to_load_func[nparams]


def set_prune_load_func(
    nparams: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Set the appropriate load function for the pruning based on the number of parameters.

    Parameters
    ----------
    nparams : int
        Number of search parameters (dimensions).

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        The appropriate load function for the given number of parameters.
    """
    nparams_to_load_func = {
        1: load_prune_folds_1d,
        2: load_prune_folds_2d,
        3: load_prune_folds_3d,
        4: load_prune_folds_4d,
    }
    return nparams_to_load_func[nparams]


@njit(cache=True, fastmath=True)
def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
    return data0 + data1


@njit(cache=True, fastmath=True)
def pack(data: np.ndarray) -> np.ndarray:
    return data


@njit(cache=True, fastmath=True)
def shift(data: np.ndarray, phase_shift: int) -> np.ndarray:
    return np_utils.nb_roll(data, phase_shift, axis=-1)


@njit(cache=True, fastmath=True)
def shift_add(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    shift_tail: int,
    shift_head: int,
) -> np.ndarray:
    n_comps, n_cols = data_tail.shape
    res = np.empty((n_comps, n_cols), dtype=data_tail.dtype)
    shift_tail = shift_tail % n_cols
    shift_head = shift_head % n_cols
    for j in range(n_cols):
        idx1 = (j - shift_tail) % n_cols
        idx2 = (j - shift_head) % n_cols
        res[0, j] = data_tail[0, idx1] + data_head[0, idx2]
        res[1, j] = data_tail[1, idx1] + data_head[1, idx2]
    return res


@njit(cache=True, fastmath=True)
def shift_add_batch(
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, n_cols = segment_batch.shape
    res = np.empty((n_batch, n_comps, n_cols), dtype=segment_batch.dtype)
    for irow in range(n_batch):
        shift = shift_batch[irow] % n_cols
        fold_row = folds[isuggest_batch[irow]]
        src_idx = (-shift) % n_cols
        for j in range(n_cols):
            res[irow, 0, j] = fold_row[0, j] + segment_batch[irow, 0, src_idx]
            res[irow, 1, j] = fold_row[1, j] + segment_batch[irow, 1, src_idx]
            src_idx += 1
            if src_idx == n_cols:
                src_idx = 0
    return res


@njit(cache=True, fastmath=True)
def get_trans_matrix(
    coord_cur: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)


@njit(cache=True, fastmath=True)
def get_validation_params() -> tuple[np.ndarray, np.ndarray, float]:
    return np.array([1, 2, 3]), np.array([4, 5, 6]), 0.1
