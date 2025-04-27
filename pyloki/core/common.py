from __future__ import annotations

import numpy as np
from numba import njit, prange, types

from pyloki.utils import np_utils, psr_utils
from pyloki.utils.misc import C_VAL


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
    phase_map = psr_utils.get_phase_idx(proper_time, freq, nbins, 0)
    segment_len = nsamples // nsegments
    nsamps_fold = nsegments * segment_len
    segments_idxs = np.arange(nsamps_fold) // segment_len

    fold = np.zeros((nsegments, 2, nbins), dtype=np.float32)
    for isamp in range(nsamps_fold):
        fold[segments_idxs[isamp], 0, phase_map[isamp]] += ts_e[isamp]
        fold[segments_idxs[isamp], 1, phase_map[isamp]] += ts_v[isamp]
    return fold


@njit
def resample(ts_e: np.ndarray, ts_v: np.ndarray, tsamp: float, accel: float) -> tuple:
    nsamps = len(ts_e) - 1 if accel > 0 else len(ts_e)
    ts_e_resamp = np.zeros_like(ts_e)
    ts_v_resamp = np.zeros_like(ts_v)

    partial_calc = (accel * tsamp) / (2 * C_VAL)
    tot_drift = partial_calc * (nsamps // 2) ** 2
    last_bin = 0
    for isamp in range(nsamps):
        index = int(isamp + partial_calc * (isamp - nsamps // 2) ** 2 - tot_drift)
        ts_e_resamp[index] = ts_e[isamp]
        ts_v_resamp[index] = ts_v[isamp]
        if index - last_bin > 1:
            ts_e_resamp[index - 1] = ts_e[isamp]
            ts_v_resamp[index - 1] = ts_v[isamp]
        last_bin = index
    return ts_e_resamp, ts_v_resamp


@njit(
    ["f4[:,:,:,:](f4[:],f4[:],f8[:],i8,i8,f8,f8)"],
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
        phase_map[ifreq] = psr_utils.get_phase_idx(
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
