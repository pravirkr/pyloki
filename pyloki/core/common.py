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


@njit(
    ["f4[:,:,:,::1](f4[::1],f4[::1],f8[::1],i8,i8,f8,f8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def brutefold_bucketed(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    segment_len: int,
    nbins: int,
    tsamp: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Fold a time series for a given set of frequencies using bucketed folding.

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
    nsamps = len(ts_e)
    nsegments = nsamps // segment_len
    if nsamps % segment_len != 0:
        msg = "The number of samples must be a multiple of the segment length."
        raise ValueError(msg)
    proper_time = np.arange(segment_len) * tsamp - t_ref

    total_buckets = nfreqs * nbins
    phase_map = np.empty(nfreqs * segment_len, dtype=np.uint32)
    bucket_indices = np.empty(nfreqs * segment_len, dtype=np.uint32)
    offsets = np.zeros(total_buckets + 1, dtype=np.uint32)

    # Build counts and buckets for efficient folding
    counts = np.zeros(total_buckets, dtype=np.int64)
    for ifreq in range(nfreqs):
        freq_offset_in = ifreq * segment_len
        for isamp in range(segment_len):
            proper_time = (isamp * tsamp) - t_ref
            iphase = psr_utils.get_phase_idx_int(
                proper_time,
                freq_arr[ifreq],
                nbins,
                0,
            )
            phase_map[freq_offset_in + isamp] = iphase
            bucket_idx = ifreq * nbins + iphase
            counts[bucket_idx] += 1

    for i in range(1, total_buckets + 1):
        offsets[i] = offsets[i - 1] + counts[i - 1]

    writers = offsets.copy()
    for ifreq in range(nfreqs):
        freq_offset_in = ifreq * segment_len
        for isamp in range(segment_len):
            iphase = phase_map[freq_offset_in + isamp]
            bucket_idx = ifreq * nbins + iphase
            bucket_indices[writers[bucket_idx]] = isamp
            writers[bucket_idx] += 1

    fold = np.zeros((nsegments, nfreqs, 2, nbins), dtype=np.float32)
    for iseg in prange(nsegments):
        start = iseg * segment_len
        ts_e_seg = ts_e[start : start + segment_len]
        ts_v_seg = ts_v[start : start + segment_len]
        for ifreq in range(nfreqs):
            base = ifreq * nbins
            for iphase in range(nbins):
                bucket_idx = base + iphase
                buck_start = offsets[bucket_idx]
                buck_end = offsets[bucket_idx + 1]
                buck_size = buck_end - buck_start
                if buck_size == 0:
                    continue
                sum_e = np.float32(0.0)
                sum_v = np.float32(0.0)
                for i in range(buck_size):
                    idx = bucket_indices[buck_start + i]
                    sum_e += ts_e_seg[idx]
                    sum_v += ts_v_seg[idx]
                fold[iseg, ifreq, 0, iphase] = sum_e
                fold[iseg, ifreq, 1, iphase] = sum_v

    return fold


@njit(
    ["c8[:,:,:,::1](f4[::1],f4[::1],f8[::1],i8,i8,f8,f8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def brutefold_start_complex(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    segment_len: int,
    nbins: int,
    tsamp: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Folds a time series directly into the Fourier domain using harmonic summing.

    This function computes the complex Fourier coefficients of the folded profile for
    each time series segment and trial frequency without intermediate time-domain
    binning, thus preserving sensitivity.

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
        Number of bins in the final time-domain folded profile.
    tsamp : float
        Sampling time of the time series.
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0).

    Returns
    -------
    np.ndarray
        Complex folded profiles with shape (nsegments, nfreqs, 2, nbins_f)
        where nbins_f = (nbins // 2) + 1.
    """
    nfreqs = len(freq_arr)
    nsamples = len(ts_e)
    nsegments = int(np.ceil(nsamples / segment_len))

    # Number of complex Fourier coefficients needed for an nbins profile
    nbins_f = (nbins // 2) + 1

    proper_time = np.arange(segment_len, dtype=np.float32) * tsamp - t_ref
    base_r = np.empty((nfreqs, segment_len), dtype=np.float32)
    base_i = np.empty((nfreqs, segment_len), dtype=np.float32)

    for ifreq in prange(nfreqs):
        phase_factor = np.float32(-2.0 * np.pi * freq_arr[ifreq])
        for t in range(segment_len):
            ang = phase_factor * proper_time[t]
            base_r[ifreq, t] = np.cos(ang)
            base_i[ifreq, t] = np.sin(ang)

    fold = np.zeros((nsegments, nfreqs, 2, nbins_f), dtype=np.complex64)
    for iseg in prange(nsegments):
        start = iseg * segment_len
        end = min(start + segment_len, nsamples)
        seg_len = end - start
        ts_e_seg = ts_e[start:end]
        ts_v_seg = ts_v[start:end]

        cur_r = np.empty(seg_len, dtype=np.float32)
        cur_i = np.empty(seg_len, dtype=np.float32)

        # Handle the DC component (harmonic m=0)
        # This is simply the sum of the data in the segment
        sum_e = np.float32(0.0)
        sum_v = np.float32(0.0)
        for t in range(seg_len):
            sum_e += ts_e_seg[t]
            sum_v += ts_v_seg[t]

        for ifreq in range(nfreqs):
            fold[iseg, ifreq, 0, 0] = sum_e + 0j
            fold[iseg, ifreq, 1, 0] = sum_v + 0j

            # cur_r/cur_i will step through m=1,2,… phasors by repeated multiply
            for t in range(seg_len):
                cur_r[t] = base_r[ifreq, t]
                cur_i[t] = base_i[ifreq, t]
            # loop over AC harmonics m=1..nbins_f-1
            for m in range(1, nbins_f):
                # accumulate sums for E and V
                acc_e_r = np.float32(0.0)
                acc_e_i = np.float32(0.0)
                acc_v_r = np.float32(0.0)
                acc_v_i = np.float32(0.0)
                for t in range(seg_len):
                    # E * phasor
                    acc_e_r += ts_e_seg[t] * cur_r[t]
                    acc_e_i += ts_e_seg[t] * cur_i[t]
                    # V * phasor
                    acc_v_r += ts_v_seg[t] * cur_r[t]
                    acc_v_i += ts_v_seg[t] * cur_i[t]
                    # update current phasor ← current * base
                    pr = cur_r[t]
                    pi = cur_i[t]
                    # (pr + i pi)*(br + i bi)
                    cur_r[t] = pr * base_r[ifreq, t] - pi * base_i[ifreq, t]
                    cur_i[t] = pr * base_i[ifreq, t] + pi * base_r[ifreq, t]

                fold[iseg, ifreq, 0, m] = acc_e_r + 1j * acc_e_i
                fold[iseg, ifreq, 1, m] = acc_v_r + 1j * acc_v_i

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


@njit(["f4[:,::1](f4[:,::1],f4[:,::1],f8,f8)"], cache=True, fastmath=True)
def shift_add(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    n_comps, nbins = data_tail.shape
    res = np.empty((n_comps, nbins), dtype=data_tail.dtype)
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    shift_tail = round(phase_shift_tail_float) % nbins
    shift_head = round(phase_shift_head_float) % nbins
    for j in range(nbins):
        idx1 = (j - shift_tail) % nbins
        idx2 = (j - shift_head) % nbins
        res[0, j] = data_tail[0, idx1] + data_head[0, idx2]
        res[1, j] = data_tail[1, idx1] + data_head[1, idx2]
    return res


@njit(cache=True, fastmath=True)
def shift_add_complex_direct(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    n_comps, nbins_f = data_tail.shape
    nbins = (nbins_f - 1) * 2
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    k = np.arange(nbins_f)
    phase1 = np.exp(-2j * np.pi * k * phase_shift_tail_float / nbins)
    phase2 = np.exp(-2j * np.pi * k * phase_shift_head_float / nbins)
    return (data_tail * phase1) + (data_head * phase2)


@njit(["c8[:,::1](c8[:,::1], c8[:,::1], f8, f8)"], cache=True, fastmath=True)
def shift_add_complex(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    n_comps, nbins_f = data_tail.shape
    res = np.empty((n_comps, nbins_f), dtype=data_tail.dtype)
    nbins = (nbins_f - 1) * 2

    # Precompute the angular steps
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    step_tail = np.float32(-2.0 * np.pi * phase_shift_tail_float / nbins)
    step_head = np.float32(-2.0 * np.pi * phase_shift_head_float / nbins)

    # Compute the complex delta (rotation factors)
    delta_tail = np.complex64(np.cos(step_tail) + 1j * np.sin(step_tail))
    delta_head = np.complex64(np.cos(step_head) + 1j * np.sin(step_head))
    phase_tail = np.complex64(1.0 + 0.0j)
    phase_head = np.complex64(1.0 + 0.0j)

    for k in range(nbins_f):
        res[0, k] = data_tail[0, k] * phase_tail + data_head[0, k] * phase_head
        res[1, k] = data_tail[1, k] * phase_tail + data_head[1, k] * phase_head
        phase_tail *= delta_tail
        phase_head *= delta_head
    return res


@njit(
    ["f4[:,:,::1](f4[:,:,::1],f8[::1],f4[:,:,::1], i8[::1])"],
    cache=True,
    fastmath=True,
)
def shift_add_batch(
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, nbins = segment_batch.shape
    res = np.empty((n_batch, n_comps, nbins), dtype=segment_batch.dtype)
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        shift = round(shift_float) % nbins
        fold_row = folds[isuggest_batch[irow]]
        src_idx = (-shift) % nbins
        for j in range(nbins):
            res[irow, 0, j] = fold_row[0, j] + segment_batch[irow, 0, src_idx]
            res[irow, 1, j] = fold_row[1, j] + segment_batch[irow, 1, src_idx]
            src_idx += 1
            if src_idx == nbins:
                src_idx = 0
    return res


@njit(
    ["c8[:,:,::1](c8[:,:,::1],f8[::1],c8[:,:,::1], i8[::1])"],
    cache=True,
    fastmath=True,
)
def shift_add_complex_batch(
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, nbins_f = segment_batch.shape
    res = np.empty((n_batch, n_comps, nbins_f), dtype=segment_batch.dtype)
    nbins = (nbins_f - 1) * 2
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        # Precompute phase step and delta
        angle = np.float32(-2.0 * np.float32(np.pi) * shift_float / nbins)
        delta = np.complex64(np.cos(angle) + 1j * np.sin(angle))
        phase = np.complex64(1.0 + 0.0j)
        fold = folds[isuggest_batch[irow]]
        for k in range(nbins_f):
            res[irow, 0, k] = segment_batch[irow, 0, k] * phase + fold[0, k]
            res[irow, 1, k] = segment_batch[irow, 1, k] * phase + fold[1, k]
            phase *= delta

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
