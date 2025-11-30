from __future__ import annotations

import numpy as np
from numba import njit, prange, types

from pyloki.utils import np_utils, psr_utils, transforms


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


@njit(
    ["c8[:,:,:,::1](f4[::1],f4[::1],f8[::1],i8,i8,f8,f8,i8)"],
    cache=True,
    parallel=True,
    fastmath=True,
)
def brutefold_complex_oversampled(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    segment_len: int,
    nbins: int,
    tsamp: float,
    t_ref: float = 0,
    oversample_factor: int = 4,
) -> np.ndarray:
    nfreqs = len(freq_arr)
    nsamples = len(ts_e)
    nsegments = int(np.ceil(nsamples / segment_len))
    fine_nbins = nbins * oversample_factor
    # Get time-domain folded profiles on oversampled grid
    fold_time = brutefold_bucketed(
        ts_e,
        ts_v,
        freq_arr,
        segment_len,
        fine_nbins,
        tsamp,
        t_ref,
    )

    # Output size: keep only first (nbins//2)+1 Fourier coefficients
    nbins_f = (nbins // 2) + 1
    fold_out = np.zeros((nsegments, nfreqs, 2, nbins_f), dtype=np.complex64)
    for iseg in prange(nsegments):
        for ifreq in range(nfreqs):
            fold_out[iseg, ifreq] = np.fft.rfft(fold_time[iseg, ifreq], axis=-1)[
                :, :nbins_f,
            ]
    return fold_out


@njit(cache=True, fastmath=True)
def ffa_taylor_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
    bseg_brute: int,
    fold_bins: int,
    tsamp: float,
) -> np.ndarray:
    """Initialize the fold for the FFA search.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series intensity (signal weighted by E/V).
    ts_v : np.ndarray
        Time series variance (E**2/V).
    param_arr : types.ListType[types.Array]
        Parameter grid array for each search parameter dimension.
    bseg_brute : int
        Brute force segment size in bins.
    fold_bins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the data.

    Returns
    -------
    np.ndarray
        Initial fold for the FFA search.
    """
    freq_arr = param_arr[-1]
    nparams = len(param_arr)
    t_ref = 0 if nparams == 1 else bseg_brute * tsamp / 2
    return brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        fold_bins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def ffa_taylor_init_complex(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
    bseg_brute: int,
    fold_bins: int,
    tsamp: float,
) -> np.ndarray:
    """Initialize the fold for the FFA search (complex-domain).

    Parameters
    ----------
    ts_e : np.ndarray
        Time series intensity (signal weighted by E/V).
    ts_v : np.ndarray
        Time series variance (E**2/V).
    param_arr : types.ListType[types.Array]
        Parameter grid array for each search parameter dimension.
    bseg_brute : int
        Brute force segment size in bins.
    fold_bins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the data.

    Returns
    -------
    np.ndarray
        Initial fold for the FFA search.
    """
    freq_arr = param_arr[-1]
    nparams = len(param_arr)
    t_ref = 0 if nparams == 1 else bseg_brute * tsamp / 2
    return brutefold_start_complex(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        fold_bins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def ffa_taylor_resolve(
    pset_cur: np.ndarray,
    param_arr: types.ListType[types.Array],
    ffa_level: int,
    latter: int,
    tseg_brute: float,
    fold_bins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the params to find the closest index in grid and absolute phase shift.

    Parameters
    ----------
    pset_cur : np.ndarray
        The current iter parameter set to resolve.
    param_arr : types.ListType[types.Array]
        Parameter grid array for the previous iteration (ffa_level - 1).
    ffa_level : int
        Current FFA level (same level as pset_cur).
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tseg_brute : float
        Duration of the brute force segment.
    fold_bins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, float]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    phase_shift is complete phase shift with fractional part.

    Notes
    -----
    delta_t is the time difference between the reference time of the current segment
    (0) and the reference time of the previous segment.
    """
    nparams = len(pset_cur)
    if nparams == 1:
        delta_t = latter * 2 ** (ffa_level - 1) * tseg_brute
        pset_new, delay = pset_cur, 0
    else:
        delta_t = (latter - 0.5) * 2 ** (ffa_level - 1) * tseg_brute
        pset_new, delay = transforms.shift_taylor_params_d_f(pset_cur, delta_t)
    relative_phase = psr_utils.get_phase_idx(delta_t, pset_cur[-1], fold_bins, delay)
    pindex_prev = np.zeros(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(param_arr[ip], pset_new[ip])
    return pindex_prev, relative_phase
