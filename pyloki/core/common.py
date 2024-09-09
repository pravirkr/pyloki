from __future__ import annotations

import numpy as np
from numba import njit, types, vectorize
from numba.experimental import jitclass

from pyloki.utils import math, np_utils
from pyloki.utils.misc import C_VAL


@vectorize(nopython=True, cache=True)
def get_phase_idx(proper_time: float, freq: float, nbins: int, delay: float) -> int:
    """Calculate the phase index of the proper time in the folded profile.

    Parameters
    ----------
    proper_time : float
        Proper time of the signal in time units.
    period : float
        Period of the signal in time units.
    nbins : int
        Number of bins in the folded profile.
    delay : float
        Signal delay due to pulsar binary motion in time units.

    Returns
    -------
    int
        Phase bin index of the proper time in the folded profile.
    """
    phase = ((proper_time + delay) * freq) % 1
    return int(np.round(phase * nbins)) % nbins


@njit(cache=True)
def get_phase_idx_helper(
    proper_time: float,
    freq: float,
    nbins: int,
    delay: float,
) -> int:
    if proper_time >= 0:
        return get_phase_idx(proper_time, freq, nbins, delay)
    phase_abs = get_phase_idx(abs(proper_time), freq, nbins, delay)
    phase_dist = get_phase_idx(2 * abs(proper_time), freq, nbins, delay)
    phase_neg = phase_abs - phase_dist
    return phase_neg + nbins if phase_neg < 0 else phase_neg


@njit
def param_step(
    tobs: float,
    tsamp: float,
    deriv: int,
    tol: float,
    t_ref: float = 0,
) -> float:
    """Calculate the parameter step size for polynomial search.

    Parameters
    ----------
    tobs : float
        Total observation time of the segment in seconds.
    tsamp : float
        Sampling time of the segment in seconds.
    deriv : int
        Derivative of the parameter (2: acceleration, 3: jerk, etc.)
    tol : float
        Tolerance parameter for the polynomial search (in bins).
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0)

    Returns
    -------
    float
        Optimal parameter step size
    """
    if deriv < 2:
        msg = "deriv must be >= 2"
        raise ValueError(msg)
    dparam = tsamp * math.fact(deriv) * C_VAL / (tobs - t_ref) ** deriv
    return tol * dparam


@njit
def freq_step(tobs: int, nbins: int, f_max: float, tol: float) -> float:
    m_cycle = tobs * f_max
    tsamp_min = 1 / (f_max * nbins)
    return tol * f_max**2 * tsamp_min / (m_cycle - 1)


@njit
def freq_step_approx(tobs: int, f_max: float, tsamp: float, tol: float) -> float:
    m_cycle = tobs * f_max
    return tol * f_max**2 * tsamp / (m_cycle - 1)


@njit
def period_step_init(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit
def period_step(tobs: float, tsamp: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    return tol * (tsamp * 2) / (m_cycle - 1)


@njit(cache=True, fastmath=True)
def branch_param(
    param_cur: float,
    dparam_cur: float,
    dparam_new: float,
    param_min: float,
    param_max: float,
) -> tuple[np.ndarray, float]:
    """
    Refine a parameter range around a current value with a new step size.

    Parameters
    ----------
    param_cur : float
        current parameter value (center of the range)
    dparam_cur : float
        current parameter step size (half-width of the range)
    dparam_new : float
        new parameter step size (half-width of the new range)
    param_min : float
        minimum value of the parameter range
    param_max : float
        maximum value of the parameter range

    Returns
    -------
    tuple[np.ndarray, float]
        Array of new parameter values and the actual new parameter step size
    """
    if not (dparam_cur > 0 and dparam_new > 0):
        msg = "dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    if not (param_min <= param_cur <= param_max):
        msg = "Invalid input: ensure param_min < param_cur < param_max."
        raise ValueError(msg)
    if dparam_new > (param_max - param_min) / 2:
        return np.array([param_cur]), dparam_new
    n = 2 + int(np.ceil(dparam_cur / dparam_new))
    if n < 3:
        msg = "Invalid input: ensure dparam_cur > dparam_new."
        raise ValueError(msg)
    # 0.5 < confidence_const < 1
    confidence_const = 0.5 * (1 + 1 / (n - 2))
    half_range = confidence_const * dparam_cur
    param_arr_new = np.linspace(param_cur - half_range, param_cur + half_range, n)[1:-1]
    dparam_new_actual = dparam_cur / (n - 2)
    return param_arr_new, dparam_new_actual


@njit(cache=True, fastmath=True)
def range_param(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """
    Return a range of parameters with a given step size.

    Parameters
    ----------
    vmin : float
        Minimum value of the parameter range.
    vmax : float
        Maximum value of the parameter range.
    dv : float
        Step size of the parameter range.

    Returns
    -------
    np.ndarray
        Array of parameter values.

    Notes
    -----
    Correctly stepping a parameter is a non-trivial ugly question.
    Make sure this is right another time.
    """
    if not (vmin < vmax and dv > 0):
        msg = "Invalid input: ensure vmin < vmax and dv > 0."
        raise ValueError(msg)
    if dv > (vmax - vmin) / 2:
        return np.array([(vmax + vmin) / 2])
    npoints = int((vmax - vmin) / dv)
    return np.linspace(vmin, vmax, npoints + 2)[1:-1]


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
    phase_map = get_phase_idx(proper_time, freq, nbins, 0)
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


@njit(["f4[:,:,:,:](f4[:],f4[:],f8[:],i8,i8,f8,f8)"], cache=True, fastmath=True)
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
        Time series signal (intensity)
    ts_v : np.ndarray
        Time series variance
    freq_arr : np.ndarray
        Frequency array to fold the time series
    segment_len : int
        Length of each folded segment
    nbins : int
        Number of bins in the folded profile
    tsamp : float
        Sampling time of the time series
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0)

    Returns
    -------
    np.ndarray
        Folded time series with shape (nsegments, nfreqs, 2, nbins)
    """
    nfreqs = len(freq_arr)
    nsamples = len(ts_e)
    nsegments = int(np.ceil(nsamples / segment_len))
    proper_time = np.arange(segment_len) * tsamp - t_ref
    phase_map = np.zeros((nfreqs, segment_len), dtype=np.int32)
    for ifreq in range(nfreqs):
        phase_map[ifreq] = get_phase_idx(proper_time, freq_arr[ifreq], nbins, 0)

    fold = np.zeros((nsegments, nfreqs, 2, nbins), dtype=np.float32)
    for iseg in range(nsegments):
        segment_start = iseg * segment_len
        ts_e_seg = ts_e[segment_start : segment_start + segment_len]
        ts_v_seg = ts_v[segment_start : segment_start + segment_len]
        for ifreq in range(nfreqs):
            for isamp in range(segment_len):
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
def get_unique_indices(params: np.ndarray) -> np.ndarray:
    nparams = params.shape[0]
    unique_dict = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.round(params[ii][-1:, 0][0] * 10**9))
        if key not in unique_dict:
            unique_dict[key] = True
            unique_indices[count] = ii
            count += 1

    return unique_indices[:count]


@njit(cache=True, fastmath=True)
def get_unique_indices_scores(params: np.ndarray, scores: np.ndarray) -> np.ndarray:
    nparams = params.shape[0]
    unique_dict: dict[int, bool] = {}
    scores_dict: dict[int, float] = {}
    count_dict: dict[int, int] = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.sum(params[ii][-2:, 0] * 10**9) + 0.5)
        if unique_dict.get(key, False):
            if scores[ii] > scores_dict[key]:
                scores_dict[key] = scores[ii]
                count = count_dict[key]
                unique_indices[count] = ii
        else:
            unique_dict[key] = True
            scores_dict[key] = scores[ii]
            count_dict[key] = count
            unique_indices[count] = ii
            count += 1
    return unique_indices[:count]


@jitclass(
    spec=[
        ("param_sets", types.f8[:, :, :]),
        ("folds", types.f4[:, :, :]),
        ("scores", types.f8[:]),
        ("backtracks", types.f8[:, :]),
        ("_actual_size", types.int64),
    ],
)
class SuggestionStruct:
    """
    A struct to hold suggestions for pruning.

    Parameters
    ----------
    param_sets : np.ndarray
        Array of parameter sets with shape (nsuggestions, nparams, 2)
    folds : np.ndarray
        Array of folded profiles with shape (nsuggestions, ..., 2, nbins)
    scores : np.ndarray
        Array of scores for each suggestion (nsuggestions,)
    backtracks : np.ndarray
        Array of backtracks for each suggestion (nsuggestions, 2 + nparams)
    """

    def __init__(
        self,
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> None:
        self.param_sets = param_sets
        self.folds = folds
        self.scores = scores
        self.backtracks = backtracks
        self._actual_size = self.param_sets.shape[0]

    @property
    def actual_size(self) -> int:
        return self._actual_size

    @property
    def size(self) -> int:
        return self.param_sets.shape[0]

    @property
    def nparams(self) -> int:
        return self.param_sets.shape[1]

    @property
    def score_max(self) -> float:
        return np.max(self.scores) if self.size > 0 else 0

    @property
    def score_min(self) -> float:
        return np.min(self.scores) if self.size > 0 else 0

    def get_new(self, max_sugg: int) -> SuggestionStruct:
        param_sets = np.zeros((max_sugg, self.nparams, 2))
        folds = np.zeros((max_sugg, *self.folds.shape[1:]), dtype=self.folds.dtype)
        scores = np.zeros(max_sugg)
        backtracks = np.zeros((max_sugg, self.backtracks.shape[1]))
        return SuggestionStruct(param_sets, folds, scores, backtracks)

    def remove_repetitions(self) -> SuggestionStruct:
        idx = get_unique_indices_scores(self.param_sets, self.scores)
        idx_bool = np.zeros(self.size, dtype=np.bool_)
        idx_bool[idx] = True
        return self._keep(idx_bool)

    def apply_threshold(self, threshold: float) -> SuggestionStruct:
        idx = self.scores >= threshold
        return self._keep(idx)

    def remove_repeat_threshold(self, threshold: float) -> SuggestionStruct:
        idx_repeat = get_unique_indices_scores(self.param_sets, self.scores)
        idx_repeat_bool = np.zeros(self.size, dtype=np.bool_)
        idx_repeat_bool[idx_repeat] = True
        idx_threshold = self.scores >= threshold
        idx = np.logical_and(idx_repeat_bool, idx_threshold)
        return self._keep(idx)

    def trim_empty(self, ind_size: int) -> SuggestionStruct:
        return SuggestionStruct(
            self.param_sets[:ind_size],
            self.folds[:ind_size],
            self.scores[:ind_size],
            self.backtracks[:ind_size],
        )

    def get_best(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.argmax(self.scores)
        return self.param_sets[idx], self.folds[idx], self.scores[idx]

    def _keep(self, indices: np.ndarray) -> SuggestionStruct:
        ind_size = np.sum(indices)
        sug_new = self.get_new(self.size)
        sug_new.param_sets[:ind_size] = self.param_sets[indices]
        sug_new.folds[:ind_size] = self.folds[indices]
        sug_new.scores[:ind_size] = self.scores[indices]
        sug_new.backtracks[:ind_size] = self.backtracks[indices]
        sug_new._actual_size = ind_size  # noqa: SLF001
        return sug_new
