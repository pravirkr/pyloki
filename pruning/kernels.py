from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit, types, vectorize
from numba.experimental import jitclass

from pruning import math, utils


@njit(cache=True)
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


@vectorize(cache=True)
def get_phase_idx(proper_time: float, freq: float, nbins: int, delay: float) -> int:
    """Calculate the phase index of the folded profile.

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
    phase_idx = int(np.round(phase * nbins))
    return phase_idx % nbins


@njit
def fold_ts(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    ind_arrs: np.ndarray,
    nbins: int,
    nsubints: int,
) -> np.ndarray:
    """Fold a time series for given phase bin indices.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series signal
    ts_v : np.ndarray
        Time series variance
    ind_arrs : np.ndarray
        Phase bin indices of each time sample for each period
    nbins : int
        Number of bins in the folded profile
    nsubints : int
        Number of sub-integrations in time.

    Returns
    -------
    np.ndarray
        Folded time series with shape (nperiods, nsubints, 2, nbins)
    """
    nsamps = len(ts_e)
    nperiod = len(ind_arrs)
    samp_per_subint = nsamps // nsubints
    samps_final = nsubints * samp_per_subint
    subint_idxs = np.arange(samps_final) // samp_per_subint
    res = np.zeros(shape=(nperiod, nsubints, 2, nbins), dtype=ts_e.dtype)
    for iperiod in range(nperiod):
        ind_arr = ind_arrs[iperiod]
        for isamp in range(samps_final):
            isubint = subint_idxs[isamp]
            iphase = ind_arr[isamp]
            res[iperiod, isubint, 0, iphase] += ts_e[isamp]
            res[iperiod, isubint, 1, iphase] += ts_v[isamp]
    return res


@njit
def fold_brute_start(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    chunk_len: int,
    nbins: int,
    dt: float,
) -> np.ndarray:
    nsamples = len(ts_e)
    chunk_idxs = np.arange(0, nsamples, chunk_len)
    proper_time = np.arange(chunk_len) * dt
    phase_idx_arrs = np.empty((len(freq_arr), chunk_len), dtype=np.int64)
    for ifreq, freq in enumerate(freq_arr):
        phase_idx_arrs[ifreq] = get_phase_idx(proper_time, freq, nbins, 0)
    nchunks = int(np.ceil(nsamples / chunk_len))
    fold = np.zeros((nchunks, len(freq_arr), 2, nbins))

    for isig_ind, chunk_ind in enumerate(chunk_idxs):
        ts_e_chunk = ts_e[chunk_ind : chunk_ind + chunk_len]
        ts_v_chunk = ts_v[chunk_ind : chunk_ind + chunk_len]
        fold[isig_ind] = fold_ts(ts_e_chunk, ts_v_chunk, phase_idx_arrs, nbins, 1)[:, 0]
    return fold


@njit
def resample(ts_e: np.ndarray, ts_v: np.ndarray, tsamp: float, accel: float) -> tuple:
    nsamps = len(ts_e) - 1 if accel > 0 else len(ts_e)
    ts_e_resamp = np.zeros_like(ts_e)
    ts_v_resamp = np.zeros_like(ts_v)

    partial_calc = (accel * tsamp) / (2 * utils.c)
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


@njit
def get_init_period_arr(
    tobs: float,
    p_min: float,
    p_max: float,
    nbins: int,
    tol_bins: int,
) -> np.ndarray:
    n_jumps = int(tobs / p_min)
    dphi = tol_bins / nbins
    dp = dphi * p_min / n_jumps
    return np.arange(p_min, p_max, dp)


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
    param_cart = cartesian_prod(param_arr)
    param_mat = np.expand_dims(param_cart, axis=2)
    dparams_set = np.broadcast_to(np.expand_dims(dparams, 1), param_mat.shape)
    return np.concatenate((param_mat, dparams_set), axis=2)


@njit
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


@njit
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
    """A struct to hold suggestions for pruning.

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

    def get_new(self, max_sugg: int) -> SuggestionStruct:
        param_sets = np.zeros((max_sugg, self.nparams, 2))
        folds = np.zeros((max_sugg, *self.folds.shape[1:]), dtype=self.folds.dtype)
        scores = np.zeros(max_sugg)
        backtracks = np.zeros((max_sugg, 2 + self.nparams))
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


@jitclass(
    spec=[
        ("n_leaves_tot", types.i8),
        ("n_leaves_phy", types.i8),
        ("n_branches", types.i8),
        ("n_leaves_surv", types.i8),
    ],
)
class PruneStats:
    def __init__(
        self,
        n_branches: int,
        n_leaves_tot: int,
        n_leaves_phy: int,
        n_leaves_surv: int,
    ) -> None:
        self.n_branches = n_branches
        self.n_leaves_tot = n_leaves_tot
        self.n_leaves_phy = n_leaves_phy
        self.n_leaves_surv = n_leaves_surv

    @property
    def lb_leaves(self) -> float:
        return np.round(np.log2(self.n_leaves_tot), 2)

    @property
    def branch_frac_tot(self) -> float:
        return np.round(self.n_leaves_tot / self.n_branches, 2)

    @property
    def branch_frac_phy(self) -> float:
        return np.round(self.n_leaves_phy / self.n_branches, 2)

    @property
    def surv_frac(self) -> float:
        return np.round(self.n_leaves_surv / self.n_leaves_tot, 2)


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


@njit
def param_split_condition(
    dparam_opt: float,
    dparam_cur: float,
    duration: float,
    tol_bins: int,
    dt: float,
    deriv: int,
) -> bool:
    shift = (dparam_cur - dparam_opt) / 2 * (duration) ** deriv / math.fact(deriv)
    return shift > tol_bins * dt


@njit
def param_step(
    tobs: float,
    tsamp: float,
    deriv: int,
    tol_bins: float,
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
    tol_bins : float
        Tolerance parameter for the polynomial search (in units of time bins)
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
    dparam = tsamp * math.fact(deriv) * utils.c_val / (tobs - t_ref) ** deriv
    return tol_bins * dparam


@njit
def freq_step(tobs: int, nbins: int, f_max: float, tol_bins: float) -> float:
    m_cycle = tobs * f_max
    tsamp_min = 1 / (f_max * nbins)
    return tol_bins * f_max**2 * tsamp_min / (m_cycle - 1)


@njit
def period_step_init(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit
def period_step(tobs: float, tsamp: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    return tol * (tsamp * 2) / (m_cycle - 1)


@njit
def cheb_step(poly_order: int, tsamp: float, tol: int) -> np.ndarray:
    return np.zeros(poly_order + 1, np.float32) + ((tol * tsamp) * utils.c_val)


@njit
def branch_param(
    dparam_opt: float,
    dparam_cur: float,
    param_cur: float,
) -> tuple[np.ndarray, float]:
    n = 2 + int(np.ceil(dparam_cur / dparam_opt))
    confidence_const = 0.5 * (1 + 1 / float(n - 2))
    par_array = np.linspace(
        param_cur - confidence_const * dparam_cur,
        param_cur + confidence_const * dparam_cur,
        n,
    )[1:-1]
    actual_d_par = dparam_cur / (n - 2)
    return par_array, actual_d_par


@njit
def range_param(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """Return a range of parameters with a given step size.

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
    if dv > (vmax - vmin) / 2:
        return np.array([(vmax + vmin) / 2])
    npoints = int((vmax - vmin) / dv)
    return np.linspace(vmin, vmax, npoints + 2)[1:-1]
