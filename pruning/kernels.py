from __future__ import annotations
import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from pruning import utils


@njit
def find_nearest_sorted_idx(array: np.ndarray, value: float) -> int:
    """Finds the index of the closest value in a sorted array.

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
    idx = np.searchsorted(array, value, side="left")
    if idx > 0:
        if idx == len(array) or abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
    return idx


def find_nearest_sorted_idx_cart(cart_arr: np.ndarray, values: np.ndarray) -> int:
    idx = find_nearest_sorted_idx(cart_arr[:, 0], values[0])
    for ival in range(1, len(values)):
        col = cart_arr[:, ival]
        rows_to_consider = np.where(
            (cart_arr[:, :ival] == cart_arr[idx, :ival]).all(axis=1)
        )[0]
        nearest_idx = find_nearest_sorted_idx(col[rows_to_consider], values[ival])
        idx = rows_to_consider[nearest_idx]
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


@njit
def get_phase_idx(proper_time: float, period: float, nbins: int, delay: float = 0) -> int:
    """Calculate the phase bin index of the folded series.

    Parameters
    ----------
    proper_time : float
        The proper time of the signal
    period : float
        The period of the signal
    nbins : int
        The number of bins in the folded time series
    delay : float
        The delay of the signal due to binary motion. Correction for acceleration.

    Returns
    -------
    int
        The index of the folded time series
    """
    factor = nbins / period
    index = np.round(((proper_time % period) + delay) * factor) % nbins
    return int(index)


@njit
def get_phase_idx_many(
    proper_time: np.ndarray, period: float, nbins: int, delay: float = 0
) -> np.ndarray:
    factor = nbins / period
    indices = np.round(((proper_time % period) + delay) * factor) % nbins
    return indices.astype(np.uint32)


@njit
def fold_ts(
    ts_e: np.ndarray, ts_v: np.ndarray, ind_arrs: np.ndarray, nbins: int
) -> np.ndarray:
    """Fold a time series for given phase bin indices.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series signal
    ts_v : np.ndarray
        Time series variance
    ind_arrs : np.ndarray
        Array of phase bin indices
    nbins : int
        Number of bins in the folded time series

    Returns
    -------
    np.ndarray
        Folded time series with shape (nperiods, 2, nbins)
    """
    nperiod = len(ind_arrs)
    nsamples = len(ts_e)
    res = np.zeros(shape=(nperiod, 2, nbins))

    for iperiod in range(nperiod):
        ind_arr = ind_arrs[iperiod]
        for isamp in range(nsamples):
            res[iperiod, 0, ind_arr[isamp]] += ts_e[isamp]
            res[iperiod, 1, ind_arr[isamp]] += ts_v[isamp]
    return res


@njit
def fold_subints(
    ts_e: np.ndarray, ts_v: np.ndarray, ind_arr: np.ndarray, nbins: int, nsubints: int
) -> np.ndarray:
    nsamps = len(ts_e)
    samp_per_subint = nsamps // nsubints
    res = np.zeros(shape=(nsubints, 2, nbins))
    for isamp in range(nsubints * samp_per_subint):
        isubint = isamp // samp_per_subint
        res[isubint, 0, ind_arr[isamp]] += ts_e[isamp]
        res[isubint, 1, ind_arr[isamp]] += ts_v[isamp]
    return res


@njit
def fold_brute_start(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    period_arr: np.ndarray,
    chunk_len: int,
    nbins: int,
    dt: float,
) -> np.ndarray:
    nsamples = len(ts_e)
    chunk_idxs = np.arange(0, nsamples, chunk_len)
    proper_time = np.arange(chunk_len) * dt
    phase_idx_arrs = np.empty((len(period_arr), chunk_len), dtype=np.int64)
    for iperiod, period in enumerate(period_arr):
        phase_idx_arrs[iperiod] = get_phase_idx_many(proper_time, period, nbins)
    nchunks = int(np.ceil(nsamples / chunk_len))
    fold = np.zeros((nchunks, len(period_arr), 2, nbins))

    for isig_ind, chunk_ind in enumerate(chunk_idxs):
        ts_e_chunk = ts_e[chunk_ind : chunk_ind + chunk_len]
        ts_v_chunk = ts_v[chunk_ind : chunk_ind + chunk_len]
        fold[isig_ind] = fold_ts(ts_e_chunk, ts_v_chunk, phase_idx_arrs, nbins)
    return fold


@njit
def resample(ts_e: np.ndarray, ts_v: np.ndarray, tsamp, accel):
    if accel > 0:
        nsamps = len(ts_e) - 1
    else:
        nsamps = len(ts_e)
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
def get_init_period_arr(tobs, p_min, p_max, nbins, tol_bins):
    n_jumps = int(tobs / p_min)
    dphi = tol_bins / nbins
    dp = dphi * p_min / n_jumps
    return np.arange(p_min, p_max, dp)


@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in {0, 1}
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
def nb_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@njit
def downsample_1d(array, factor):
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
def get_unique_indices(params):
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
def get_unique_indices_scores(params, scores):
    nparams = params.shape[0]
    unique_dict = {}
    scores_dict = {}
    count_dict = {}
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
        ("data", types.f4[:, :, :]),
        ("scores", types.f8[:]),
        ("backtracks", types.f8[:, :]),
        ("_actual_size", types.int64),
    ]
)
class SuggestionStruct(object):
    def __init__(
        self,
        param_sets: np.ndarray,
        data: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> None:
        self.param_sets = param_sets
        self.data = data
        self.scores = scores
        self.backtracks = backtracks
        self._actual_size = self.scores.size

    @property
    def size(self):
        return self.scores.size

    @property
    def actual_size(self):
        return self._actual_size

    @property
    def nparams(self):
        return len(self.param_sets[0])

    def init_new(self, max_suggestions: int) -> SuggestionStruct:
        param_sets = np.zeros((max_suggestions, self.nparams, 2))
        data = np.zeros((max_suggestions, *self.data.shape[1:]), dtype=np.float32)
        scores = np.zeros(max_suggestions)
        backtracks = np.zeros((max_suggestions, 2 + self.nparams))
        return SuggestionStruct(param_sets, data, scores, backtracks)

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
            self.data[:ind_size],
            self.scores[:ind_size],
            self.backtracks[:ind_size],
        )

    def _keep(self, indices: np.ndarray) -> SuggestionStruct:
        ind_size = np.count_nonzero(indices)
        sug_new = self.init_new(self.size)
        sug_new.param_sets[:ind_size] = self.param_sets[indices]
        sug_new.data[:ind_size] = self.data[indices]
        sug_new.scores[:ind_size] = self.scores[indices]
        sug_new.backtracks[:ind_size] = self.backtracks[indices]
        sug_new._actual_size = ind_size
        return sug_new


@njit
def numba_bf_row_vector_unique(arr):
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
def numba_bypass_all_close(arr1, arr2, tol=1e-8):
    return np.all(np.abs(arr1 - arr2) < tol)


def fact_factory(n_tab_out=100):
    fact_tab = np.ones(n_tab_out)

    def _fact(num, n_tab=n_tab_out):
        if num < n_tab:
            return fact_tab[num]
        ret = 1
        for nn in range(1, num + 1):
            ret *= nn
        return ret

    for ii in range(n_tab_out):
        fact_tab[ii] = _fact(ii, 0)
    return _fact


fact = njit(fact_factory(120))


@njit
def param_split_condition(opt_dpar, cur_dpar, duration, tol, dt, deriv=1):
    return ((cur_dpar - opt_dpar) / 2 * (duration) ** deriv / fact(deriv)) > tol * dt


@njit
def param_step(
    tobs: float, tsamp: float, deriv: int, tol: float, t_ref: float = 0
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
        Tolerance parameter for the polynomial search (in units of time bins)
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0)

    Returns
    -------
    float
        Optimal parameter step size
    """
    if deriv < 2:
        raise ValueError("deriv must be >= 2")
    dparam = tsamp * fact(deriv) * utils.c_val / (tobs - t_ref) ** deriv
    return tol * dparam


@njit
def period_step_init(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    """Calculate the period step size for polynomial search.

    Parameters
    ----------
    tobs : float
        Total observation time
    nbins : float
        Number of bins in the folded time series
    p_min : float
        Minimum period to search
    tol : float
        Tolerance parameter for the polynomial search
        (in units of number of time bins across the pulsar ducy)

    Returns
    -------
    float
        Optimal period step size

    Notes
    -----
    n_jumps * dp = (dphi * P)
    """
    n_jumps = tobs // p_min
    dphi = tol / nbins
    return dphi * p_min / n_jumps


@njit
def period_step(tobs: float, tsamp: int, p_min: float, tol: float) -> float:
    n_jumps = tobs / p_min
    return tol * tsamp / (n_jumps / 2)

@njit
def freq_step(tobs: int, tsamp: int, f_min: float, tol: float) -> float:
    return tol * f_min**2 * tsamp / (tobs * f_min - 1)


@njit
def branch_param(opt_d_par, cur_d_par, cur_val):
    n = 2 + int(np.ceil(cur_d_par / opt_d_par))
    confidence_const = 0.5 * (1 + 1 / float(n - 2))
    par_array = np.linspace(
        cur_val - confidence_const * cur_d_par,
        cur_val + confidence_const * cur_d_par,
        n,
    )[1:-1]
    actual_d_par = cur_d_par / (n - 2)
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
