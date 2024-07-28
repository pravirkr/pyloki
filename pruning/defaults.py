import numpy as np
from numba import njit, typed, types

from pruning import kernels, math, utils


@njit(cache=True)
def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
    return data0 + data1


@njit(cache=True)
def pack(data: np.ndarray) -> np.ndarray:
    return data


@njit(cache=True)
def shift(data: np.ndarray, phase_shift: int) -> np.ndarray:
    return math.nb_roll2d(data, phase_shift)


@njit(cache=True)
def get_trans_matrix(
    coord_cur: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)



@njit(cache=True, fastmath=True)
def shift_params(param_vec: np.ndarray, tj_minus_ti: float) -> np.ndarray:
    """Shift the parameters to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., a, v, d]
    tj_minus_ti : float
        Reference time to shift the parameters to. t_ref = t_j - t_i

    Returns
    -------
    np.ndarray
        Parameters at the new reference time.
    """
    nparams = len(param_vec)
    powers = np.tril(np.arange(nparams)[:, np.newaxis] - np.arange(nparams))
    # Calculate the transformation matrix (taylor coefficients)
    coeffs = tj_minus_ti**powers / math.fact(powers) * np.tril(np.ones_like(powers))
    return np.dot(coeffs, param_vec)


@njit(cache=True)
def ffa_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
    tchunk_init: float,
    nbins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the parameters of the current iter among the previous iter parameters.

    Parameters
    ----------
    pset_cur : np.ndarray
        Parameter set of the current iteration to resolve.
    parr_prev : np.ndarray
        Parameter array of the previous iteration.
    ffa_level : int
        Current FFA level.
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tchunk_init : float
        Initial chunk duration.
    nbins : int
        Number of bins in the data.

    Returns
    -------
    tuple[np.ndarray, float]
        The resolved parameter set index and the relative phase shift.
    """
    nparams = len(pset_cur)
    t_ref_prev = (latter - 0.5) * 2 ** (ffa_level - 1) * tchunk_init
    if nparams == 1:
        pset_prev, delay_rel = pset_cur, 0
    else:
        kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
        kvec_cur[:-2] = pset_cur[:-1]  # till acceleration
        kvec_prev = shift_params(kvec_cur, t_ref_prev)
        pset_prev = kvec_prev[:-1]
        pset_prev[-1] = pset_cur[-1] * (1 + kvec_prev[-2] / utils.c_val)
        delay_rel = kvec_prev[-1] / utils.c_val
    phase_rel = kernels.get_phase_idx_helper(
        t_ref_prev,
        pset_prev[-1],
        nbins,
        delay_rel,
    )
    pindex_prev = np.empty(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = math.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, phase_rel


@njit(cache=True)
def ffa_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType,
    segment_len: int,
    nbins: int,
    tsamp: float,
) -> np.ndarray:
    freq_arr = param_arr[-1]
    t_ref = segment_len * tsamp / 2
    return kernels.brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        segment_len,
        nbins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    t_ref_cur: float,
    t_ref_init: float,
    nbins: int,
) -> tuple[tuple[int, int], int]:
    """Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    t_ref_cur : float
        The reference time for the current segment.
    t_ref_init : float
        The reference time for the initial segment (pruning level 0).
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[tuple[int, int], int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to t_ref_init, so we need to shift it to t_ref_cur to get the
    resolved parameters index and phase shift.

    """
    nparams = len(leaf)
    # distance between the current segment reference time and the global reference time
    tpoly = t_ref_cur - t_ref_init

    kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
    kvec_cur[:-2] = leaf[:-1, 0]  # till acceleration
    kvec_new = shift_params(kvec_cur, tpoly)

    old_f = leaf[-1, 0]

    new_a = kvec_new[-3]
    new_f = old_f * (1 + kvec_new[-2] / utils.c_val)
    delay = kvec_new[-1] / utils.c_val

    relative_phase = kernels.get_phase_idx(tpoly, old_f, nbins, delay)
    prev_index_a = math.find_nearest_sorted_idx(param_arr[-2], new_a)
    prev_index_f = math.find_nearest_sorted_idx(param_arr[-1], new_f)
    return (prev_index_a, prev_index_f), relative_phase


@njit
def branch2leaves(
    param_set: np.ndarray,
    tchunk_cur: float,
    tolerance: float,
    tsamp: float,
    nbins: int,
    t_ref: float,
) -> np.ndarray:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    param_set : np.ndarray
        Parameter set to branch. Shape: (nparams, 2)
    tchunk_cur : float
        Total chunk duration at the current pruning level.
    tolerance : float
        Tolerance for the parameter step size in bins.
    tsamp : float
        Sampling time.
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    nparams, _ = param_set.shape
    param_cur = param_set[:, 0]
    dparam_cur = param_set[:, 1]
    dparam_opt = np.zeros(nparams)
    for iparam in range(nparams):
        deriv = nparams - iparam
        if deriv == 1:
            dparam_opt_p = kernels.freq_step(
                tchunk_cur,
                nbins,
                param_cur[iparam],
                tolerance,
            )
            # for param_cur = freq, tchunk is number of period jumps
            tchunk_cur *= param_cur[iparam]
        else:
            dparam_opt_p = kernels.param_step(
                tchunk_cur,
                tsamp,
                deriv,
                tolerance,
                t_ref=t_ref,
            )
        dparam_opt[iparam] = dparam_opt_p
    tol_time = tolerance * tsamp
    return split_params(param_cur, dparam_cur, dparam_opt, tol_time, tchunk_cur)


@njit
def split_params(
    param_cur: np.ndarray,
    dparam_cur: np.ndarray,
    dparam_opt: np.ndarray,
    tol_time: float,
    tchunk_cur: float,
) -> np.ndarray:
    nparams = len(param_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(nparams, dtype=np.float64)
    for iparam in range(nparams):
        deriv = nparams - iparam
        shift = (
            (dparam_cur[iparam] - dparam_opt[iparam])
            / 2
            * (tchunk_cur) ** deriv
            / math.fact(deriv)
        )
        if shift > tol_time:
            leaf_params, dparam_act = kernels.branch_param(
                dparam_opt[iparam],
                dparam_cur[iparam],
                param_cur[iparam],
            )
        else:
            leaf_params, dparam_act = np.array([param_cur[iparam]]), dparam_cur[iparam]
        leaf_dparams[iparam] = dparam_act
        leaf_params.append(leaf_params)
    return kernels.get_leaves(leaf_params, leaf_dparams)


@njit
def suggestion_struct(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    score_func: types.FunctionType,
) -> kernels.SuggestionStruct:
    """Generate a suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_period, 2, n_bins). Parameter dimensions are first two.
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.
    score_func : _type_
        Function to score the folded data.

    Returns
    -------
    kernels.SuggestionStruct
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    # \n_param_sets = n_accel * n_period
    # \param_sets_shape = [n_param_sets, 2]
    param_sets = kernels.get_leaves(param_arr, dparams)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)))
    return kernels.SuggestionStruct(param_sets, data, scores, backtracks)


@njit
def generate_branching_pattern(
    param_arr: types.ListType,
    dparams: np.ndarray,
    tchunk_ffa: float,
    nsegments: int,
    tol_bins: float,
    tsamp: float,
    nbins: int,
    isuggest: int,
) -> np.ndarray:
    leaf_param_sets = kernels.get_leaves(param_arr, dparams)
    branching_pattern = []
    for prune_level in range(1, nsegments):
        tchunk_cur = tchunk_ffa * (prune_level + 1)
        leaves_arr = branch2leaves(
            leaf_param_sets[isuggest],
            tchunk_cur,
            tol_bins,
            tsamp,
            nbins,
            tchunk_cur / 2,
        )
        branching_pattern.append(len(leaves_arr))
        leaf_param_sets = leaves_arr
    return np.array(branching_pattern)
