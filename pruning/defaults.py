import numpy as np
from numba import njit, types, typed
from pruning import kernels, utils


@njit
def add(data0, data1):
    return data0 + data1


@njit
def pack(data, iter_num):
    return data


@njit
def shift(data, phase_shift):
    return kernels.nb_roll2d(data, phase_shift)


@njit
def aggregate_stats(scores_arr):
    """Detect changes in the score behavior, pointing to something going wrong."""
    return NotImplementedError


@njit
def log(text, log_params):
    """Log statistics, parameters and path traces to debug the pruning algoithm."""
    return NotImplementedError


@njit
def coord_trans_params(coord_trans_matrix, coord_params, sug_params):
    return sug_params, 0


@njit
def get_phase(sug_params):
    return 0


@njit
def prepare_coordinate_trans(data_access_scheme):
    return False, False


def prepare_param_validation(*args, **kwargs):
    return False


@njit
def ffa_shift_reference(param_set: np.ndarray, t_ref: float) -> tuple[np.ndarray, float]:
    """Shift the reference time of the parameter set.

    Parameters
    ----------
    param_set : np.ndarray
        Current parameter set (e.g. accel, period)
    t_ref : float
        Reference time of the new parameter set.

    Returns
    -------
    tuple[np.ndarray, float]
        Parameter set at the new reference time and the delay.
    """
    nparams = len(param_set)
    delay_arr = np.zeros(nparams + 1)
    for i_delay in range(nparams + 1):
        for iparam in range(nparams - 1):
            expont = i_delay - iparam
            delay_arr[i_delay] += param_set[iparam] * t_ref**expont / kernels.fact(expont)
    param_set_new = delay_arr[:-1]
    param_set_new[-1] = param_set[-1] * (1 - delay_arr[-2] / utils.c_val)
    offset = delay_arr[-1] / utils.c_val
    return param_set_new, offset


@njit
def ffa_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level_cur: int,
    latter: int,
    tchunk_init: float,
    nbins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the parameters of the current iteration among the previous iteration parameters.

    Parameters
    ----------
    pset_cur : np.ndarray
        Parameter set of the current iteration to resolve. 
    parr_prev : np.ndarray
        Parameter array of the previous iteration.
    ffa_level_cur : int
        FFA level of the current iteration.
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
    if nparams == 1:
        t_ref_prev = latter * 2 ** (ffa_level_cur - 1) * tchunk_init
        pset_prev, delay_rel = pset_cur, 0
    else:
        t_ref_prev = (latter - 0.5) * 2 ** (ffa_level_cur - 1) * tchunk_init
        pset_prev, delay_rel = ffa_shift_reference(pset_cur, t_ref_prev)
    phase_rel = kernels.get_phase_idx(t_ref_prev, pset_prev[-1], nbins, delay_rel)
    pindex_prev = np.empty(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = kernels.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, phase_rel


@njit
def ffa_init(
    ts_e: np.ndarray, ts_v: np.ndarray, param_arr, chunk_len, nbins, dt
) -> np.ndarray:
    period_arr = param_arr[-1]
    fold = kernels.fold_brute_start(ts_e, ts_v, period_arr, chunk_len, nbins, dt)
    # rotating the phases to be centered in the middle of the segment
    if len(param_arr) > 1:
        for iperiod, period in enumerate(period_arr):
            tmiddle = chunk_len * dt / 2
            n_roll = kernels.get_phase_idx(tmiddle, period, nbins)
            fold[:, iperiod] = kernels.nb_roll3d(fold[:, iperiod], -n_roll)
    return fold


@njit
def prune_resolve_accel(
    param_set: np.ndarray,
    indexing_distance: int,
    param_arr: types.ListType,
    tchunk_cur: float,
    nbins: int,
) -> tuple[tuple[int, int], float]:
    total_duration = indexing_distance * tchunk_cur
    nparams = param_set.shape[0]
    del_a = 0
    del_v = 0
    del_d = 0
    for iparam in range(nparams - 1):
        deriv = nparams - iparam
        par = param_set[iparam, 0]
        del_a += total_duration ** (deriv - 2) * par / kernels.fact(deriv - 2)
        del_v += total_duration ** (deriv - 1) * par / kernels.fact(deriv - 1)
        del_d += total_duration ** (deriv - 0) * par / kernels.fact(deriv - 0)
    old_p = param_set[-1, 0]
    new_a = del_a
    new_p = old_p * (1 - del_v / utils.c_val)
    delay = del_d / utils.c_val
    relative_phase = kernels.get_phase_idx(total_duration, old_p, nbins, delay)
    old_p_index = kernels.find_nearest_sorted_idx(param_arr[-1], new_p)
    old_a_index = kernels.find_nearest_sorted_idx(param_arr[-2], new_a)
    return (old_a_index, old_p_index), relative_phase


@njit
def prune_resolve_jerk(
    param_set: np.ndarray,
    indexing_distance: int,
    param_arr: types.ListType,
    tchunk_cur: float,
    nbins: int,
) -> tuple[tuple[int, int], float]:
    total_duration = indexing_distance * tchunk_cur
    nparams = param_set.shape[0]
    del_a = 0
    del_v = 0
    del_d = 0
    for iparam in range(nparams - 1):
        deriv = nparams - iparam
        par = param_set[iparam, 0]
        del_a += total_duration ** (deriv - 2) * par / kernels.fact(deriv - 2)
        del_v += total_duration ** (deriv - 1) * par / kernels.fact(deriv - 1)
        del_d += total_duration ** (deriv - 0) * par / kernels.fact(deriv - 0)
    old_p = param_set[-1, 0]
    new_a = del_a
    new_j = param_set[-3, 0]
    new_p = old_p * (1 - del_v / utils.c_val)
    delay = del_d / utils.c_val
    relative_phase = kernels.get_phase_idx(total_duration, old_p, nbins, delay)
    old_p_index = kernels.find_nearest_sorted_idx(param_arr[-1], new_p)
    old_a_index = kernels.find_nearest_sorted_idx(param_arr[-2], new_a)
    old_j_index = kernels.find_nearest_sorted_idx(param_arr[-3], new_j)
    return (old_j_index, old_a_index, old_p_index), relative_phase


@njit
def prune_resolve_snap(
    param_set: np.ndarray,
    indexing_distance: int,
    param_arr: types.ListType,
    tchunk_cur: float,
    nbins: int,
) -> tuple[tuple[int, int], float]:
    total_duration = indexing_distance * tchunk_cur
    nparams = param_set.shape[0]
    del_j = 0
    del_a = 0
    del_v = 0
    del_d = 0
    for iparam in range(nparams - 1):
        deriv = nparams - iparam
        par = param_set[iparam, 0]
        del_j += total_duration ** (deriv - 3) * par / kernels.fact(deriv - 3)
        del_a += total_duration ** (deriv - 2) * par / kernels.fact(deriv - 2)
        del_v += total_duration ** (deriv - 1) * par / kernels.fact(deriv - 1)
        del_d += total_duration ** (deriv - 0) * par / kernels.fact(deriv - 0)
    old_p = param_set[-1, 0]
    new_a = del_a
    new_j = del_j
    new_s = param_set[-4, 0]
    new_p = old_p * (1 - del_v / utils.c_val)
    delay = del_d / utils.c_val
    relative_phase = kernels.get_phase_idx(total_duration, old_p, nbins, delay)
    old_p_index = kernels.find_nearest_sorted_idx(param_arr[-1], new_p)
    old_a_index = kernels.find_nearest_sorted_idx(param_arr[-2], new_a)
    old_j_index = kernels.find_nearest_sorted_idx(param_arr[-3], new_j)
    old_s_index = kernels.find_nearest_sorted_idx(param_arr[-4], new_s)
    return (old_s_index, old_j_index, old_a_index, old_p_index), relative_phase


@njit
def branch2leaves(
    sug_params_cur: np.ndarray,
    indexing_distance: int,
    tchunk_cur: float,
    tol: float,
    dt: float,
) -> np.ndarray:
    """Branch the parameter set into a tree of parameter sets.

    Parameters
    ----------
    sug_params_cur : np.ndarray
        Parameter set to branch.
    indexing_distance : int
        Distance between the current and the reference segment.
    tchunk_cur : float
        Duration of the current segment.
    tol : float
        Tolerance for the parameter step size (in number of bins)

    Returns
    -------
    np.ndarray
        Array of parameter sets called leaves.
    """
    total_duration = indexing_distance * tchunk_cur
    nparams = sug_params_cur.shape[0]

    dparams = np.empty(nparams, dtype=np.float64)
    param_branch = typed.List.empty_list(types.float64[:])
    for iparam in range(nparams):
        deriv = nparams - iparam
        cur_par, cur_dpar = sug_params_cur[iparam]
        if deriv == 1:
            opt_dpar = kernels.period_step(abs(total_duration), dt, cur_par, tol)
            # The total duration is the number of periods
            total_duration = abs(total_duration) / cur_par
        else:
            opt_dpar = kernels.param_step(abs(total_duration), dt, deriv, tol)
        if kernels.param_split_condition(
            opt_dpar, cur_dpar, total_duration, tol, dt, deriv
        ):
            leaf_par, act_dpar = kernels.branch_param(opt_dpar, cur_dpar, cur_par)
        else:
            leaf_par = np.array((cur_par,))
            act_dpar = cur_dpar
        dparams[iparam] = act_dpar
        param_branch.append(leaf_par)
    return kernels.get_leaves(param_branch, dparams)


@njit
def suggestion_struct(
    fold_segment: np.ndarray, param_arr: types.ListType, dparams: np.ndarray, score_func
) -> kernels.SuggestionStruct:
    """Generate a suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        N+2-dimensional array containing the data segment with first
        N dimensions corresponding to the parameter space and the last
        two dimensions corresponding to the folded data.
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
    tchunk_cur: float,
    n_iters: int,
    tol: float,
    dt: float,
):
    leaf_params = kernels.get_leaves(param_arr, dparams)[0]
    branching_pattern = []
    for ii in range(1, n_iters + 1):
        leaves_arr = branch2leaves(leaf_params, ii, tchunk_cur, tol, dt)
        branching_pattern.append(len(leaves_arr))
        leaf_params = leaves_arr[0]
    return np.array(branching_pattern)
