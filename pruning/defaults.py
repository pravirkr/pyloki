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
def ffa_resolve_accel(param_set: np.ndarray, t0_star: float) -> tuple[float, float, float]:
    nparams = len(param_set)
    del_v = 0
    del_d = 0
    for iparam in range(nparams - 1):
        deriv = nparams - iparam
        par = param_set[iparam]
        del_v += t0_star ** (deriv - 1) * par / kernels.fact(deriv - 1)
        del_d += t0_star ** (deriv - 0) * par / kernels.fact(deriv - 0)
    old_p = param_set[-1]
    new_a = param_set[-2]
    new_p = old_p * (1 - del_v / utils.c_val)
    delay = del_d / utils.c_val
    return new_a, new_p, delay


@njit
def ffa_resolve_jerk(param_set: np.ndarray, t0_star: float) -> tuple[float, float, float]:
    nparams = len(param_set)
    del_a = 0
    del_v = 0
    del_d = 0
    for iparam in range(nparams - 1):
        deriv = nparams - iparam
        par = param_set[iparam]
        del_a += t0_star ** (deriv - 2) * par / kernels.fact(deriv - 2)
        del_v += t0_star ** (deriv - 1) * par / kernels.fact(deriv - 1)
        del_d += t0_star ** (deriv - 0) * par / kernels.fact(deriv - 0)
    old_p = param_set[-1]
    new_a = del_a
    new_j = param_set[-3]
    new_p = old_p * (1 - del_v / utils.c_val)
    delay = del_d / utils.c_val
    return new_j, new_a, new_p, delay


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
