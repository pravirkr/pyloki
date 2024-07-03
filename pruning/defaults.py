import numpy as np
from numba import njit, typed, types

from pruning import kernels, math, utils


@njit(cache=True)
def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
    return data0 + data1


@njit(cache=True)
def pack(data: np.ndarray, ffa_level: int) -> np.ndarray:
    return data


@njit(cache=True)
def shift(data: np.ndarray, phase_shift: int) -> np.ndarray:
    return kernels.nb_roll2d(data, phase_shift)


@njit(cache=True)
def ffa_shift_ref(param_vec: np.ndarray, t_ref: float) -> np.ndarray:
    """Shift the parameters to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., a, v, d]
    t_ref : float
        Reference time to shift the parameters to. t_ref = t_j - t_i

    Returns
    -------
    np.ndarray
        Parameters at the new reference time.
    """
    nparams = len(param_vec)
    powers = np.tril(np.arange(nparams)[:, np.newaxis] - np.arange(nparams))
    # Calculate the transformation matrix (taylor coefficients)
    coeffs = t_ref**powers / math.fact(powers) * np.tril(np.ones_like(powers))
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
        kvec_prev = ffa_shift_ref(kvec_cur, t_ref_prev)
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
        pindex_prev[ip] = kernels.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, phase_rel


@njit(cache=True)
def ffa_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType,
    chunk_len: int,
    nbins: int,
    dt: float,
) -> np.ndarray:
    freq_arr = param_arr[-1]
    t_ref = chunk_len * dt / 2
    return kernels.fold_brute_start(
        ts_e,
        ts_v,
        freq_arr,
        chunk_len,
        nbins,
        dt,
        t_ref=t_ref,
    )


@njit(cache=True)
def prune_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    t_ref_prev: float,
    nbins: int,
) -> tuple[np.ndarray, float]:
    nparams = len(pset_cur)
    kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
    kvec_cur[:-2] = pset_cur[:-1, 0]  # till acceleration
    kvec_prev = ffa_shift_ref(kvec_cur, t_ref_prev)
    pset_prev = kvec_prev[:-1]
    pset_prev[-1] = pset_cur[-1, 0] * (1 + kvec_prev[-2] / utils.c_val)  # new frequency
    delay_rel = -kvec_prev[-1] / utils.c_val
    phase_rel = kernels.get_phase_idx(t_ref_prev, pset_prev[-1], nbins, delay_rel)
    pindex_prev = np.empty(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = kernels.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, phase_rel


@njit
def branch2leaves(
    param_set: np.ndarray,
    tchunk_cur: float,
    tol_bins: float,
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
    tol_bins : float
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
    dparams = np.empty(nparams, dtype=np.float64)
    param_branch = typed.List.empty_list(types.float64[:])
    for iparam in range(nparams):
        deriv = nparams - iparam
        param_cur, dparam_cur = param_set[iparam]
        if deriv == 1:
            dparam_opt = kernels.freq_step(tchunk_cur, nbins, param_cur, tol_bins)
            # for param_cur = freq, tchunk is number of period jumps
            tchunk_cur *= param_cur
        else:
            dparam_opt = kernels.param_step(
                tchunk_cur,
                tsamp,
                deriv,
                tol_bins,
                t_ref=t_ref,
            )
        split_param = kernels.param_split_condition(
            dparam_opt,
            dparam_cur,
            tchunk_cur,
            tol_bins,
            tsamp,
            deriv,
        )
        if split_param:
            leaf_params, dparam_act = kernels.branch_param(
                dparam_opt,
                dparam_cur,
                param_cur,
            )
        else:
            leaf_params, dparam_act = np.array([param_cur]), dparam_cur
        dparams[iparam] = dparam_act
        param_branch.append(leaf_params)
    return kernels.get_leaves(param_branch, dparams)


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
