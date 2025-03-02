from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import np_utils, psr_utils
from pyloki.utils.misc import C_VAL
from pyloki.utils.suggestion import SuggestionStruct


@njit(cache=True, fastmath=True)
def ffa_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType,
    bseg_brute: int,
    nbins: int,
    tsamp: float,
) -> np.ndarray:
    """Initialize the fold for the FFA search.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series (signal) intensity.
    ts_v : np.ndarray
        Time series variance.
    param_arr : types.ListType
        Parameter array for each dimension.
    bseg_brute : int
        Brute force segment size in bins.
    nbins : int
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
    return common.brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        nbins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def ffa_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
    tseg_brute: float,
    nbins: int,
) -> tuple[np.ndarray, int]:
    """Resolve the parameters of the current iter among the previous iter parameters.

    Parameters
    ----------
    pset_cur : np.ndarray
        Parameter set of the current iteration to resolve.
    parr_prev : np.ndarray
        Parameter array of the previous iteration.
    ffa_level : int
        Current FFA level (same level as pset_cur).
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tseg_brute : float
        Duration of the brute force segment.
    nbins : int
        Number of bins in the data.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter set index and the relative phase shift.

    Notes
    -----
    delta_t is the time difference between the reference time of the current segment
    (0) and the reference time of the previous segment.
    """
    nparams = len(pset_cur)
    if nparams == 1:
        delta_t = latter * 2 ** (ffa_level - 1) * tseg_brute
        pset_prev, delay = pset_cur, 0
    else:
        delta_t = (latter - 0.5) * 2 ** (ffa_level - 1) * tseg_brute
        pset_prev, delay = psr_utils.shift_params(pset_cur, delta_t)
    relative_phase = psr_utils.get_phase_idx(delta_t, pset_cur[-1], nbins, delay)
    pindex_prev = np.empty(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    nbins: int,
) -> tuple[np.ndarray, int]:
    """Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    coord_add : tuple[float, float]
        The coordinates of the added segment (level cur).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to coord_init, so we need to shift it to coord_add to get the
    resolved parameters index and relative phase shift.

    """
    nparams = len(param_arr)
    # distance between the current segment reference time and the global reference time
    tpoly = coord_add[0] - coord_init[0]

    kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
    kvec_cur[:-2] = leaf[:-3, 0]  # till acceleration
    kvec_new = psr_utils.shift_params(kvec_cur, tpoly)

    f0 = leaf[-3, 0]

    new_a = kvec_new[-3]
    new_f = f0 * (1 + kvec_new[-2] / C_VAL)
    delay = kvec_new[-1] / C_VAL

    relative_phase = psr_utils.get_phase_idx(tpoly, f0, nbins, delay)
    idx_a = np_utils.find_nearest_sorted_idx(param_arr[-2], new_a)
    idx_f = np_utils.find_nearest_sorted_idx(param_arr[-1], new_f)
    index_prev = np.empty(nparams, dtype=np.int64)
    index_prev[-1] = idx_f
    index_prev[-2] = idx_a
    return index_prev, relative_phase


@njit(cache=True, fastmath=True)
def split_taylor_params(
    param_cur: np.ndarray,
    dparam_cur: np.ndarray,
    dparam_new: np.ndarray,
    tseg_new: float,
    fold_bins: int,
    tol_bins: float,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    nparams = len(param_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(nparams, dtype=np.float64)
    shift_bins = psr_utils.poly_taylor_shift_d(
        dparam_cur,
        dparam_new,
        tseg_new,
        fold_bins,
        param_cur[-1],
        t_ref=tseg_new / 2,
    )

    for i in range(nparams):
        if shift_bins[i] > tol_bins:
            leaf_param, dparam_act = psr_utils.branch_param(
                param_cur[i],
                dparam_cur[i],
                dparam_new[i],
                param_limits[i][0],
                param_limits[i][1],
            )
        else:
            leaf_param, dparam_act = np.array([param_cur[i]]), dparam_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param)
    return common.get_leaves(leaf_params, leaf_dparams)


@njit(cache=True, fastmath=True)
def poly_taylor_branch(
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    param_set : np.ndarray
        Parameter set to branch. Shape: (nparams + 2, 2)
    tol : float
        Tolerance for the parameter step size in bins.
    tsamp : float
        Sampling time.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    _, scale_cur = coord_cur
    param_cur = param_set[0:-2, 0]
    dparam_cur = param_set[0:-2, 1]
    f0, _ = param_set[-2]
    t0, scale = param_set[-1]
    nparams = len(param_cur)

    duration = 2 * scale_cur
    dparam_opt = psr_utils.poly_taylor_step_d(
        nparams,
        duration,
        fold_bins,
        tol_bins,
        param_cur[-1],
        t_ref=duration / 2,
    )
    leafs_taylor = split_taylor_params(
        param_cur,
        dparam_cur,
        dparam_opt,
        duration,
        fold_bins,
        tol_bins,
        param_limits,
    )
    leaves = np.zeros((len(leafs_taylor), poly_order + 2, 2))
    leaves[:, :-2] = leafs_taylor
    leaves[:, -2, 0] = f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the leaf parameter sets for Taylor polynomials.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, period).
    dparams : np.ndarray
        Parameter step sizes for each dimension. Shape is (poly_order,).
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
        - coord_init[0] -> t0 (reference time)
        - coord_init[1] -> scale (duration of the segment)

    Returns
    -------
    np.ndarray
        The leaf parameter sets.

    """
    t0, scale = coord_init
    leafs_taylor = common.get_leaves(param_arr, dparams)
    leaves = np.zeros((len(leafs_taylor), poly_order + 2, 2), dtype=np.float64)
    leaves[:, :-2] = leafs_taylor
    leaves[:, -2, 0] = leafs_taylor[:, -1, 0]  # f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_suggest(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
) -> SuggestionStruct:
    """Generate a suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_period, 2, n_bins). Parameter dimensions are first two.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_arr : types.ListType
        Parameter values for each dimension (accel, period).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Taylor polynomial.
    score_func : _type_
        Function to score the folded data.

    Returns
    -------
    SuggestionStruct
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = scoring.snr_score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def generate_branching_pattern(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    isuggest: int = 0,
) -> np.ndarray:
    """Generate the branching pattern for the pruning Taylor search.

    Returns
    -------
    np.ndarray
        Branching pattern for the pruning Taylor search.
        1D array containing the initial number of trees and the subsequent branching
        pattern for each level.
    """
    poly_order = len(dparams)
    leaf = poly_taylor_leaves(param_arr, dparams, poly_order, (0, tchunk_ffa))[isuggest]
    branching_pattern = []
    for prune_level in range(1, nstages + 1):
        duration = tchunk_ffa * (prune_level + 1)
        coord_cur = (0, duration / 2)
        leaves_arr = poly_taylor_branch(
            leaf,
            coord_cur,
            fold_bins,
            tol_bins,
            poly_order,
            param_limits,
        )
        branching_pattern.append(len(leaves_arr))
        leaf = leaves_arr[isuggest]
    return np.array(branching_pattern)
