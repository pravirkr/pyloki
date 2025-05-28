from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import np_utils, psr_utils
from pyloki.utils.suggestion import SuggestionStruct


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
    return common.brutefold_start(
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
) -> tuple[np.ndarray, int]:
    """Resolve the params to find the closest index in grid and phase shift.

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
    tuple[np.ndarray, int]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

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
    relative_phase = psr_utils.get_phase_idx(delta_t, pset_cur[-1], fold_bins, delay)
    pindex_prev = np.zeros(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(param_arr[ip], pset_prev[ip])
    return pindex_prev, relative_phase


@njit(cache=True, fastmath=True)
def ffa_taylor_resolve_fft(
    pset_cur: np.ndarray,
    param_arr: types.ListType[types.Array],
    ffa_level: int,
    latter: int,
    tseg_brute: float,
    fold_bins: int,
) -> tuple[np.ndarray, float]:
    nparams = len(pset_cur)
    if nparams == 1:
        delta_t = latter * 2 ** (ffa_level - 1) * tseg_brute
        pset_prev, delay = pset_cur, 0
    else:
        delta_t = (latter - 0.5) * 2 ** (ffa_level - 1) * tseg_brute
        pset_prev, delay = psr_utils.shift_params(pset_cur, delta_t)
    relative_phase = psr_utils.get_phase_idx_complete(
        delta_t,
        pset_cur[-1],
        fold_bins,
        delay,
    )
    pindex_prev = np.zeros(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(param_arr[ip], pset_prev[ip])
    return pindex_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, int]:
    """Resolve the leaf params to find the closest index in grid and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set (shape: (nparams + 2, 2)).
    coord_add : tuple[float, float]
        The coordinates for the added segment (level current).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_arr : types.ListType[types.Array]
        Parameter grid array for the ``coord_add`` segment (dim: 2)
    fold_bins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    Notes
    -----
    leaf is referenced to coord_init, so we need to shift it to coord_add to get the
    resolved parameters index and relative phase shift.

    """
    nparams = len(param_arr)
    # distance between the current segment reference time and the global reference time
    delta_t = coord_add[0] - coord_init[0]
    kvec_new, delay = psr_utils.shift_params(leaf[:-2, 0], delta_t)
    freq_old = leaf[-3, 0]
    relative_phase = psr_utils.get_phase_idx(delta_t, freq_old, fold_bins, delay)

    param_idx = np.zeros(nparams, dtype=np.int64)
    param_idx[-1] = np_utils.find_nearest_sorted_idx(param_arr[-1], kvec_new[-1])
    param_idx[-2] = np_utils.find_nearest_sorted_idx(param_arr[-2], kvec_new[-2])
    return param_idx, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_batch(
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_leaves = len(leaf_batch)
    nparams = len(param_arr)
    delta_t = coord_add[0] - coord_init[0]
    param_vec_batch = leaf_batch[:, :-2]  # Take last dimension as it is
    freq_old_batch = leaf_batch[:, -3, 0]
    kvec_new_batch, delay_batch = psr_utils.shift_params_batch(param_vec_batch, delta_t)
    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_old_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_leaves, nparams), dtype=np.int64)
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        kvec_new_batch[:, -1, 0],
    )
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        kvec_new_batch[:, -2, 0],
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_snap_batch(
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_leaves = len(leaf_batch)
    nparams = len(param_arr)
    delta_t = coord_add[0] - coord_init[0]
    param_vec_batch = leaf_batch[:, :-2]  # Take last dimension as it is
    freq_old_batch = leaf_batch[:, -3, 0]
    snap_old_batch = leaf_batch[:, 0, 0]
    dsnap_old_batch = leaf_batch[:, 0, 1]
    accel_old_batch = leaf_batch[:, 2, 0]
    mask = (
        (accel_old_batch != 0)
        & (snap_old_batch != 0)
        & ((-snap_old_batch / accel_old_batch) > 0)
        & (np.abs(snap_old_batch / dsnap_old_batch) > 5)
    )
    idx_circular = np.where(mask)[0]
    idx_normal = np.where(~mask)[0]
    kvec_new_batch = np.empty_like(param_vec_batch)
    delay_batch = np.empty(n_leaves, dtype=param_vec_batch.dtype)
    if idx_circular.size > 0:
        kvec_new_circ, delay_circ = psr_utils.shift_params_circular_batch(
            param_vec_batch[idx_circular],
            delta_t,
        )
        if kvec_new_circ.shape == kvec_new_batch[idx_circular].shape:
            kvec_new_batch[idx_circular] = kvec_new_circ
            delay_batch[idx_circular] = delay_circ
        else:
            msg = "kvec_new_circ.shape != kvec_new_batch[idx_circular].shape"
            raise ValueError(msg)

    if idx_normal.size > 0:
        kvec_new_norm, delay_norm = psr_utils.shift_params_batch(
            param_vec_batch[idx_normal],
            delta_t,
        )
        if kvec_new_norm.shape == kvec_new_batch[idx_normal].shape:
            kvec_new_batch[idx_normal] = kvec_new_norm
            delay_batch[idx_normal] = delay_norm
        else:
            msg = "kvec_new_norm.shape != kvec_new_batch[idx_normal].shape"
            raise ValueError(msg)

    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_old_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_leaves, nparams), dtype=np.int64)
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        kvec_new_batch[:, -1, 0],
    )
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        kvec_new_batch[:, -2, 0],
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def split_taylor_params(
    param_cur: np.ndarray,
    dparam_cur: np.ndarray,
    dparam_new: np.ndarray,
    tseg_cur: float,
    fold_bins: int,
    tol_bins: float,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    nparams = len(param_cur)
    shift_bins = psr_utils.poly_taylor_shift_d(
        dparam_cur,
        dparam_new,
        tseg_cur,
        fold_bins,
        param_cur[-1],
        t_ref=tseg_cur / 2,
    )

    total_size = 1
    leaf_params = typed.List.empty_list(types.float64[::1])
    leaf_dparams = np.empty(nparams, dtype=np.float64)
    shapes = np.empty(nparams, dtype=np.int64)
    for i in range(nparams):
        if shift_bins[i] > tol_bins:
            leaf_param_arr, dparam_act = psr_utils.branch_param(
                param_cur[i],
                dparam_cur[i],
                dparam_new[i],
                param_limits[i][0],
                param_limits[i][1],
            )
        else:
            leaf_param_arr = np.array([param_cur[i]], dtype=np.float64)
            dparam_act = dparam_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param_arr)
        shapes[i] = len(leaf_param_arr)
        total_size *= shapes[i]

    # Allocate and populate the final array directly
    leaves_taylor = np.empty((total_size, nparams, 2), dtype=np.float64)
    leaves_taylor[:, :, 1] = leaf_dparams
    # Fill column 0 (parameter values) using Cartesian product logic
    for i in range(total_size):
        idx = i
        # last dimension changes fastest
        for j in range(nparams - 1, -1, -1):
            arr = leaf_params[j]
            arr_idx = idx % shapes[j]
            leaves_taylor[i, j, 0] = arr[arr_idx]
            idx //= shapes[j]
    return leaves_taylor


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
        Parameter set to branch. Shape: (nparams + 2, 2).
    coord_cur : tuple[float, float]
        The coordinates for the current segment.
    fold_bins : int
        Number of bins in the folded profile.
    tol_bins : float
        Tolerance for the parameter step size in bins.
    poly_order : int
        The order of the Taylor polynomial.
    param_limits : types.ListType[types.Tuple[float, float]]
        The limits for each parameter.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets (shape: (nbranch, nparams + 2, 2)).
    """
    _, scale_cur = coord_cur
    nparams = poly_order
    param_cur = param_set[0:-2, 0]
    dparam_cur = param_set[0:-2, 1]
    f0 = param_set[-2, 0]
    t0 = param_set[-1, 0]
    scale = param_set[-1, 1]
    f_max = param_cur[-1]

    tseg_cur = 2 * scale_cur
    dparam_opt = psr_utils.poly_taylor_step_d(
        nparams,
        tseg_cur,
        fold_bins,
        tol_bins,
        f_max,
        t_ref=tseg_cur / 2,
    )
    leafs_taylor = split_taylor_params(
        param_cur,
        dparam_cur,
        dparam_opt,
        tseg_cur,
        fold_bins,
        tol_bins,
        param_limits,
    )
    leaves = np.zeros((len(leafs_taylor), poly_order + 2, 2), dtype=np.float64)
    leaves[:, :-2] = leafs_taylor
    leaves[:, -2, 0] = f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_branch_batch(
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_set_batch)
    nparams = poly_order
    _, scale_cur = coord_cur
    param_cur_batch = param_set_batch[:, :-2, 0]
    dparam_cur_batch = param_set_batch[:, :-2, 1]
    f0_batch = param_set_batch[:, -2, 0]
    t0_batch = param_set_batch[:, -1, 0]
    scale_batch = param_set_batch[:, -1, 1]
    f_max_batch = param_cur_batch[:, -1]

    tseg_cur = 2 * scale_cur
    dparam_opt_batch = psr_utils.poly_taylor_step_d_vec(
        nparams,
        tseg_cur,
        fold_bins,
        tol_bins,
        f_max_batch,
        t_ref=tseg_cur / 2,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_opt_batch,
        tseg_cur,
        fold_bins,
        f_max_batch,
        t_ref=tseg_cur / 2,
    )

    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, nparams, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, nparams), dtype=np.float64)
    branched_counts = np.empty((n_batch, nparams), dtype=np.int64)
    for i in range(n_batch):
        for j in range(nparams):
            p_min, p_max = param_limits[j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_opt_batch[i, j],
                p_min,
                p_max,
            )
            pad_branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # --- Vectorized Selection ---
    # Select based on mask: shape (n_batch, nparams, 1)
    mask_2d = shift_bins_batch > tol_bins  # Shape (n_batch, nparams)
    for i in range(n_batch):
        for j in range(nparams):
            if not mask_2d[i, j]:
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- Optimized Padded Cartesian Product ---
    batch_leaves_taylor, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        nparams,
    )
    total_leaves = len(batch_origins)
    batch_leaves = np.zeros((total_leaves, poly_order + 2, 2), dtype=np.float64)
    batch_leaves[:, :-2, 0] = batch_leaves_taylor
    batch_leaves[:, :-2, 1] = pad_branched_dparams[batch_origins]
    batch_leaves[:, -2, 0] = f0_batch[batch_origins]
    batch_leaves[:, -1, 0] = t0_batch[batch_origins]
    batch_leaves[:, -1, 1] = scale_batch[batch_origins]
    return batch_leaves, batch_origins


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
    score_widths: np.ndarray,
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
        scores[iparam] = scoring.snr_score_func(data[iparam], score_widths)
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
