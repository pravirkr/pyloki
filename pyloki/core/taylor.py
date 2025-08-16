from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import np_utils, psr_utils
from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


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
    return common.brutefold_start_complex(
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
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, float]:
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
    tuple[np.ndarray, float]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    Notes
    -----
    leaf is referenced to coord_init, so we need to shift it to coord_add to get the
    resolved parameters index and relative phase shift.

    relative_phase is complete phase shift with fractional part.

    """
    nparams = len(param_arr)
    # distance between the current segment reference time and the global reference time
    delta_t = coord_add[0] - coord_init[0]
    kvec_new, delay = psr_utils.shift_params(leaf[:-2, 0], delta_t)
    freq_cur = leaf[-3, 0]
    relative_phase = psr_utils.get_phase_idx(delta_t, freq_cur, fold_bins, delay)

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
    freq_cur_batch = leaf_batch[:, -3, 0]
    kvec_new_batch, delay_batch = psr_utils.shift_params_batch(param_vec_batch, delta_t)
    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_cur_batch,
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
    freq_cur_batch = leaf_batch[:, -3, 0]
    snap_cur_batch = leaf_batch[:, 0, 0]
    dsnap_cur_batch = leaf_batch[:, 0, 1]
    accel_cur_batch = leaf_batch[:, 2, 0]
    mask = (
        (accel_cur_batch != 0)
        & (snap_cur_batch != 0)
        & ((-snap_cur_batch / accel_cur_batch) > 0)
        & (np.abs(snap_cur_batch / dsnap_cur_batch) > 5)
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
        kvec_new_batch[idx_circular] = kvec_new_circ
        delay_batch[idx_circular] = delay_circ

    if idx_normal.size > 0:
        kvec_new_norm, delay_norm = psr_utils.shift_params_batch(
            param_vec_batch[idx_normal],
            delta_t,
        )
        kvec_new_batch[idx_normal] = kvec_new_norm
        delay_batch[idx_normal] = delay_norm

    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_cur_batch,
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
def poly_taylor_validate_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a batch of leaf params.

    Filters out unphysical orbits. Currently removes cases with imaginary orbital
    frequency (i.e., -snap/accel <= 0).

    Parameters
    ----------
    leaves_batch : np.ndarray
        The leaf parameter sets to validate. Shape: (N, nparams + 2, 2)
    leaves_origins : np.ndarray
        The origins of the leaves. Shape: (N,)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered leaf parameter sets (only physically plausible) and their origins.
    """
    snap = leaves_batch[:, 0, 0]
    accel = leaves_batch[:, 2, 0]
    # omega_orb = -snap/accel > 0:
    # 1) snap=0 → omega_orb=0 (valid)
    # 2) snap≠0 & accel=0 → omega_orb=±∞ (allowed)
    # 3) snap≠0 & accel≠0 & -snap/accel>0 (valid)
    eps = 1e-15
    zero_case = (np.abs(snap) < eps) | (np.abs(accel) < eps)
    real_omega = (-snap / accel) >= 0
    mask = zero_case | real_omega
    idx = np.where(mask)[0]
    return leaves_batch[idx], leaves_origins[idx]


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
        if shift_bins[i] >= tol_bins:
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
    mask_2d = shift_bins_batch >= tol_bins  # Shape (n_batch, nparams)
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
def poly_taylor_suggest_complex(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStructComplex:
    """Generate a suggestion struct from a fold segment in FFT format.

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
    SuggestionStructComplex
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = scoring.snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def generate_branching_pattern_approx(
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

    This is a simplified version of the branching pattern that only tracks the
    worst-case branching factor.

    Returns
    -------
    np.ndarray
        Branching pattern for the pruning Taylor search.
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


@njit(cache=True, fastmath=True)
def generate_branching_pattern(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
) -> np.ndarray:
    """Generate the branching pattern for the pruning Taylor search.

    This tracks the exact number of branches per node to compute the average
    branching factor.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension.
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    param_limits : types.ListType[types.Tuple[float, float]]
        Parameter limits for each dimension.
    tchunk_ffa : float
        The duration of the starting segment.
    nstages : int
        The number of stages to generate the branching pattern for.
    fold_bins : int
        The number of bins in the frequency array.
    tol_bins : float
        The tolerance for the branching pattern.

    Returns
    -------
    np.ndarray
        The branching pattern for the pruning Taylor search.
    """
    poly_order = len(dparams)
    freq_arr = param_arr[-1]
    n0 = len(freq_arr)
    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams

    weights = np.ones(n0, dtype=np.int64)
    branching_pattern = np.empty(nstages, dtype=np.float64)
    param_ranges = np.array([(p_max - p_min) / 2 for p_min, p_max in param_limits])

    for prune_level in range(1, nstages + 1):
        tseg_cur = tchunk_ffa * (prune_level + 1)
        dparam_opt_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            tseg_cur,
            fold_bins,
            tol_bins,
            freq_arr,
            t_ref=tseg_cur / 2,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_opt_batch,
            tseg_cur,
            fold_bins,
            freq_arr,
            t_ref=tseg_cur / 2,
        )

        nfreq = len(freq_arr)
        dparam_next_tmp = np.empty((nfreq, poly_order), dtype=np.float64)
        n_branch_freq = np.ones(nfreq, dtype=np.int64)
        n_branch_nonfreq = np.ones(nfreq, dtype=np.int64)

        # Pre-compute masks and ratios
        needs_branching = shift_bins_batch >= tol_bins
        too_large_step = dparam_opt_batch > param_ranges[np.newaxis, :]

        # Vectorized branching decision
        needs_branching = shift_bins_batch >= tol_bins
        too_large_step = dparam_opt_batch > param_ranges[np.newaxis, :]

        weighted_sum = 0.0
        total_weight = 0
        total_freq_branches = 0

        for i in range(nfreq):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                num_points = max(
                    1,
                    int(np.ceil(dparam_cur_batch[i, j] / dparam_opt_batch[i, j])),
                )
                if j == poly_order - 1:
                    n_branch_freq[i] = num_points
                else:
                    n_branch_nonfreq[i] *= num_points
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

            total_weight += weights[i]
            weighted_sum += weights[i] * (n_branch_nonfreq[i] * n_branch_freq[i])
            total_freq_branches += n_branch_freq[i]

        branching_pattern[prune_level - 1] = weighted_sum / total_weight
        freq_arr_next = np.empty(total_freq_branches, dtype=np.float64)
        weights_next = np.empty(total_freq_branches, dtype=np.int64)
        dparam_cur_next = np.empty((total_freq_branches, poly_order), dtype=np.float64)

        pos = 0
        for i in range(nfreq):
            cfreq = n_branch_freq[i]
            weight = weights[i] * n_branch_nonfreq[i]
            if cfreq == 1:
                freq_arr_next[pos] = freq_arr[i]
                weights_next[pos] = weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                pos += 1
            elif cfreq == 2:
                dparam_cur_freq = dparam_cur_batch[i, poly_order - 1]
                delta = 0.25 * dparam_cur_freq
                f = freq_arr[i]
                freq_arr_next[pos] = f - delta
                weights_next[pos] = weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                pos += 1
                freq_arr_next[pos] = f + delta
                weights_next[pos] = weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                pos += 1
            else:
                msg = f"cfreq == {cfreq} is not supported"
                raise ValueError(msg)

        freq_arr = freq_arr_next
        dparam_cur_batch = dparam_cur_next
        weights = weights_next

    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_branching_pattern_circular(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
) -> np.ndarray:
    poly_order = len(dparams)
    if poly_order != 4:
        msg = "Circular branching pattern requires exactly 4 parameters."
        raise ValueError(msg)

    freq_arr = param_arr[-1]
    n0 = len(freq_arr)

    # Broadcast current dparams to each freq row
    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams

    param_ranges = np.array([(p_max - p_min) / 2 for p_min, p_max in param_limits])
    weights = np.ones(n0, dtype=np.float64)
    branching_pattern = np.empty(nstages, dtype=np.float64)

    for prune_level in range(1, nstages + 1):
        tseg_cur = tchunk_ffa * (prune_level + 1)

        dparam_opt_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            tseg_cur,
            fold_bins,
            tol_bins,
            freq_arr,
            t_ref=tseg_cur / 2,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_opt_batch,
            tseg_cur,
            fold_bins,
            freq_arr,
            t_ref=tseg_cur / 2,
        )

        nfreq = len(freq_arr)
        dparam_next_tmp = np.empty((nfreq, poly_order), dtype=np.float64)
        n_branch_freq = np.ones(nfreq, dtype=np.int64)
        n_branch_accel = np.ones(nfreq, dtype=np.int64)
        n_branch_jerk = np.ones(nfreq, dtype=np.int64)
        n_branch_snap = np.ones(nfreq, dtype=np.int64)

        needs_branching = shift_bins_batch >= tol_bins
        too_large_step = dparam_opt_batch > param_ranges[np.newaxis, :]

        weighted_sum = 0.0
        total_weight = 0.0
        total_freq_branches = 0

        # First pass: determine branching counts, update dparams, compute stats
        for i in range(nfreq):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                num_points = max(
                    1,
                    int(np.ceil(dparam_cur_batch[i, j] / dparam_opt_batch[i, j])),
                )
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

                if j == 0:
                    n_branch_snap[i] = num_points
                elif j == 1:
                    n_branch_jerk[i] = num_points
                elif j == 2:
                    n_branch_accel[i] = num_points
                else:
                    n_branch_freq[i] = num_points

            # Expected validated non-frequency combinations
            raw_nonfreq_combinations = (
                float(n_branch_snap[i])
                * float(n_branch_jerk[i])
                * float(n_branch_accel[i])
            )
            # if snap is branching, assume symmetric branching around current value, if
            # snap is not branching, assume all combinations valid
            validation_fraction = 1.0 if n_branch_snap[i] == 1 else 0.5

            valid_combinations = validation_fraction * raw_nonfreq_combinations
            total_weight += weights[i]
            weighted_sum += weights[i] * (valid_combinations * float(n_branch_freq[i]))
            total_freq_branches += n_branch_freq[i]

        branching_pattern[prune_level - 1] = weighted_sum / total_weight
        freq_arr_next = np.empty(total_freq_branches, dtype=np.float64)
        weights_next = np.empty(total_freq_branches, dtype=np.float64)
        dparam_cur_next = np.empty((total_freq_branches, poly_order), dtype=np.float64)

        pos = 0
        for i in range(nfreq):
            cfreq = n_branch_freq[i]
            validation_fraction = 1.0 if n_branch_snap[i] == 1 else 0.5
            raw_nonfreq = (
                float(n_branch_snap[i])
                * float(n_branch_jerk[i])
                * float(n_branch_accel[i])
            )
            valid_weight = weights[i] * validation_fraction * raw_nonfreq

            if cfreq == 1:
                freq_arr_next[pos] = freq_arr[i]
                weights_next[pos] = valid_weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                pos += 1
            elif cfreq == 2:
                dparam_cur_freq = dparam_cur_batch[i, poly_order - 1]
                confidence_const = 0.5 * (1.0 + 1.0 / float(cfreq))
                half_range = confidence_const * dparam_cur_freq
                f_center = freq_arr[i]

                freq_arr_next[pos] = f_center - half_range * 0.5
                freq_arr_next[pos + 1] = f_center + half_range * 0.5

                weights_next[pos] = valid_weight
                weights_next[pos + 1] = valid_weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                dparam_cur_next[pos + 1] = dparam_next_tmp[i]
                pos += 2
            else:
                msg = f"cfreq == {cfreq} is not supported"
                raise ValueError(msg)

        # Advance to next stage
        freq_arr = freq_arr_next
        dparam_cur_batch = dparam_cur_next
        weights = weights_next

    return branching_pattern
