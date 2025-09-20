from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core.common import get_leaves
from pyloki.detection.scoring import snr_score_func, snr_score_func_complex
from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@njit(cache=True, fastmath=True)
def poly_taylor_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the leaf parameter sets for Taylor polynomial search.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, frequency).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension. Shape is (poly_order,).
        Order is reversed [..., acc, freq].
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
        - coord_init[0] -> t0 (reference time) measured from t=0
        - coord_init[1] -> scale (half duration of the segment)

    Returns
    -------
    np.ndarray
        The leaf parameter sets. Shape is (n_param_sets, poly_order + 2, 2).

    Notes
    -----
    Conventions for each leaf parameter set:
    leaf[:-1, 0] -> Taylor polynomial coefficients,
                    order is [d_poly_order, ..., d_1, d_0]
    leaf[:-1, 1] -> Grid size (error) on each coefficient,
    leaf[-1, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf[-1, 1]  -> Flag to indicate basis change (placeholder for now)
    """
    _, _ = coord_init
    leaves_taylor = get_leaves(param_arr, dparams)
    f0_batch = leaves_taylor[:, -1, 0]
    df_batch = leaves_taylor[:, -1, 1]
    leaves = np.zeros((len(leaves_taylor), poly_order + 2, 2), dtype=np.float64)
    # Copy till accel
    leaves[:, :-3] = leaves_taylor[:, :-1]
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    leaves[:, -3, 0] = 0
    leaves[:, -3, 1] = df_batch * (C_VAL / f0_batch)
    # intialize d0 (measure from t=t_init)
    leaves[:, -2, 0] = 0  # we never branch on d0
    # Store f0
    leaves[:, -1, 0] = f0_batch
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
    """Generate a Taylor suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_freq, 2, n_bins). Parameter dimensions are first two.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_arr : types.ListType
        Parameter values for each dimension (accel, freq).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Taylor polynomial.
    score_widths : np.ndarray
        Boxcar widths for the score computation.

    Returns
    -------
    SuggestionStruct
        Suggestion struct
        - param_sets: The parameter sets (n_param_sets, poly_order + 2, 2).
        - data: The folded data for each leaf.
        - scores: The scores for each leaf.
        - backtracks: The backtracks for each leaf.
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks, "taylor")


@njit(cache=True, fastmath=True)
def poly_taylor_suggest_complex(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStructComplex:
    """Generate a Taylor suggestion struct from a fold segment in Fourier domain.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_freq, 2, n_bins_f). Parameter dimensions are first two.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_arr : types.ListType
        Parameter values for each dimension (accel, freq).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Taylor polynomial.
    score_widths : np.ndarray
        Boxcar widths for the score computation.

    Returns
    -------
    SuggestionStructComplex
        Suggestion struct in Fourier domain.
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks, "taylor")


@njit(cache=True, fastmath=True)
def poly_taylor_branch(
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    leaf : np.ndarray
        Parameter set (leaf) to branch. Shape: (n_params + 2, 2).
    coord_cur : tuple[float, float]
        Coordinates for the accumulated segment in the current stage.
    fold_bins : int
        Number of bins in the folded profile.
    tol_bins : float
        Tolerance for the parameter step size in bins.
    poly_order : int
        The order of the Taylor polynomial.
    param_limits : types.ListType[types.Tuple[float, float]]
        The limits for each parameter in Taylor basis (reverse order).

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets. Shape: (n_branch, n_params + 2, 2).
    """
    _, t_obs_minus_t_ref = coord_cur
    param_cur = leaf[:-2, 0]
    dparam_cur = leaf[:-2, 1]
    d0_cur = leaf[-2, 0]
    f0 = leaf[-1, 0]

    param_limits_d = np.empty((poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[i, 0] = param_limits[i][0]
        param_limits_d[i, 1] = param_limits[i][1]
    param_limits_d[-1, 0] = (1 - param_limits[poly_order - 1][1] / f0) * C_VAL
    param_limits_d[-1, 1] = (1 - param_limits[poly_order - 1][0] / f0) * C_VAL

    dparam_new = psr_utils.poly_taylor_step_d(
        poly_order,
        t_obs_minus_t_ref,
        fold_bins,
        tol_bins,
        f0,
        t_ref=0,
    )
    shift_bins = psr_utils.poly_taylor_shift_d(
        dparam_cur,
        dparam_new,
        t_obs_minus_t_ref,
        fold_bins,
        f0,
        t_ref=0,
    )

    eps = 1e-6
    total_size = 1
    leaf_branch_params = typed.List.empty_list(types.float64[::1])
    leaf_branch_dparams = np.empty(poly_order, dtype=np.float64)
    shapes = np.empty(poly_order, dtype=np.int64)
    for i in range(poly_order):
        param_min, param_max = param_limits_d[i]
        if shift_bins[i] >= (tol_bins - eps):
            leaf_param_arr, dparam_act = psr_utils.branch_param(
                param_cur[i],
                dparam_cur[i],
                dparam_new[i],
                param_min,
                param_max,
            )
        else:
            leaf_param_arr = np.array([param_cur[i]], dtype=np.float64)
            dparam_act = dparam_cur[i]
        leaf_branch_dparams[i] = dparam_act
        leaf_branch_params.append(leaf_param_arr)
        shapes[i] = len(leaf_param_arr)
        total_size *= shapes[i]

    # Allocate and populate the final array directly
    leaves_branch_taylor = np.empty((total_size, poly_order, 2), dtype=np.float64)
    leaves_branch_taylor[:, :, 1] = leaf_branch_dparams
    # Fill column 0 (parameter values) using Cartesian product logic
    for i in range(total_size):
        idx = i
        # last dimension changes fastest
        for j in range(poly_order - 1, -1, -1):
            arr = leaf_branch_params[j]
            arr_idx = idx % shapes[j]
            leaves_branch_taylor[i, j, 0] = arr[arr_idx]
            idx //= shapes[j]

    leaves_branch = np.zeros((total_size, poly_order + 2, 2), dtype=np.float64)
    leaves_branch[:, :-2] = leaves_branch_taylor
    leaves_branch[:, -2, 0] = d0_cur
    leaves_branch[:, -1, 0] = f0
    return leaves_branch


@njit(cache=True, fastmath=True)
def poly_taylor_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch, _, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur
    param_cur_batch = leaves_batch[:, :-2, 0]
    dparam_cur_batch = leaves_batch[:, :-2, 1]
    d0_cur_batch = leaves_batch[:, -2, 0]
    f0_batch = leaves_batch[:, -1, 0]

    param_limits_d = np.empty((n_batch, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL

    dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
        poly_order,
        t_obs_minus_t_ref,
        fold_bins,
        tol_bins,
        f0_batch,
        t_ref=0,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_new_batch,
        t_obs_minus_t_ref,
        fold_bins,
        f0_batch,
        t_ref=0,
    )
    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, poly_order, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, poly_order), dtype=np.float64)
    branched_counts = np.empty((n_batch, poly_order), dtype=np.int64)
    for i in range(n_batch):
        for j in range(poly_order):
            param_min, param_max = param_limits_d[i, j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
                param_min,
                param_max,
            )
            pad_branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # --- Vectorized Selection ---
    eps = 1e-6  # Small tolerance for floating-point comparison
    needs_branching = shift_bins_batch >= (tol_bins - eps)
    for i in range(n_batch):
        for j in range(poly_order):
            if not needs_branching[i, j]:
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- Optimized Padded Cartesian Product ---
    leaves_branch_taylor_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        poly_order,
    )
    total_leaves = len(batch_origins)
    leaves_branch_batch = np.zeros((total_leaves, poly_order + 2, 2), dtype=np.float64)
    leaves_branch_batch[:, :-2, 0] = leaves_branch_taylor_batch
    leaves_branch_batch[:, :-2, 1] = pad_branched_dparams[batch_origins]
    leaves_branch_batch[:, -2, 0] = d0_cur_batch[batch_origins]
    leaves_branch_batch[:, -1, 0] = f0_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_branch_circular_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch, _, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur
    param_cur_batch = leaves_batch[:, :-2, 0]
    dparam_cur_batch = leaves_batch[:, :-2, 1]
    d0_cur_batch = leaves_batch[:, -2, 0]
    f0_batch = leaves_batch[:, -1, 0]

    param_limits_d = np.empty((n_batch, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL

    dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
        poly_order,
        t_obs_minus_t_ref,
        fold_bins,
        tol_bins,
        f0_batch,
        t_ref=0,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_new_batch,
        t_obs_minus_t_ref,
        fold_bins,
        f0_batch,
        t_ref=0,
    )
    # --- Vectorized Padded Branching ( All params except crackle) ---
    pad_branched_params = np.empty((n_batch, poly_order, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, poly_order), dtype=np.float64)
    branched_counts = np.empty((n_batch, poly_order), dtype=np.int64)
    for i in range(n_batch):
        for j in range(1, poly_order):  # skip crackle (j=0)
            param_min, param_max = param_limits_d[i, j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
                param_min,
                param_max,
            )
            pad_branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # --- Vectorized Selection (mask non-crackle branched params)---
    eps = 1e-6  # Small tolerance for floating-point comparison
    needs_branching = shift_bins_batch >= (tol_bins - eps)
    for i in range(n_batch):
        for j in range(poly_order):
            if not needs_branching[i, j] or j == 0:
                # crackle - don't branch yet, keep at current value
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- First Cartesian Product (All Except Crackle) ---
    leaves_branch_taylor_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        poly_order,
    )
    total_intermediate_leaves = len(batch_origins)
    leaves_intermediate = np.zeros(
        (total_intermediate_leaves, poly_order + 2, 2),
        dtype=np.float64,
    )
    leaves_intermediate[:, :-2, 0] = leaves_branch_taylor_batch
    leaves_intermediate[:, :-2, 1] = pad_branched_dparams[batch_origins]
    leaves_intermediate[:, -2, 0] = d0_cur_batch[batch_origins]
    leaves_intermediate[:, -1, 0] = f0_batch[batch_origins]

    # --- Clasify Intermediate Leaves into Circular and Non-Circular ---
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask(
        leaves_intermediate,
        snap_threshold=5,
    )
    if idx_circular_crackle.size == 0:
        # No crackle branching needed, return intermediate leaves
        return leaves_intermediate, batch_origins

    # Branch crackle for idx_circular_crackle cases
    crackle_branch_leaves = leaves_intermediate[idx_circular_crackle]
    crackle_origins = batch_origins[idx_circular_crackle]
    n_crackle_branch = len(idx_circular_crackle)

    # Branch crackle parameter for these specific leaves
    crackle_branched_params = np.empty((n_crackle_branch, branch_max), dtype=np.float64)
    crackle_branched_dparams = np.empty(n_crackle_branch, dtype=np.float64)
    crackle_branched_counts = np.empty(n_crackle_branch, dtype=np.int64)

    for i in range(n_crackle_branch):
        orig_batch_idx = crackle_origins[i]
        param_min, param_max = param_limits_d[orig_batch_idx, 0]  # crackle limits
        dparam_act, count = psr_utils.branch_param_padded(
            crackle_branched_params[i],
            crackle_branch_leaves[i, 0, 0],
            crackle_branch_leaves[i, 0, 1],
            dparam_new_batch[orig_batch_idx, 0],
            param_min,
            param_max,
        )
        crackle_branched_dparams[i] = dparam_act
        crackle_branched_counts[i] = count

    for i in range(n_crackle_branch):
        orig_batch_idx = crackle_origins[i]
        # Check if crackle actually needs branching
        if not needs_branching[orig_batch_idx, 0]:
            crackle_branched_params[i, :] = 0
            crackle_branched_params[i, 0] = crackle_branch_leaves[i, 0, 0]
            crackle_branched_dparams[i] = crackle_branch_leaves[i, 0, 1]
            crackle_branched_counts[i] = 1

    # Create new leaves with branched crackle
    total_crackle_branches = np.sum(crackle_branched_counts)
    # Keep non-crackle-branching leaves as-is
    keep_indices = np.concatenate((idx_circular_snap, idx_taylor))
    total_leaves = len(keep_indices) + total_crackle_branches
    leaves_final = np.empty((total_leaves, poly_order + 2, 2), dtype=np.float64)
    origins_final = np.empty(total_leaves, dtype=np.int64)

    n_keep = len(keep_indices)
    if n_keep > 0:
        leaves_final[:n_keep] = leaves_intermediate[keep_indices]
        origins_final[:n_keep] = batch_origins[keep_indices]

    current_idx = n_keep
    for i in range(n_crackle_branch):
        count_i = crackle_branched_counts[i]
        orig_leaf = crackle_branch_leaves[i]
        orig_batch_idx = crackle_origins[i]
        # Vectorized leaf copying
        end_idx = current_idx + count_i
        leaves_final[current_idx:end_idx] = orig_leaf  # Broadcast copy
        leaves_final[current_idx:end_idx, 0, 0] = crackle_branched_params[i, :count_i]
        leaves_final[current_idx:end_idx, 0, 1] = crackle_branched_dparams[i]
        origins_final[current_idx:end_idx] = orig_batch_idx
        current_idx = end_idx

    return leaves_final, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_validate_circular_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    p_orb_min: float,
    snap_threshold: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a batch of leaf params.

    Filters out unphysical orbits. Currently removes cases with imaginary orbital
    frequency (i.e., -snap/accel <= 0) and orbital frequency > omega_orb_max.

    Parameters
    ----------
    leaves_batch : np.ndarray
        The leaf parameter sets to validate. Shape: (N, nparams + 2, 2)
    leaves_origins : np.ndarray
        The origins of the leaves. Shape: (N,)
    p_orb_min : float
        Minimum allowed orbital period.
    snap_threshold: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered leaf parameter sets (only physically plausible) and their origins.
    """
    eps = 1e-12
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    accel = leaves_batch[:, 3, 0]
    omega_orb_max_sq = (2 * np.pi / p_orb_min) ** 2
    # omega_orb = -snap/accel > 0:
    # 1) snap=0 → omega_orb=0 (valid)
    # 2) snap≠0 & accel=0 → omega_orb=±∞ (allowed)
    # 3) snap≠0 & accel≠0 & -snap/accel>0 (valid)

    omega_sq = -snap / (accel + eps)

    # Physically usable via s/a
    is_usable = (np.abs(accel) > eps) & (np.abs(snap) > eps) & (omega_sq > 0.0)
    # Numerically degenerate but not obviously unphysical
    is_zero = (np.abs(snap) <= eps) | (np.abs(accel) <= eps)

    # Within maximum orbital frequency limit
    is_within_omega_limit = omega_sq <= omega_orb_max_sq

    # Delays circular validation until snap is well-measured.
    is_significant = np.abs(snap / (dsnap + eps)) > snap_threshold

    # No filtering for each leaf until its snap is significant
    mask = ~is_significant | (is_zero | (is_usable & is_within_omega_limit))
    idx = np.where(mask)[0]
    return leaves_batch[idx], leaves_origins[idx]


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the leaf params to find the closest index in grid and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set (shape: (poly_order + 2, 2)).
    coord_add : tuple[float, float]
        The coordinates for the added segment (level current).
    coord_cur : tuple[float, float]
        The coordinates for the current pruning suggestion tree.
    coord_init : tuple[float, float]
        The coordinates for the initial pruning suggestion tree.
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
    leaf is referenced to coord_cur, so we need to shift it to coord_add to get
    the resolved parameters index and relative phase shift. We also need to correct for
    the tree phase offset from coord_init to coord_cur.

    relative_phase is complete phase shift with fractional part.
    """
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_set = leaf[:-1, 0]
    f0 = leaf[-1, 0]

    dvec_t_add = transforms.shift_taylor_params(param_set, t0_add - t0_cur)
    dvec_t_init = transforms.shift_taylor_params(param_set, t0_init - t0_cur)
    accel_new = dvec_t_add[-3]
    vel_new = dvec_t_add[-2] - dvec_t_init[-2]
    freq_new = f0 * (1 - vel_new / C_VAL)
    delay = (dvec_t_add[-1] - dvec_t_init[-1]) / C_VAL
    relative_phase = psr_utils.get_phase_idx(t0_add - t0_init, f0, fold_bins, delay)
    param_idx = np.zeros(len(param_arr), dtype=np.int64)
    param_idx[-2] = np_utils.find_nearest_sorted_idx(param_arr[-2], accel_new)
    param_idx[-1] = np_utils.find_nearest_sorted_idx(param_arr[-1], freq_new)
    return param_idx, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_set_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    dvec_t_add = transforms.shift_taylor_params(param_set_batch, t0_add - t0_cur)
    dvec_t_init = transforms.shift_taylor_params(param_set_batch, t0_init - t0_cur)
    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2] - dvec_t_init[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = (dvec_t_add[:, -1] - dvec_t_init[:, -1]) / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t0_add - t0_init,
        f0_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_batch, len(param_arr)), dtype=np.int64)
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        accel_new_batch,
    )
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        freq_new_batch,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def get_circular_mask(
    leaves_batch: np.ndarray,
    snap_threshold: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a robust mask to identify circular orbit candidates.

    Filters out physically implausible and numerically unstable orbits.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Shape (n_batch, nparams + 2, 2)
    snap_threshold: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a high-quality circular orbit candidate.

    """
    eps = 1e-12
    crackle = leaves_batch[:, 0, 0]
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    jerk = leaves_batch[:, 2, 0]
    accel = leaves_batch[:, 3, 0]

    # Delays circular classification until snap is well-measured.
    is_significant = np.abs(snap / (dsnap + eps)) > snap_threshold

    # Numerical Stability - acceleration and snap must be significantly non-zero
    is_stable = (np.abs(accel) > eps) & (np.abs(snap) > eps)

    # Physicality - for Omega^2 = -s/a to be positive, s and a must have opposite signs
    omega_sq = -snap / (accel + eps)
    is_physical = omega_sq > 0
    # Classification logic
    stable_and_significant = is_stable & is_significant
    # idx_circular_snap: stable, significant, and physical
    mask_circular_snap = stable_and_significant & is_physical

    # For unstable but significant cases, check crackle/jerk approximation
    unstable_and_significant = (~is_stable) & is_significant
    crackle_jerk_nonzero = (np.abs(crackle) > eps) & (np.abs(jerk) > eps)
    omega_sq_crackle_jerk = -crackle / (jerk + eps)
    crackle_jerk_physical = omega_sq_crackle_jerk > 0

    # idx_circular_crackle: unstable snap/accel but stable crackle/jerk
    mask_circular_crackle = (
        unstable_and_significant & crackle_jerk_nonzero & crackle_jerk_physical
    )

    # idx_taylor: everything else (no hope for circularity)
    mask_taylor = ~(mask_circular_snap | mask_circular_crackle)

    idx_circular_snap = np.where(mask_circular_snap)[0]
    idx_circular_crackle = np.where(mask_circular_crackle)[0]
    idx_taylor = np.where(mask_taylor)[0]

    return idx_circular_snap, idx_circular_crackle, idx_taylor


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_circular_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_batch, _, _ = leaves_batch.shape
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_set_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask(
        leaves_batch,
        snap_threshold=5,
    )
    dvec_t_add = np.empty((n_batch, 6), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 6), dtype=np.float64)

    if idx_circular_snap.size > 0:
        dvec_t_add_circ_snap = transforms.shift_taylor_params_circular_batch(
            param_set_batch[idx_circular_snap],
            t0_add - t0_cur,
        )
        dvec_t_init_circ_snap = transforms.shift_taylor_params_circular_batch(
            param_set_batch[idx_circular_snap],
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular_snap] = dvec_t_add_circ_snap
        dvec_t_init[idx_circular_snap] = dvec_t_init_circ_snap

    if idx_circular_crackle.size > 0:
        dvec_t_add_circ_crackle = transforms.shift_taylor_params_circular_crackle_batch(
            param_set_batch[idx_circular_crackle],
            t0_add - t0_cur,
        )
        dvec_t_init_circ_crackle = (
            transforms.shift_taylor_params_circular_crackle_batch(
                param_set_batch[idx_circular_crackle],
                t0_init - t0_cur,
            )
        )
        dvec_t_add[idx_circular_crackle] = dvec_t_add_circ_crackle
        dvec_t_init[idx_circular_crackle] = dvec_t_init_circ_crackle

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.shift_taylor_params(
            param_set_batch[idx_taylor],
            t0_add - t0_cur,
        )
        dvec_t_init_norm = transforms.shift_taylor_params(
            param_set_batch[idx_taylor],
            t0_init - t0_cur,
        )
        dvec_t_add[idx_taylor] = dvec_t_add_norm
        dvec_t_init[idx_taylor] = dvec_t_init_norm

    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2] - dvec_t_init[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = (dvec_t_add[:, -1] - dvec_t_init[:, -1]) / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t0_add - t0_init,
        f0_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_batch, len(param_arr)), dtype=np.int64)
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        accel_new_batch,
    )
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        freq_new_batch,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def poly_taylor_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    leaves_batch_trans = np.zeros_like(leaves_batch)
    leaves_batch_trans[:, :-1] = transforms.shift_taylor_full(
        leaves_batch[:, :-1],
        delta_t,
        conservative_errors,
    )
    leaves_batch_trans[:, -1] = leaves_batch[:, -1]
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_taylor_transform_circular_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask(
        leaves_batch,
        snap_threshold=5,
    )
    leaves_batch_trans = leaves_batch.copy()
    if idx_circular_snap.size > 0:
        leaves_batch_trans[idx_circular_snap, :-1] = (
            transforms.shift_taylor_full_circular_batch(
                leaves_batch[idx_circular_snap, :-1],
                delta_t,
                conservative_errors,
            )
        )
    if idx_circular_crackle.size > 0:
        leaves_batch_trans[idx_circular_crackle, :-1] = (
            transforms.shift_taylor_full_circular_crackle_batch(
                leaves_batch[idx_circular_crackle, :-1],
                delta_t,
                conservative_errors,
            )
        )
    if idx_taylor.size > 0:
        leaves_batch_trans[idx_taylor, :-1] = transforms.shift_taylor_full(
            leaves_batch[idx_taylor, :-1],
            delta_t,
            conservative_errors,
        )
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def generate_bp_taylor_approx(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    isuggest: int = 0,
    use_conservative_errors: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the approximate branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nsegments) - ref_seg), kind="mergesort")
    coord_init = (ref_seg + 0.5) * tseg_ffa, tseg_ffa / 2
    leaf = poly_taylor_leaves(param_arr, dparams_lim, poly_order, coord_init)[isuggest]
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    for prune_level in range(1, nsegments):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tseg_ffa, scale * tseg_ffa
        scheme_till_now_prev = scheme_data[:prune_level]
        ref_prev = (np.min(scheme_till_now_prev) + np.max(scheme_till_now_prev) + 1) / 2
        scale_prev = ref_prev - np.min(scheme_till_now_prev)
        coord_prev = ref_prev * tseg_ffa, scale_prev * tseg_ffa
        coord_cur = coord_prev[0], coord_next[1]
        leaves_arr = poly_taylor_branch(
            leaf,
            coord_cur,
            fold_bins,
            tol_bins,
            poly_order,
            param_limits,
        )
        branching_pattern[prune_level - 1] = len(leaves_arr)
        leaves_arr_trans = poly_taylor_transform_batch(
            leaves_arr,
            coord_next,
            coord_cur,
            use_conservative_errors,
        )
        leaf = leaves_arr_trans[0]
    return np.array(branching_pattern)


@njit(cache=True, fastmath=True)
def generate_bp_taylor(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    use_conservative_errors: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nsegments) - ref_seg), kind="mergesort")
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    param_limits_d = np.empty((n_freqs, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL
    param_ranges = (param_limits_d[:, :, 1] - param_limits_d[:, :, 0]) / 2

    for prune_level in range(1, nsegments):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tseg_ffa, scale * tseg_ffa
        scheme_till_now_prev = scheme_data[:prune_level]
        ref_prev = (np.min(scheme_till_now_prev) + np.max(scheme_till_now_prev) + 1) / 2
        scale_prev = ref_prev - np.min(scheme_till_now_prev)
        coord_prev = ref_prev * tseg_ffa, scale_prev * tseg_ffa
        coord_cur = coord_prev[0], coord_next[1]
        _, t_obs_minus_t_ref = coord_cur
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            fold_bins,
            tol_bins,
            f0_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            fold_bins,
            f0_batch,
            t_ref=0,
        )

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > (param_ranges + eps)

        for i in range(n_freqs):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points
        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs

        # Transform dparams to the next segment
        delta_t = coord_next[0] - coord_cur[0]
        dparam_d_vec = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
        dparam_d_vec[:, :-1] = dparam_cur_next
        dparam_d_vec_new = transforms.shift_taylor_errors(
            dparam_d_vec,
            delta_t,
            use_conservative_errors,
        )
        dparam_cur_batch = dparam_d_vec_new[:, :-1]

    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_bp_taylor_circular(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    use_conservative_errors: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor circular pruning search."""
    poly_order = len(dparams_lim)
    if poly_order != 5:
        msg = "Circular branching pattern requires exactly 5 parameters."
        raise ValueError(msg)

    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nsegments) - ref_seg), kind="mergesort")
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    param_limits_d = np.empty((n_freqs, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL
    param_ranges = (param_limits_d[:, :, 1] - param_limits_d[:, :, 0]) / 2

    # Track when first snap branching occurs for each frequency
    snap_first_branched = np.zeros(n_freqs, dtype=np.bool_)
    for prune_level in range(1, nsegments):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tseg_ffa, scale * tseg_ffa
        scheme_till_now_prev = scheme_data[:prune_level]
        ref_prev = (np.min(scheme_till_now_prev) + np.max(scheme_till_now_prev) + 1) / 2
        scale_prev = ref_prev - np.min(scheme_till_now_prev)
        coord_prev = ref_prev * tseg_ffa, scale_prev * tseg_ffa
        coord_cur = coord_prev[0], coord_next[1]
        _, t_obs_minus_t_ref = coord_cur
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            fold_bins,
            tol_bins,
            f0_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            fold_bins,
            f0_batch,
            t_ref=0,
        )

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branch_accel = np.ones(n_freqs, dtype=np.int64)
        n_branch_snap = np.ones(n_freqs, dtype=np.int64)
        n_branches = np.ones(n_freqs, dtype=np.int64)
        validation_fractions = np.ones(n_freqs, dtype=np.float64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > (param_ranges + eps)

        for i in range(n_freqs):
            for j in range(1, poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points

                if j == 1:
                    n_branch_snap[i] = num_points
                if j == 3:
                    n_branch_accel[i] = num_points

            # Determine validation fraction
            snap_branches_now = n_branch_snap[i] > 1
            accel_branches_now = n_branch_accel[i] > 1
            if snap_branches_now and not snap_first_branched[i]:
                # First time snap branches - apply validation filtering
                # Since we start at snap=0, branching creates symmetric positive
                # negative values. Validation removes same-sign combinations with accel
                # If accel is also branching or non-zero, expect ~50% filtering
                if accel_branches_now or dparam_cur_batch[i, 2] != 0:
                    validation_fractions[i] = 0.5
                else:
                    # If accel is exactly 0, all combinations are valid
                    validation_fractions[i] = 1.0
                snap_first_branched[i] = True
            else:
                # Either snap doesn't branch this level, or it has branched before
                # If it branched before, symmetry is broken and no further filtering
                validation_fractions[i] = 1.0
            n_branches[i] *= validation_fractions[i]

        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs

        # Transform dparams to the next segment
        delta_t = coord_next[0] - coord_cur[0]
        dparam_d_vec = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
        dparam_d_vec[:, :-1] = dparam_cur_next
        dparam_d_vec_new = transforms.shift_taylor_errors(
            dparam_d_vec,
            delta_t,
            use_conservative_errors,
        )
        dparam_cur_batch = dparam_d_vec_new[:, :-1]
    return branching_pattern
