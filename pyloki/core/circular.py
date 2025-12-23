from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def get_circular_mask_taylor(
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
    dcrackle = leaves_batch[:, 0, 1]
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    jerk = leaves_batch[:, 2, 0]
    accel = leaves_batch[:, 3, 0]

    # Delays circular classification until snap is well-measured.
    is_sig_snap = np.abs(snap / (dsnap + eps)) > snap_threshold
    is_sig_crackle = np.abs(crackle / (dcrackle + eps)) > snap_threshold

    # Snap-Dominated Region (Standard)
    # We check if implied Omega is physical (-d4/d2 > 0)
    omega_sq_snap = -snap / (accel + eps)
    is_physical_snap = (omega_sq_snap > 0) & (np.abs(accel) > eps)
    mask_circular_snap = is_sig_snap & is_physical_snap

    # Crackle-Dominated Region (The Hole)
    # Condition: Snap is weak (in the null), but Crackle is strong.
    # Note: d2 and d4 vanish in the hole, so we rely on d3 and d5.
    in_the_hole = (~is_sig_snap) & is_sig_crackle

    # Check physics using recurrence: d5 = -Omega^2 * d3
    omega_sq_crackle = -crackle / (jerk + eps)
    is_physical_crackle = (omega_sq_crackle > 0) & (np.abs(jerk) > eps)
    mask_circular_crackle = in_the_hole & is_physical_crackle

    # Taylor Region (Noise / Unresolved)
    # Everything that isn't a confident circular candidate
    mask_taylor = ~(mask_circular_snap | mask_circular_crackle)

    idx_circular_snap = np.where(mask_circular_snap)[0]
    idx_circular_crackle = np.where(mask_circular_crackle)[0]
    idx_taylor = np.where(mask_taylor)[0]

    return idx_circular_snap, idx_circular_crackle, idx_taylor


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
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask_taylor(
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
    x_mass_const: float,
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
    crackle = leaves_batch[:, 0, 0]
    dcrackle = leaves_batch[:, 0, 1]
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    jerk = leaves_batch[:, 2, 0]
    accel = leaves_batch[:, 3, 0]

    omega_max_sq = (2 * np.pi / p_orb_min) ** 2

    # Classification (for gatekeeping)
    is_sig_snap = np.abs(snap / (dsnap + eps)) > snap_threshold
    is_sig_crackle = np.abs(crackle / (dcrackle + eps)) > snap_threshold

    # Keep noise (unrefined Taylor cells) for now
    is_noise = (~is_sig_snap) & (~is_sig_crackle)

    # Snap-Dominated Region (Standard)
    omega_sq_snap = -snap / (accel + eps)
    valid_omega_snap_sign = omega_sq_snap > 0
    safe_omega_sq_snap = np.where(valid_omega_snap_sign, omega_sq_snap, 0.0)
    omega_snap = np.sqrt(safe_omega_sq_snap)

    # Omega Validity
    valid_omega_snap = valid_omega_snap_sign & (safe_omega_sq_snap < omega_max_sq)

    # Physical Amplitude Limits
    # Is Accel (d2) physical? |d2| < x * w^2
    limit_accel = x_mass_const * (omega_snap ** (4 / 3) + eps)
    valid_accel = np.abs(accel) < limit_accel

    # Is Jerk (d3) physical? |d3| < x * w^3
    limit_jerk_snap = limit_accel * omega_snap
    valid_jerk_in_snap_region = np.abs(jerk) < limit_jerk_snap

    mask_snap_valid = (
        is_sig_snap & valid_omega_snap & valid_accel & valid_jerk_in_snap_region
    )

    # Crackle-Dominated Region (The Hole)
    omega_sq_crackle = -crackle / (jerk + eps)
    valid_omega_crackle_sign = omega_sq_crackle > 0
    safe_omega_sq_crackle = np.where(valid_omega_crackle_sign, omega_sq_crackle, 0.0)
    omega_crackle = np.sqrt(safe_omega_sq_crackle)

    # Omega Validity
    valid_omega_crackle = valid_omega_crackle_sign & (
        safe_omega_sq_crackle < omega_max_sq
    )
    # Physical Amplitude Limits
    # Is Accel (d2) small?
    limit_accel_in_hole = x_mass_const * (omega_crackle ** (4 / 3) + eps)
    valid_accel_in_hole = np.abs(accel) < limit_accel_in_hole

    # Is Jerk (d3) physical? (Primary check in hole)
    limit_jerk_hole = limit_accel_in_hole * omega_crackle
    valid_jerk_in_hole = np.abs(jerk) < limit_jerk_hole

    mask_crackle_valid = (
        (~is_sig_snap)
        & is_sig_crackle
        & valid_omega_crackle
        & valid_jerk_in_hole
        & valid_accel_in_hole
    )

    mask_keep = is_noise | mask_snap_valid | mask_crackle_valid
    idx = np.where(mask_keep)[0]
    return leaves_batch[idx], leaves_origins[idx]


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

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask_taylor(
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
def poly_taylor_transform_circular_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circular_mask_taylor(
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
def poly_taylor_fixed_resolve_circular_batch(
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
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    delta_t = t0_add - t0_init
    param_vec_batch = leaves_batch[:, :-2, 0]
    freq_cur_batch = leaves_batch[:, -3, 0]

    circ_mask = get_circular_mask_taylor(leaves_batch, snap_threshold=5)
    idx_circular = np.where(circ_mask)[0]
    idx_taylor = np.where(~circ_mask)[0]
    param_vec_new_batch = np.empty_like(param_vec_batch)
    delay_batch = np.empty(n_batch, dtype=param_vec_batch.dtype)

    if idx_circular.size > 0:
        param_vec_new_circ, delay_circ = (
            transforms.shift_taylor_params_circular_d_f_batch(
                param_vec_batch[idx_circular],
                delta_t,
            )
        )
        param_vec_new_batch[idx_circular] = param_vec_new_circ
        delay_batch[idx_circular] = delay_circ

    if idx_taylor.size > 0:
        param_vec_new_norm, delay_norm = transforms.shift_taylor_params_d_f_batch(
            param_vec_batch[idx_taylor],
            delta_t,
        )
        param_vec_new_batch[idx_taylor] = param_vec_new_norm
        delay_batch[idx_taylor] = delay_norm

    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        freq_cur_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_batch, len(param_arr)), dtype=np.int64)
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        param_vec_new_batch[:, -2],
    )
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        param_vec_new_batch[:, -1],
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def get_circular_mask_chebyshev(
    leaves_batch: np.ndarray,
    t_s: float,
    snap_threshold: float = 5,
) -> np.ndarray:
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
    alpha_4 = leaves_batch[:, 0, 0]
    dalpha_4 = leaves_batch[:, 0, 1]
    alpha_2 = leaves_batch[:, 2, 0]
    # Calculate accel and snap
    accel = (4 / t_s**2) * (alpha_2 - 4 * alpha_4)
    snap = (192 / t_s**4) * alpha_4
    dsnap = (192 / t_s**4) * dalpha_4

    # Numerical Stability
    # The acceleration and snap must be significantly non-zero to define an orbit.
    is_stable = (np.abs(accel) > eps) & (np.abs(snap) > eps)

    # Physicality
    # For Omega^2 = -s/a to be positive, s and a must have opposite signs.
    # This is the most fundamental check for oscillatory motion.
    omega_sq = -snap / (accel + eps)
    is_physical = omega_sq > 0

    # Delays circular classification until snap is well-measured.
    is_significant = np.abs(snap / (dsnap + eps)) > snap_threshold
    return is_stable & is_physical & is_significant


@njit(cache=True, fastmath=True)
def poly_chebyshev_validate_circular_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
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
    coord_cur: tuple[float, float]
        The coordinates of the current segment (level cur).
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
    _, t_s = coord_cur
    alpha_4 = leaves_batch[:, 0, 0]
    dalpha_4 = leaves_batch[:, 0, 1]
    alpha_2 = leaves_batch[:, 2, 0]
    # Calculate accel and snap
    accel = (4 / t_s**2) * (alpha_2 - 4 * alpha_4)
    snap = (192 / t_s**4) * alpha_4
    dsnap = (192 / t_s**4) * dalpha_4
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
def poly_chebyshev_resolve_circular_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the leaf parameters to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_batch, _, _ = leaves_batch.shape
    t0_cur, scale_cur = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_set_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    circ_mask = get_circular_mask_chebyshev(leaves_batch, scale_cur, snap_threshold=5)
    idx_circular = np.where(circ_mask)[0]
    idx_taylor = np.where(~circ_mask)[0]
    dvec_t_add = np.empty((n_batch, 5), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 5), dtype=np.float64)

    if idx_circular.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        param_set_batch_circ = param_set_batch[idx_circular]
        taylor_param_vec_circ = transforms.cheby_to_taylor(
            param_set_batch_circ,
            scale_cur,
        )
        dvec_t_add_circ = transforms.shift_taylor_params_circular_batch(
            taylor_param_vec_circ,
            t0_add - t0_cur,
        )
        dvec_t_init_circ = transforms.shift_taylor_params_circular_batch(
            taylor_param_vec_circ,
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular] = dvec_t_add_circ
        dvec_t_init[idx_circular] = dvec_t_init_circ

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.cheby_to_taylor_param_shift(
            param_set_batch[idx_taylor],
            t0_cur,
            scale_cur,
            t0_add,
        )
        dvec_t_init_norm = transforms.cheby_to_taylor_param_shift(
            param_set_batch[idx_taylor],
            t0_cur,
            scale_cur,
            t0_init,
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
def poly_chebyshev_transform_circular_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    _, scale_cur = coord_cur
    _, scale_next = coord_next
    circ_mask = get_circular_mask_chebyshev(leaves_batch, scale_cur, snap_threshold=5)
    idx_circular = np.where(circ_mask)[0]
    idx_taylor = np.where(~circ_mask)[0]
    leaves_batch_trans = leaves_batch.copy()
    if idx_circular.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        leaves_batch_circ = leaves_batch[idx_circular, :-1]
        leaves_batch_circ_taylor = transforms.cheby_to_taylor_full(
            leaves_batch_circ,
            scale_cur,
        )
        # Shift the taylor parameters in circular orbit
        dvec_t_add_circ = transforms.shift_taylor_full_circular_batch(
            leaves_batch_circ_taylor,
            coord_next[0] - coord_cur[0],
            conservative_errors,
        )
        # Convert the taylor parameters back to chebyshev parameters
        leaves_batch_trans[idx_circular, :-1] = transforms.taylor_to_cheby_full(
            dvec_t_add_circ,
            scale_next,
        )
    if idx_taylor.size > 0:
        leaves_batch_trans[idx_taylor, :-1] = transforms.shift_cheby_full(
            leaves_batch[idx_taylor, :-1],
            coord_next,
            coord_cur,
            conservative_errors,
        )
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_circular_resolve_batch(
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

    # Extract parameters leaf_batch[:, :-2]
    omega_orb_batch = leaf_batch[:, 0, 0]
    x_cos_nu_batch = leaf_batch[:, 1, 0]
    x_sin_nu_batch = leaf_batch[:, 2, 0]
    f_cur_batch = leaf_batch[:, 3, 0]

    # x is already in light seconds (no division by C_VAL)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb_batch * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    x_cos_nu_new_batch = x_cos_nu_batch * cos_odt - x_sin_nu_batch * sin_odt
    x_sin_nu_new_batch = x_sin_nu_batch * cos_odt + x_cos_nu_batch * sin_odt
    a_new_batch = -C_VAL * omega_orb_batch**2 * x_sin_nu_new_batch
    v_new_batch = omega_orb_batch * x_cos_nu_new_batch  # C_VAL division included
    delay_batch = x_sin_nu_new_batch - x_sin_nu_batch  # C_VAL division included
    f_new_batch = f_cur_batch * (1 + v_new_batch)

    relative_phase_batch = psr_utils.get_phase_idx(
        delta_t,
        f_cur_batch,
        fold_bins,
        delay_batch,
    )
    param_idx_batch = np.zeros((n_leaves, nparams), dtype=np.int64)
    param_idx_batch[:, -1] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-1],
        f_new_batch,
    )
    param_idx_batch[:, -2] = np_utils.find_nearest_sorted_idx_vect(
        param_arr[-2],
        a_new_batch,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def poly_circular_branch_batch(
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
    # Only two parameters to branch: omega and frequency; x_cos_nu, x_sin_nu params
    # never get refined (their accuracy does not increase with time).
    nparams = 2
    _, scale_cur = coord_cur
    param_cur_batch = param_set_batch[:, :nparams, 0]
    dparam_cur_batch = param_set_batch[:, :nparams, 1]
    x_cos_nu_cur_batch = param_set_batch[:, 2, 0]
    dx_cos_nu_cur_batch = param_set_batch[:, 2, 1]
    x_sin_nu_cur_batch = param_set_batch[:, 3, 0]
    dx_sin_nu_cur_batch = param_set_batch[:, 3, 1]
    f0_batch = param_set_batch[:, -2, 0]
    t0_batch = param_set_batch[:, -1, 0]
    scale_batch = param_set_batch[:, -1, 1]
    x_cur_batch = np.sqrt(x_cos_nu_cur_batch**2 + x_sin_nu_cur_batch**2)

    tseg_cur = 2 * scale_cur
    dparam_opt_batch = np.empty((n_batch, nparams), dtype=np.float64)
    domega_opt_batch = psr_utils.poly_taylor_step_d_vec(
        nparams,
        tseg_cur,
        fold_bins,
        tol_bins,
        x_cur_batch,
    )
    dfreq_opt_batch = psr_utils.poly_taylor_step_f(
        1,
        tseg_cur,
        fold_bins,
        tol_bins,
        t_ref=tseg_cur / 2,
    )
    dparam_opt_batch[:, 0] = domega_opt_batch
    dparam_opt_batch[:, 1] = dfreq_opt_batch

    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_opt_batch,
        tseg_cur,
        fold_bins,
        x_cur_batch,
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
    batch_leaves_circular, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        nparams,
    )
    total_leaves = len(batch_origins)
    batch_leaves = np.zeros((total_leaves, poly_order + 2, 2), dtype=np.float64)
    batch_leaves[:, :-2, 0] = batch_leaves_circular
    batch_leaves[:, :-2, 1] = pad_branched_dparams[batch_origins]
    batch_leaves[:, 2, 0] = x_cos_nu_cur_batch[batch_origins]
    batch_leaves[:, 2, 1] = dx_cos_nu_cur_batch[batch_origins]
    batch_leaves[:, 3, 0] = x_sin_nu_cur_batch[batch_origins]
    batch_leaves[:, 3, 1] = dx_sin_nu_cur_batch[batch_origins]
    batch_leaves[:, -2, 0] = f0_batch[batch_origins]
    batch_leaves[:, -1, 0] = t0_batch[batch_origins]
    batch_leaves[:, -1, 1] = scale_batch[batch_origins]
    return batch_leaves, batch_origins


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


@njit(cache=True, fastmath=True)
def generate_bp_taylor_fixed_circular(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor fixed circular pruning."""
    poly_order = len(dparams_lim)
    if poly_order != 4:
        msg = "Circular branching pattern requires exactly 4 parameters."
        raise ValueError(msg)

    freq_arr = param_arr[-1]
    n0 = len(freq_arr)
    param_ranges = np.array([(p_max - p_min) / 2 for p_min, p_max in param_limits])

    # Track when first snap branching occurs for each frequency
    snap_first_branched = np.zeros(n0, dtype=np.bool_)
    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nsegments) - ref_seg), kind="mergesort")

    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams_lim

    weights = np.ones(n0, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    t0_init = (ref_seg + 0.5) * tseg_ffa
    scale_init = tseg_ffa / 2

    for prune_level in range(1, nsegments):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        t0_next = float(ref * tseg_ffa)
        scale_next = float(scale * tseg_ffa)
        left_edge = t0_next - scale_next
        right_edge = t0_next + scale_next
        scale_cur = max(abs(left_edge - t0_init), abs(right_edge - t0_init))
        t_obs_minus_t_ref = scale_cur - scale_init
        dparam_new_batch = psr_utils.poly_taylor_step_d_f_vec(
            poly_order,
            t_obs_minus_t_ref,
            fold_bins,
            tol_bins,
            freq_arr,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_f_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            fold_bins,
            freq_arr,
            t_ref=0,
        )

        nfreq = len(freq_arr)
        dparam_next_tmp = np.empty((nfreq, poly_order), dtype=np.float64)
        n_branch_freq = np.ones(nfreq, dtype=np.int64)
        n_branch_accel = np.ones(nfreq, dtype=np.int64)
        n_branch_jerk = np.ones(nfreq, dtype=np.int64)
        n_branch_snap = np.ones(nfreq, dtype=np.int64)
        validation_fractions = np.ones(nfreq, dtype=np.float64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > (param_ranges + eps)

        weighted_sum = 0.0
        total_weight = 0.0
        total_freq_branches = 0

        # First pass: determine branching counts, update dparams, compute stats
        for i in range(nfreq):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

                if j == 0:
                    n_branch_snap[i] = num_points
                elif j == 1:
                    n_branch_jerk[i] = num_points
                elif j == 2:
                    n_branch_accel[i] = num_points
                else:
                    n_branch_freq[i] = num_points

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
            # Calculate total combinations for this frequency
            raw_combinations = (
                float(n_branch_snap[i])
                * float(n_branch_jerk[i])
                * float(n_branch_accel[i])
                * float(n_branch_freq[i])
            )
            valid_combinations = validation_fractions[i] * raw_combinations
            total_weight += weights[i]
            weighted_sum += weights[i] * valid_combinations
            total_freq_branches += n_branch_freq[i]

        # Compute average branching factor
        branching_pattern[prune_level - 1] = weighted_sum / total_weight
        freq_arr_next = np.empty(total_freq_branches, dtype=np.float64)
        weights_next = np.empty(total_freq_branches, dtype=np.float64)
        dparam_cur_next = np.empty((total_freq_branches, poly_order), dtype=np.float64)
        snap_first_branched_next = np.empty(total_freq_branches, dtype=np.bool_)

        pos = 0
        for i in range(nfreq):
            cfreq = n_branch_freq[i]
            # Weight includes validation effects
            adjusted_weight = (
                weights[i]
                * validation_fractions[i]
                * (
                    float(n_branch_snap[i])
                    * float(n_branch_jerk[i])
                    * float(n_branch_accel[i])
                )
            )

            if cfreq == 1:
                freq_arr_next[pos] = freq_arr[i]
                weights_next[pos] = adjusted_weight
                dparam_cur_next[pos] = dparam_next_tmp[i]
                snap_first_branched_next[pos] = snap_first_branched[i]
                pos += 1
            else:
                dparam_cur_freq = dparam_cur_batch[i, poly_order - 1]
                delta = 0.25 * dparam_cur_freq
                f = freq_arr[i]
                # Create cfreq evenly spaced frequency points centered around f
                for k in range(cfreq):
                    offset = (k - (cfreq - 1) / 2) * delta
                    freq_arr_next[pos] = f + offset
                    weights_next[pos] = adjusted_weight
                    dparam_cur_next[pos] = dparam_next_tmp[i]
                    snap_first_branched_next[pos] = snap_first_branched[i]
                    pos += 1
        # Advance to next stage
        freq_arr = freq_arr_next
        dparam_cur_batch = dparam_cur_next
        weights = weights_next
        snap_first_branched = snap_first_branched_next
    return branching_pattern
