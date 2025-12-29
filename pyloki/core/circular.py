from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.snail import MiddleOutScheme


@njit(cache=True, fastmath=True)
def get_circ_mask(
    crackle: np.ndarray,
    dcrackle: np.ndarray,
    snap: np.ndarray,
    dsnap: np.ndarray,
    jerk: np.ndarray,
    accel: np.ndarray,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12

    # Delays circular classification until snap is well-measured.
    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + eps))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + eps))

    # Snap-Dominated Region (Standard)
    # We check if implied Omega is physical (-d4/d2 > 0)
    is_physical_snap = ((-snap * accel) > 0) & (np.abs(accel) > eps)
    mask_circular_snap = is_sig_snap & is_physical_snap

    # Crackle-Dominated Region (The Hole)
    # Condition: Snap is weak (in the null), but Crackle is strong.
    # Note: d2 and d4 vanish in the hole, so we rely on d3 and d5.
    in_the_hole = (~is_sig_snap) & is_sig_crackle

    # Check if implied Omega is physical (-d5/d3 > 0)
    is_physical_crackle = ((-crackle * jerk) > 0) & (np.abs(jerk) > eps)
    mask_circular_crackle = in_the_hole & is_physical_crackle

    # Taylor Region (Noise / Unresolved)
    # Everything that isn't a confident circular candidate
    mask_taylor = ~(mask_circular_snap | mask_circular_crackle)

    idx_circular_snap = np.where(mask_circular_snap)[0]
    idx_circular_crackle = np.where(mask_circular_crackle)[0]
    idx_taylor = np.where(mask_taylor)[0]

    return idx_circular_snap, idx_circular_crackle, idx_taylor


@njit(cache=True, fastmath=True)
def circ_validate_batch(
    crackle: np.ndarray,
    dcrackle: np.ndarray,
    snap: np.ndarray,
    dsnap: np.ndarray,
    jerk: np.ndarray,
    accel: np.ndarray,
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float = 5,
) -> np.ndarray:
    eps = 1e-12

    n_leaves = len(crackle)
    mask_keep = np.zeros(n_leaves, dtype=np.bool_)
    omega_max_sq = (2 * np.pi / p_orb_min) ** 2

    # Classification (for gatekeeping)
    # |val / step| > thresh  <=>  |val| > thresh * |step|
    # dsnap and dcrackle are step sizes (always positive)
    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + eps))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + eps))

    # 1. Noise (Unresolved Taylor cells) - Always keep
    is_noise = (~is_sig_snap) & (~is_sig_crackle)
    mask_keep |= is_noise

    # 2. Snap-Dominated Region
    idx_snap = np.where(is_sig_snap)[0]
    if len(idx_snap) > 0:
        s_snap = snap[idx_snap]
        s_accel = accel[idx_snap]
        s_jerk = jerk[idx_snap]
        s_omega_sq = -s_snap / (s_accel + eps)
        # Check: Physical Sign (-d4/d2 > 0)
        valid_sign = (s_omega_sq > 0) & (np.abs(s_accel) > eps)
        # Check: Max Orbital Frequency
        valid_omega = s_omega_sq < omega_max_sq

        # |d2| < x * omega^(4/3)
        s_omega_sq_safe = np.abs(s_omega_sq)
        limit_accel = x_mass_const * (s_omega_sq_safe ** (2 / 3) + eps)
        valid_accel = np.abs(s_accel) < limit_accel

        # |d3| < |d2| * omega
        valid_jerk = (s_jerk**2) < (s_accel**2 * s_omega_sq_safe)

        mask_snap_local = valid_sign & valid_omega & valid_accel & valid_jerk
        mask_keep[idx_snap] |= mask_snap_local

    # 3. Crackle-Dominated Region (The Hole)
    # Only if snap is NOT significant but crackle is
    is_hole = (~is_sig_snap) & is_sig_crackle
    idx_hole = np.where(is_hole)[0]
    if len(idx_hole) > 0:
        h_crackle = crackle[idx_hole]
        h_jerk = jerk[idx_hole]
        h_accel = accel[idx_hole]
        h_omega_sq = -h_crackle / (h_jerk + eps)
        valid_sign = (h_omega_sq > 0) & (np.abs(h_jerk) > eps)
        valid_omega = h_omega_sq < omega_max_sq

        # |d2| < x * omega^(4/3)
        h_omega_sq_safe = np.abs(h_omega_sq)
        limit_accel = x_mass_const * (h_omega_sq_safe ** (2 / 3) + eps)
        valid_accel = np.abs(h_accel) < limit_accel

        # |d3| < |d2| * omega
        valid_jerk = (h_jerk**2) < (limit_accel**2 * h_omega_sq_safe)

        mask_hole_local = valid_sign & valid_omega & valid_accel & valid_jerk
        mask_keep[idx_hole] |= mask_hole_local

    return np.where(mask_keep)[0]


@njit(cache=True, fastmath=True)
def get_circ_taylor_mask(
    leaves_batch: np.ndarray,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a robust mask to identify circular orbit candidates.

    Filters out physically implausible and numerically unstable orbits.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Shape (n_batch, nparams + 2, 2)
    minimum_snap_cells: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a high-quality circular orbit candidate.

    """
    crackle = leaves_batch[:, 0, 0]
    dcrackle = leaves_batch[:, 0, 1]
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    jerk = leaves_batch[:, 2, 0]
    accel = leaves_batch[:, 3, 0]

    return get_circ_mask(
        crackle,
        dcrackle,
        snap,
        dsnap,
        jerk,
        accel,
        minimum_snap_cells=minimum_snap_cells,
    )


@njit(cache=True, fastmath=True)
def get_circ_taylor_mask_branch(
    leaves_params_batch: np.ndarray,
    leaves_dparams_batch: np.ndarray,
    batch_needs_crackle: np.ndarray,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of the leaves that need to be expanded (crackle)."""
    eps = 1e-12
    crackle = leaves_params_batch[:, 0]
    dcrackle = leaves_dparams_batch[:, 0]
    snap = leaves_params_batch[:, 1]
    dsnap = leaves_dparams_batch[:, 1]
    jerk = leaves_params_batch[:, 2]

    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + eps))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + eps))
    in_the_hole = (~is_sig_snap) & is_sig_crackle
    is_physical_crackle = ((-crackle * jerk) > 0) & (np.abs(jerk) > eps)
    mask_circular_crackle = in_the_hole & is_physical_crackle

    # Optimization: Filter out 'crackle' candidates that don't actually need branching
    # True crackle candidates: in the hole AND step size is large enough
    mask_expand_crackle = mask_circular_crackle & batch_needs_crackle
    idx_expand_crackle = np.where(mask_expand_crackle)[0]
    # Keep indices: Taylor + Snap + Crackle-that-doesnt-need-branching
    mask_keep = ~mask_expand_crackle
    idx_keep = np.where(mask_keep)[0]

    return idx_expand_crackle, idx_keep


@njit(cache=True, fastmath=True)
def circ_taylor_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch, _, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur
    param_cur_batch = leaves_batch[:, :-2, 0]
    dparam_cur_batch = leaves_batch[:, :-2, 1]
    d0_cur_batch = leaves_batch[:, -2, 0]
    f0_batch = leaves_batch[:, -1, 0]
    basis_flag_batch = leaves_batch[:, -1, 1]

    param_limits_d = np.empty((n_batch, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL

    dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
        poly_order,
        t_obs_minus_t_ref,
        nbins,
        eta,
        f0_batch,
        t_ref=0,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_new_batch,
        t_obs_minus_t_ref,
        nbins,
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
    needs_branching = shift_bins_batch >= (eta - eps)
    for i in range(n_batch):
        for j in range(poly_order):
            if not needs_branching[i, j] or j == 0:
                # crackle - don't branch yet, keep at current value
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- First Cartesian Product (All Except Crackle) ---
    # Note: These 'leaves' contain unbranched crackle (parent values)
    leaves_branch_taylor_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        poly_order,
    )

    # Check if the batch item originating this leaf needs crackle branching
    leaves_branched_dparams = pad_branched_dparams[batch_origins]
    batch_needs_crackle = needs_branching[batch_origins, 0]
    idx_expand_crackle, idx_keep = get_circ_taylor_mask_branch(
        leaves_branch_taylor_batch,
        leaves_branched_dparams,
        batch_needs_crackle,
        minimum_snap_cells=minimum_snap_cells,
    )
    n_keep = len(idx_keep)
    n_crackle_expand = len(idx_expand_crackle)

    # If no expansion needed, return early (fast path)
    if n_crackle_expand == 0:
        total_leaves = n_keep
        leaves_final = np.empty((total_leaves, poly_order + 2, 2), dtype=np.float64)
        leaves_final[:, :-2, 0] = leaves_branch_taylor_batch
        leaves_final[:, :-2, 1] = leaves_branched_dparams
        leaves_final[:, -2, 0] = d0_cur_batch[batch_origins]
        leaves_final[:, -1, 0] = f0_batch[batch_origins]
        leaves_final[:, -1, 1] = basis_flag_batch[batch_origins]
        return leaves_final, batch_origins

    # Branch crackle parameter for these specific leaves
    crackle_branched_params = np.empty((n_crackle_expand, branch_max), dtype=np.float64)
    crackle_branched_dparams = np.empty(n_crackle_expand, dtype=np.float64)
    crackle_branched_counts = np.empty(n_crackle_expand, dtype=np.int64)
    crackle_origins = batch_origins[idx_expand_crackle]

    for i in range(n_crackle_expand):
        orig_batch_idx = crackle_origins[i]
        # We know needs_branching is True here, so we always branch
        param_min, param_max = param_limits_d[orig_batch_idx, 0]  # crackle limits
        dparam_act, count = psr_utils.branch_param_padded(
            crackle_branched_params[i],
            leaves_branch_taylor_batch[idx_expand_crackle[i], 0],
            leaves_branched_dparams[idx_expand_crackle[i], 0],
            dparam_new_batch[orig_batch_idx, 0],
            param_min,
            param_max,
        )
        crackle_branched_dparams[i] = dparam_act
        crackle_branched_counts[i] = count

    # Construct Final Array
    total_crackle_branches = np.sum(crackle_branched_counts)
    total_leaves = n_keep + total_crackle_branches
    leaves_final = np.empty((total_leaves, poly_order + 2, 2), dtype=np.float64)
    origins_final = np.empty(total_leaves, dtype=np.int64)

    if n_keep > 0:
        leaves_final[:n_keep, :-2, 0] = leaves_branch_taylor_batch[idx_keep]
        leaves_final[:n_keep, :-2, 1] = leaves_branched_dparams[idx_keep]
        origins_keep = batch_origins[idx_keep]
        leaves_final[:n_keep, -2, 0] = d0_cur_batch[origins_keep]
        leaves_final[:n_keep, -1, 0] = f0_batch[origins_keep]
        origins_final[:n_keep] = origins_keep

    current_idx = n_keep
    for i in range(n_crackle_expand):
        count_i = crackle_branched_counts[i]
        end_idx = current_idx + count_i
        src_idx = idx_expand_crackle[i]
        src_origin = crackle_origins[i]

        leaves_final[current_idx:end_idx, :-2, 0] = leaves_branch_taylor_batch[src_idx]
        leaves_final[current_idx:end_idx, :-2, 1] = leaves_branched_dparams[src_idx]
        leaves_final[current_idx:end_idx, 0, 0] = crackle_branched_params[i, :count_i]
        leaves_final[current_idx:end_idx, 0, 1] = crackle_branched_dparams[i]
        leaves_final[current_idx:end_idx, -2, 0] = d0_cur_batch[src_origin]
        leaves_final[current_idx:end_idx, -1, 0] = f0_batch[src_origin]
        leaves_final[current_idx:end_idx, -1, 1] = basis_flag_batch[src_origin]
        origins_final[current_idx:end_idx] = src_origin
        current_idx = end_idx

    return leaves_final, origins_final


@njit(cache=True, fastmath=True)
def circ_taylor_validate_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float = 5,
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
    minimum_snap_cells: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered leaf parameter sets (only physically plausible) and their origins.
    """
    crackle = leaves_batch[:, 0, 0]
    dcrackle = leaves_batch[:, 0, 1]
    snap = leaves_batch[:, 1, 0]
    dsnap = leaves_batch[:, 1, 1]
    jerk = leaves_batch[:, 2, 0]
    accel = leaves_batch[:, 3, 0]

    idx = circ_validate_batch(
        crackle,
        dcrackle,
        snap,
        dsnap,
        jerk,
        accel,
        p_orb_min,
        x_mass_const,
        minimum_snap_cells=minimum_snap_cells,
    )
    return leaves_batch[idx], leaves_origins[idx]


@njit(cache=True, fastmath=True)
def circ_taylor_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_batch, _, _ = leaves_batch.shape
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_taylor_mask(
        leaves_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 6), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 6), dtype=np.float64)

    if idx_circular_snap.size > 0:
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circular_snap],
            t0_add - t0_cur,
        )
        dvec_t_init_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circular_snap],
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular_snap] = dvec_t_add_circ_snap
        dvec_t_init[idx_circular_snap] = dvec_t_init_circ_snap

    if idx_circular_crackle.size > 0:
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            param_vec_batch[idx_circular_crackle],
            t0_add - t0_cur,
        )
        dvec_t_init_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            param_vec_batch[idx_circular_crackle],
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular_crackle] = dvec_t_add_circ_crackle
        dvec_t_init[idx_circular_crackle] = dvec_t_init_circ_crackle

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.shift_taylor_params(
            param_vec_batch[idx_taylor],
            t0_add - t0_cur,
        )
        dvec_t_init_norm = transforms.shift_taylor_params(
            param_vec_batch[idx_taylor],
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
        nbins,
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
def circ_taylor_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_taylor_mask(
        leaves_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 6), dtype=np.float64)
    if idx_circular_snap.size > 0:
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circular_snap],
            t0_add - t0_init,
        )
        dvec_t_add[idx_circular_snap] = dvec_t_add_circ_snap

    if idx_circular_crackle.size > 0:
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            param_vec_batch[idx_circular_crackle],
            t0_add - t0_init,
        )
        dvec_t_add[idx_circular_crackle] = dvec_t_add_circ_crackle

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.shift_taylor_params(
            param_vec_batch[idx_taylor],
            t0_add - t0_init,
        )
        dvec_t_add[idx_taylor] = dvec_t_add_norm

    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = dvec_t_add[:, -1] / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t0_add - t0_init,
        f0_batch,
        nbins,
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
def circ_taylor_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
    minimum_snap_cells: float,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_taylor_mask(
        leaves_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    leaves_batch_trans = leaves_batch.copy()
    if idx_circular_snap.size > 0:
        leaves_batch_trans[idx_circular_snap, :-1] = (
            transforms.shift_taylor_circular_full(
                leaves_batch[idx_circular_snap, :-1],
                delta_t,
                use_conservative_tile,
            )
        )
    if idx_circular_crackle.size > 0:
        leaves_batch_trans[idx_circular_crackle, :-1] = (
            transforms.shift_taylor_circular_crackle_full(
                leaves_batch[idx_circular_crackle, :-1],
                delta_t,
                use_conservative_tile,
            )
        )
    if idx_taylor.size > 0:
        leaves_batch_trans[idx_taylor, :-1] = transforms.shift_taylor_full(
            leaves_batch[idx_taylor, :-1],
            delta_t,
            use_conservative_tile,
        )
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def get_circ_chebyshev_mask(
    leaves_batch: np.ndarray,
    t_s: float,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a robust mask to identify circular orbit candidates.

    Filters out physically implausible and numerically unstable orbits.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Shape (n_batch, nparams + 2, 2)
    minimum_snap_cells: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a high-quality circular orbit candidate.

    """
    alpha_5 = leaves_batch[:, 0, 0]
    dalpha_5 = leaves_batch[:, 0, 1]
    alpha_4 = leaves_batch[:, 1, 0]
    dalpha_4 = leaves_batch[:, 1, 1]
    alpha_3 = leaves_batch[:, 2, 0]
    alpha_2 = leaves_batch[:, 3, 0]
    # Calculate accel and snap
    crackle = (1920 / t_s**5) * alpha_5
    dcrackle = (1920 / t_s**5) * dalpha_5
    snap = (192 / t_s**4) * alpha_4
    dsnap = (192 / t_s**4) * dalpha_4
    jerk = (24 / t_s**3) * (alpha_3 - 5 * alpha_5)
    accel = (4 / t_s**2) * (alpha_2 - 4 * alpha_4)

    return get_circ_mask(
        crackle,
        dcrackle,
        snap,
        dsnap,
        jerk,
        accel,
        minimum_snap_cells=minimum_snap_cells,
    )


@njit(cache=True, fastmath=True)
def circ_chebyshev_validate_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float = 5,
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
    minimum_snap_cells: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered leaf parameter sets (only physically plausible) and their origins.
    """
    _, t_s = coord_cur
    alpha_5 = leaves_batch[:, 0, 0]
    dalpha_5 = leaves_batch[:, 0, 1]
    alpha_4 = leaves_batch[:, 1, 0]
    dalpha_4 = leaves_batch[:, 1, 1]
    alpha_3 = leaves_batch[:, 2, 0]
    alpha_2 = leaves_batch[:, 3, 0]
    # Calculate accel and snap
    crackle = (1920 / t_s**5) * alpha_5
    dcrackle = (1920 / t_s**5) * dalpha_5
    snap = (192 / t_s**4) * alpha_4
    dsnap = (192 / t_s**4) * dalpha_4
    jerk = (24 / t_s**3) * (alpha_3 - 5 * alpha_5)
    accel = (4 / t_s**2) * (alpha_2 - 4 * alpha_4)

    idx = circ_validate_batch(
        crackle,
        dcrackle,
        snap,
        dsnap,
        jerk,
        accel,
        p_orb_min,
        x_mass_const,
        minimum_snap_cells=minimum_snap_cells,
    )
    return leaves_batch[idx], leaves_origins[idx]


@njit(cache=True, fastmath=True)
def circ_chebyshev_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the leaf parameters to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_batch, _, _ = leaves_batch.shape
    t0_cur, scale_cur = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_chebyshev_mask(
        leaves_batch,
        scale_cur,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 5), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 5), dtype=np.float64)

    if idx_circular_snap.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        param_vec_batch_circ = param_vec_batch[idx_circular_snap]
        taylor_param_vec_circ = transforms.cheby_to_taylor(
            param_vec_batch_circ,
            scale_cur,
        )
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            taylor_param_vec_circ,
            t0_add - t0_cur,
        )
        dvec_t_init_circ_snap = transforms.shift_taylor_circular_params(
            taylor_param_vec_circ,
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular_snap] = dvec_t_add_circ_snap
        dvec_t_init[idx_circular_snap] = dvec_t_init_circ_snap

    if idx_circular_crackle.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        param_vec_batch_circ = param_vec_batch[idx_circular_crackle]
        taylor_param_vec_circ = transforms.cheby_to_taylor(
            param_vec_batch_circ,
            scale_cur,
        )
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            taylor_param_vec_circ,
            t0_add - t0_cur,
        )
        dvec_t_init_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            taylor_param_vec_circ,
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circular_crackle] = dvec_t_add_circ_crackle
        dvec_t_init[idx_circular_crackle] = dvec_t_init_circ_crackle

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.cheby_to_taylor_param_shift(
            param_vec_batch[idx_taylor],
            t0_cur,
            scale_cur,
            t0_add,
        )
        dvec_t_init_norm = transforms.cheby_to_taylor_param_shift(
            param_vec_batch[idx_taylor],
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
        nbins,
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
def circ_chebyshev_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur_fixed: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the leaf parameters to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 4
    n_batch, _, _ = leaves_batch.shape
    _, scale_cur_fixed = coord_cur_fixed
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_chebyshev_mask(
        leaves_batch,
        scale_cur_fixed,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 5), dtype=np.float64)

    if idx_circular_snap.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        param_vec_batch_circ = param_vec_batch[idx_circular_snap]
        taylor_param_vec_circ = transforms.cheby_to_taylor(
            param_vec_batch_circ,
            scale_cur_fixed,
        )
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            taylor_param_vec_circ,
            t0_add - t0_init,
        )
        dvec_t_add[idx_circular_snap] = dvec_t_add_circ_snap

    if idx_circular_crackle.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        param_vec_batch_circ = param_vec_batch[idx_circular_crackle]
        taylor_param_vec_circ = transforms.cheby_to_taylor(
            param_vec_batch_circ,
            scale_cur_fixed,
        )
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_crackle_params(
            taylor_param_vec_circ,
            t0_add - t0_init,
        )
        dvec_t_add[idx_circular_crackle] = dvec_t_add_circ_crackle

    if idx_taylor.size > 0:
        dvec_t_add_norm = transforms.cheby_to_taylor_param_shift(
            param_vec_batch[idx_taylor],
            t0_init,
            scale_cur_fixed,
            t0_add,
        )
        dvec_t_add[idx_taylor] = dvec_t_add_norm

    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = dvec_t_add[:, -1] / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t0_add - t0_init,
        f0_batch,
        nbins,
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
def circ_chebyshev_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
    minimum_snap_cells: float,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    _, scale_cur = coord_cur
    _, scale_next = coord_next
    idx_circular_snap, idx_circular_crackle, idx_taylor = get_circ_chebyshev_mask(
        leaves_batch,
        scale_cur,
        minimum_snap_cells=minimum_snap_cells,
    )
    leaves_batch_trans = leaves_batch.copy()
    if idx_circular_snap.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        leaves_batch_circ = leaves_batch[idx_circular_snap, :-1]
        leaves_batch_circ_taylor = transforms.cheby_to_taylor_full(
            leaves_batch_circ,
            scale_cur,
        )
        # Shift the taylor parameters in circular orbit
        dvec_t_add_circ = transforms.shift_taylor_circular_full(
            leaves_batch_circ_taylor,
            coord_next[0] - coord_cur[0],
            use_conservative_tile,
        )
        # Convert the taylor parameters back to chebyshev parameters
        leaves_batch_trans[idx_circular_snap, :-1] = transforms.taylor_to_cheby_full(
            dvec_t_add_circ,
            scale_next,
        )
    if idx_circular_crackle.size > 0:
        # Convert the chebyshev parameters to taylor parameters
        leaves_batch_circ = leaves_batch[idx_circular_crackle, :-1]
        leaves_batch_circ_taylor = transforms.cheby_to_taylor_full(
            leaves_batch_circ,
            scale_cur,
        )
        # Shift the taylor parameters in circular orbit
        dvec_t_add_circ = transforms.shift_taylor_circular_crackle_full(
            leaves_batch_circ_taylor,
            coord_next[0] - coord_cur[0],
            use_conservative_tile,
        )
        # Convert the taylor parameters back to chebyshev parameters
        leaves_batch_trans[idx_circular_crackle, :-1] = transforms.taylor_to_cheby_full(
            dvec_t_add_circ,
            scale_next,
        )
    if idx_taylor.size > 0:
        leaves_batch_trans[idx_taylor, :-1] = transforms.shift_cheby_full(
            leaves_batch[idx_taylor, :-1],
            coord_next,
            coord_cur,
            use_conservative_tile,
        )
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_circular_resolve_batch(
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
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
        nbins,
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
    param_vec_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_vec_batch)
    # Only two parameters to branch: omega and frequency; x_cos_nu, x_sin_nu params
    # never get refined (their accuracy does not increase with time).
    nparams = 2
    _, scale_cur = coord_cur
    param_cur_batch = param_vec_batch[:, :nparams, 0]
    dparam_cur_batch = param_vec_batch[:, :nparams, 1]
    x_cos_nu_cur_batch = param_vec_batch[:, 2, 0]
    dx_cos_nu_cur_batch = param_vec_batch[:, 2, 1]
    x_sin_nu_cur_batch = param_vec_batch[:, 3, 0]
    dx_sin_nu_cur_batch = param_vec_batch[:, 3, 1]
    f0_batch = param_vec_batch[:, -2, 0]
    t0_batch = param_vec_batch[:, -1, 0]
    scale_batch = param_vec_batch[:, -1, 1]
    x_cur_batch = np.sqrt(x_cos_nu_cur_batch**2 + x_sin_nu_cur_batch**2)

    tseg_cur = 2 * scale_cur
    dparam_opt_batch = np.empty((n_batch, nparams), dtype=np.float64)
    domega_opt_batch = psr_utils.poly_taylor_step_d_vec(
        nparams,
        tseg_cur,
        nbins,
        eta,
        x_cur_batch,
    )
    dfreq_opt_batch = psr_utils.poly_taylor_step_f(
        1,
        tseg_cur,
        nbins,
        eta,
        t_ref=tseg_cur / 2,
    )
    dparam_opt_batch[:, 0] = domega_opt_batch
    dparam_opt_batch[:, 1] = dfreq_opt_batch

    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_opt_batch,
        tseg_cur,
        nbins,
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
    mask_2d = shift_bins_batch > eta  # Shape (n_batch, nparams)
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
def generate_bp_circ_taylor(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    p_orb_min: float,
    minimum_snap_cells: float = 5,
    use_conservative_tile: bool = False,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor circular pruning search."""
    poly_order = len(dparams_lim)
    if poly_order != 5:
        msg = "Circular branching pattern requires exactly 5 parameters."
        raise ValueError(msg)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa)
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
    eps = 1e-6
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(prune_level)
        _, t_obs_minus_t_ref = coord_cur
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f0_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            nbins,
            f0_batch,
            t_ref=0,
        )

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branch_snap = np.ones(n_freqs, dtype=np.int64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - eps)
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

        # Determine validation fraction
        snap_active = n_branch_snap[i] > minimum_snap_cells
        # Apply 0.5x if this is the first time snap becomes active
        just_active = snap_active & (~snap_first_branched)
        # Correction factor array
        val_factor = np.ones(n_freqs, dtype=np.float64)
        val_factor[just_active] = 0.5
        n_branches *= val_factor
        snap_first_branched |= just_active
        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs

        # Transform dparams to the next segment
        delta_t = coord_next[0] - coord_cur[0]
        dparam_d_vec = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
        dparam_d_vec[:, :-1] = dparam_cur_next
        dparam_d_vec_new = transforms.shift_taylor_circular_errors(
            dparam_d_vec,
            delta_t,
            p_orb_min,
            use_conservative_tile,
        )
        dparam_cur_batch = dparam_d_vec_new[:, :-1]
    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_bp_circ_taylor_fixed(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    minimum_snap_cells: float = 5,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor fixed circular pruning."""
    poly_order = len(dparams_lim)
    if poly_order != 5:
        msg = "Circular branching pattern requires exactly 5 parameters."
        raise ValueError(msg)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa)
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
    eps = 1e-6
    for prune_level in range(1, nsegments):
        coord_cur_fixed = snail_scheme.get_current_coord_fixed(prune_level)
        _, t_obs_minus_t_ref = coord_cur_fixed
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f0_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            nbins,
            f0_batch,
            t_ref=0,
        )

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branch_snap = np.ones(n_freqs, dtype=np.int64)
        n_branches = np.ones(n_freqs, dtype=np.int64)
        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - eps)
        too_large_step = dparam_new_batch > (param_ranges + eps)

        for i in range(n_freqs):
            for j in range(1, poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points

                if j == 1:
                    n_branch_snap[i] = num_points

        # Determine validation fraction
        snap_active = n_branch_snap[i] > minimum_snap_cells
        # Apply 0.5x if this is the first time snap becomes active
        just_active = snap_active & (~snap_first_branched)
        # Correction factor array
        val_factor = np.ones(n_freqs, dtype=np.float64)
        val_factor[just_active] = 0.5
        n_branches *= val_factor
        snap_first_branched |= just_active

        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs
        dparam_cur_batch = dparam_cur_next
    return branching_pattern
