from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL, FLOAT_EPSILON
from pyloki.utils.snail import MiddleOutScheme


@njit(cache=True, fastmath=True)
def get_circ_mask(
    crackle: np.ndarray,
    dcrackle: np.ndarray,
    snap: np.ndarray,
    dsnap: np.ndarray,
    jerk: np.ndarray,
    accel: np.ndarray,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify Taylor grid cells as circular orbital motion.

    Parameters
    ----------
    crackle : array
        Fifth derivative (d5).
    dcrackle : array
        Uncertainty in crackle.
    snap : array
        Fourth derivative (d4).
    dsnap : array
        Uncertainty in snap.
    jerk : array
        Third derivative (d3).
    accel : array
        Second derivative (d2).
    minimum_snap_cells : float
        Detection threshold for significant snap/crackle measurements
        expressed in units of the derivative uncertainty.

    Returns
    -------
    idx_circular_snap : ndarray
        Indices of cells where circular motion is detected using the
        snap-accel relation.
    idx_circular_crackle : ndarray
        Indices of cells where circular motion is detected using the
        crackle-jerk fallback relation.
    idx_taylor : ndarray
        Indices of cells not consistent with circular motion.
    """
    # Determine whether derivatives are significantly measured
    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + FLOAT_EPSILON))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + FLOAT_EPSILON))

    # Snap-Dominated Region (Standard)
    # We check if implied Omega is physical (-d4/d2 > 0)
    is_physical_snap = (
        ((-snap * accel) > 0)
        & (np.abs(accel) > FLOAT_EPSILON)
        & (np.abs(snap) > FLOAT_EPSILON)
    )
    mask_circular_snap = is_sig_snap & is_physical_snap

    # Crackle-Dominated Region (The Hole)
    # Condition: Snap is weak (in the null), but Crackle is strong.
    # Note: d2 and d4 vanish in the hole, so we rely on d3 and d5.
    is_small_accel = np.abs(accel) < FLOAT_EPSILON
    in_the_hole = (~is_sig_snap) & is_sig_crackle & is_small_accel

    # Check if implied Omega is physical (-d5/d3 > 0)
    is_physical_crackle = ((-crackle * jerk) > 0) & (np.abs(jerk) > FLOAT_EPSILON)
    mask_circular_crackle = in_the_hole & is_physical_crackle

    # Taylor Region (Noise / Unresolved)
    # Everything that isn't a confident circular candidate
    mask_taylor = ~(mask_circular_snap | mask_circular_crackle)

    idx_circular_snap = np.flatnonzero(mask_circular_snap)
    idx_circular_crackle = np.flatnonzero(mask_circular_crackle)
    idx_taylor = np.flatnonzero(mask_taylor)

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
    minimum_snap_cells: float,
) -> np.ndarray:
    n_leaves = len(crackle)
    mask_keep = np.zeros(n_leaves, dtype=np.bool_)
    omega_max_sq = (2 * np.pi / p_orb_min) ** 2

    # Classification (for gatekeeping)
    # |val / step| > thresh  <=>  |val| > thresh * |step|
    # dsnap and dcrackle are step sizes (always positive)
    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + FLOAT_EPSILON))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + FLOAT_EPSILON))

    # Unresolved Taylor cells are always kept
    is_noise = (~is_sig_snap) & (~is_sig_crackle)
    mask_keep |= is_noise

    # Snap-dominated region
    idx_snap = np.flatnonzero(is_sig_snap)
    if len(idx_snap) > 0:
        s_snap = snap[idx_snap]
        s_accel = accel[idx_snap]
        s_jerk = jerk[idx_snap]
        # Check: Physical sign (-d4/d2 > 0)
        valid_sign = (
            ((-s_snap * s_accel) > 0)
            & (np.abs(s_accel) > FLOAT_EPSILON)
            & (np.abs(s_snap) > FLOAT_EPSILON)
        )
        s_omega_sq = np.zeros_like(s_snap)
        s_omega_sq[valid_sign] = -s_snap[valid_sign] / s_accel[valid_sign]

        # Check: Max Orbital Frequency
        valid_omega = s_omega_sq < omega_max_sq

        # |d2| < x * omega^(4/3)
        s_omega_sq_safe = np.abs(s_omega_sq)
        limit_accel = x_mass_const * (s_omega_sq_safe ** (2 / 3))
        valid_accel = np.abs(s_accel) < limit_accel

        # |d3| < |d2| * omega
        valid_jerk = (s_jerk**2) < (s_accel**2 * s_omega_sq_safe)

        mask_snap_local = valid_sign & valid_omega & valid_accel & valid_jerk
        mask_keep[idx_snap] |= mask_snap_local

    # Crackle-Dominated Region (The Hole)
    # Only if snap is not significant but crackle is
    is_hole = (~is_sig_snap) & is_sig_crackle & (np.abs(accel) < FLOAT_EPSILON)
    idx_hole = np.flatnonzero(is_hole)
    if len(idx_hole) > 0:
        h_crackle = crackle[idx_hole]
        h_jerk = jerk[idx_hole]
        h_accel = accel[idx_hole]

        valid_sign = ((-h_crackle * h_jerk) > 0) & (np.abs(h_jerk) > FLOAT_EPSILON)
        h_omega_sq = np.zeros_like(h_crackle)
        h_omega_sq[valid_sign] = -h_crackle[valid_sign] / h_jerk[valid_sign]

        valid_omega = h_omega_sq < omega_max_sq

        # |d2| < x * omega^(4/3)
        h_omega_sq_safe = np.abs(h_omega_sq)
        limit_accel = x_mass_const * (h_omega_sq_safe ** (2 / 3))
        valid_accel = np.abs(h_accel) < limit_accel

        # |d3| < |d2| * omega
        valid_jerk = (h_jerk**2) < (h_accel**2 * h_omega_sq_safe)

        mask_hole_local = valid_sign & valid_omega & valid_accel & valid_jerk
        mask_keep[idx_hole] |= mask_hole_local

    return np.flatnonzero(mask_keep)


@njit(cache=True, fastmath=True)
def get_circ_taylor_mask(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    minimum_snap_cells: float = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a robust mask to identify circular orbit candidates.

    Filters out physically implausible and numerically unstable orbits.

    Parameters
    ----------
    leaf_params_batch : np.ndarray
        Shape (n_batch, nparams + 3)
    leaf_bases_batch : np.ndarray
        Shape (n_batch, nparams, nparams)
    minimum_snap_cells: float
        Threshold for significant snap (number of snap grid cells). Defaults to 5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A boolean array where True indicates a high-quality circular orbit candidate.

    """
    crackle = leaf_params_batch[:, 0]
    snap = leaf_params_batch[:, 1]
    jerk = leaf_params_batch[:, 2]
    accel = leaf_params_batch[:, 3]
    dcrackle = np.abs(leaf_bases_batch[:, 0, 0])
    dsnap = np.abs(leaf_bases_batch[:, 1, 1])
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
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the indices of the leaves that need to be expanded (crackle)."""
    crackle = leaves_params_batch[:, 0]
    snap = leaves_params_batch[:, 1]
    jerk = leaves_params_batch[:, 2]
    accel = leaves_params_batch[:, 3]

    dcrackle = leaves_dparams_batch[:, 0]
    dsnap = leaves_dparams_batch[:, 1]

    is_sig_snap = np.abs(snap) > (minimum_snap_cells * (dsnap + FLOAT_EPSILON))
    is_sig_crackle = np.abs(crackle) > (minimum_snap_cells * (dcrackle + FLOAT_EPSILON))
    in_the_hole = (~is_sig_snap) & is_sig_crackle & (np.abs(accel) < FLOAT_EPSILON)
    is_physical_crackle = ((-crackle * jerk) > 0) & (np.abs(jerk) > FLOAT_EPSILON)
    mask_circular_crackle = in_the_hole & is_physical_crackle

    # Optimization: Filter out 'crackle' candidates that don't actually need branching
    # True crackle candidates: in the hole AND step size is large enough
    mask_expand_crackle = mask_circular_crackle & batch_needs_crackle
    idx_expand_crackle = np.flatnonzero(mask_expand_crackle)
    # Keep indices: Taylor + Snap + Crackle-that-doesnt-need-branching
    mask_keep = ~mask_expand_crackle
    idx_keep = np.flatnonzero(mask_keep)

    return idx_expand_crackle, idx_keep


@njit(cache=True, fastmath=True)
def circ_taylor_branch_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: np.ndarray,
    branch_max: int,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch, _ = leaf_params_batch.shape
    _, t_obs_minus_t_ref = coord_cur
    n_params = poly_order

    dparam_cur_batch = np.empty((n_batch, n_params), dtype=np.float64)
    f0_batch = np.empty(n_batch, dtype=np.float64)
    for i in range(n_batch):
        f0_batch[i] = leaf_params_batch[i, n_params + 1]
        for j in range(n_params):
            # basis diagonal -> current valid extent
            dparam_cur_batch[i, j] = np.abs(leaf_bases_batch[i, j, j])

    dparam_new_batch = psr_utils.poly_taylor_step_d_vec_limited(
        n_params,
        t_obs_minus_t_ref,
        nbins,
        eta,
        f0_batch,
        param_limits,
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
    # Vectorized Padded Branching ( All params except crackle) ---
    pad_branched_fracs = np.empty((n_batch, n_params, branch_max), dtype=np.float64)
    branched_scales = np.empty((n_batch, n_params), dtype=np.float64)
    branched_counts = np.empty((n_batch, n_params), dtype=np.int64)
    for i in range(n_batch):
        for j in range(1, n_params):  # skip crackle (j=0)
            # Get purely fractional offsets [-0.5, 0.5]
            scale, count = psr_utils.branch_logical_padded(
                pad_branched_fracs[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
            )
            branched_scales[i, j] = scale
            branched_counts[i, j] = count

    # Vectorized Selection (mask non-crackle branched params)
    needs_branching = shift_bins_batch >= (eta - FLOAT_EPSILON)
    for i in range(n_batch):
        for j in range(poly_order):
            if j == 0 or not needs_branching[i, j]:
                # crackle - don't branch yet, keep at current value
                branched_scales[i, j] = 1.0
                branched_counts[i, j] = 1
                for k in range(branch_max):
                    pad_branched_fracs[i, j, k] = 0.0
    # First Cartesian Product (All Except Crackle)
    # Note: These 'leaves' contain unbranched crackle (parent values)
    leaves_branch_fracs_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_fracs,
        branched_counts,
        n_batch,
        poly_order,
    )
    n_branch_tmp = batch_origins.shape[0]
    leaf_params_branch_batch_tmp = np.empty((n_branch_tmp, n_params), dtype=np.float64)
    # Child centers and basis
    for i in range(n_branch_tmp):
        parent_idx = batch_origins[i]
        parent_param = leaf_params_batch[parent_idx]
        parent_basis = leaf_bases_batch[parent_idx]
        child_param = leaf_params_branch_batch_tmp[i]
        # Compute physical displacement and write child centers
        # \disp = parent_basis @ leaves_branch_fracs_batch[i]
        # New center = Old center + Physical Displacement
        for j in range(n_params):
            phys_disp = 0.0
            for k in range(n_params):
                phys_disp += parent_basis[j, k] * leaves_branch_fracs_batch[i, k]
            child_param[j] = parent_param[j] + phys_disp

    # Check if the batch item originating this leaf needs crackle branching
    dparam_new_batch_act = dparam_cur_batch * branched_scales
    leaves_branched_dparams_tmp = dparam_new_batch_act[batch_origins]
    batch_needs_crackle = needs_branching[batch_origins, 0]
    idx_expand_crackle, idx_keep = get_circ_taylor_mask_branch(
        leaf_params_branch_batch_tmp,
        leaves_branched_dparams_tmp,
        batch_needs_crackle,
        minimum_snap_cells=minimum_snap_cells,
    )
    n_keep = len(idx_keep)
    n_crackle_expand = len(idx_expand_crackle)

    if n_crackle_expand == 0:
        n_branch = n_keep
        leaf_params_branch_batch = np.empty((n_branch, n_params + 3), dtype=np.float64)
        leaf_bases_branch_batch = np.empty(
            (n_branch, n_params, n_params),
            dtype=np.float64,
        )
        origins_final = batch_origins[idx_keep]
        for i in range(n_branch):
            src = idx_keep[i]
            parent_idx = batch_origins[src]
            parent_param = leaf_params_batch[parent_idx]
            parent_basis = leaf_bases_batch[parent_idx]
            child_param = leaf_params_branch_batch[i]
            child_basis = leaf_bases_branch_batch[i]
            child_param[:n_params] = leaf_params_branch_batch_tmp[src]
            child_param[n_params:] = parent_param[n_params:]

            # scale basis
            for j in range(n_params):
                scale = branched_scales[parent_idx, j]
                for k in range(n_params):
                    child_basis[k, j] = parent_basis[k, j] * scale

        return leaf_params_branch_batch, leaf_bases_branch_batch, origins_final

    # Branch crackle parameter for these specific leaves
    crackle_fracs = np.empty((n_crackle_expand, branch_max), dtype=np.float64)
    crackle_scales = np.empty(n_crackle_expand, dtype=np.float64)
    crackle_counts = np.empty(n_crackle_expand, dtype=np.int64)

    for i in range(n_crackle_expand):
        leaf_idx = idx_expand_crackle[i]
        parent_idx = batch_origins[leaf_idx]
        scale, count = psr_utils.branch_logical_padded(
            crackle_fracs[i],
            leaves_branched_dparams_tmp[leaf_idx, 0],
            dparam_new_batch[parent_idx, 0],
        )
        crackle_scales[i] = scale
        crackle_counts[i] = count

    # Construct Final Array
    total_crackle = np.sum(crackle_counts)
    total_leaves = n_keep + total_crackle
    leaf_params_final = np.empty((total_leaves, n_params + 3), dtype=np.float64)
    leaf_bases_final = np.empty((total_leaves, n_params, n_params), dtype=np.float64)
    origins_final = np.empty(total_leaves, dtype=np.int64)

    for i in range(n_keep):
        src = idx_keep[i]
        parent_idx = batch_origins[src]
        parent_param = leaf_params_batch[parent_idx]
        parent_basis = leaf_bases_batch[parent_idx]
        child_param = leaf_params_final[i]
        child_basis = leaf_bases_final[i]
        child_param[:n_params] = leaf_params_branch_batch_tmp[src]
        child_param[n_params:] = parent_param[n_params:]
        for j in range(n_params):
            scale = branched_scales[parent_idx, j]
            for k in range(n_params):
                child_basis[k, j] = parent_basis[k, j] * scale

        origins_final[i] = parent_idx

    current = n_keep
    for i in range(n_crackle_expand):
        leaf_idx = idx_expand_crackle[i]
        parent_idx = batch_origins[leaf_idx]
        parent_param = leaf_params_batch[parent_idx]
        parent_basis = leaf_bases_batch[parent_idx]
        center = leaf_params_branch_batch_tmp[leaf_idx]
        count = crackle_counts[i]
        for a in range(count):
            child_param = leaf_params_final[current]
            child_basis = leaf_bases_final[current]
            # displacement vector
            frac = crackle_fracs[i, a]
            # crackle displacement
            for j in range(n_params):
                child_param[j] = center[j] + parent_basis[j, 0] * frac
            child_param[n_params:] = parent_param[n_params:]
            # scale basis
            for j in range(n_params):
                scale = crackle_scales[i] if j == 0 else branched_scales[parent_idx, j]
                for k in range(n_params):
                    child_basis[k, j] = parent_basis[k, j] * scale
            origins_final[current] = parent_idx
            current += 1
    return leaf_params_final, leaf_bases_final, origins_final


@njit(cache=True, fastmath=True)
def circ_taylor_validate_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    leaves_origins: np.ndarray,
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a batch of leaf params.

    Filters out unphysical orbits. Currently removes cases with imaginary orbital
    frequency (i.e., -snap/accel <= 0) and orbital frequency > omega_orb_max.

    Parameters
    ----------
    leaf_params_batch : np.ndarray
        The leaf parameter sets to validate. Shape: (N, nparams + 3)
    leaf_bases_batch : np.ndarray
        The leaf basis sets to validate. Shape: (N, nparams, nparams)
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
    crackle = leaf_params_batch[:, 0]
    snap = leaf_params_batch[:, 1]
    jerk = leaf_params_batch[:, 2]
    accel = leaf_params_batch[:, 3]
    dcrackle = np.abs(leaf_bases_batch[:, 0, 0])
    dsnap = np.abs(leaf_bases_batch[:, 1, 1])
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
    return leaf_params_batch[idx], leaf_bases_batch[idx], leaves_origins[idx]


@njit(cache=True, fastmath=True)
def circ_taylor_resolve_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    # only works for circular orbit when nparams = 5
    n_batch, _ = leaf_params_batch.shape
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add

    param_vec_batch = leaf_params_batch[:, :-2]
    f0_batch = leaf_params_batch[:, -2]

    idx_circ_snap, idx_circ_crackle, idx_taylor = get_circ_taylor_mask(
        leaf_params_batch,
        leaf_bases_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 6), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 6), dtype=np.float64)

    if idx_circ_snap.size > 0:
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_snap],
            t0_add - t0_cur,
        )
        dvec_t_init_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_snap],
            t0_init - t0_cur,
        )
        dvec_t_add[idx_circ_snap] = dvec_t_add_circ_snap
        dvec_t_init[idx_circ_snap] = dvec_t_init_circ_snap

    if idx_circ_crackle.size > 0:
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_crackle],
            t0_add - t0_cur,
            in_hole=True,
        )
        dvec_t_init_circ_crackle = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_crackle],
            t0_init - t0_cur,
            in_hole=True,
        )
        dvec_t_add[idx_circ_crackle] = dvec_t_add_circ_crackle
        dvec_t_init[idx_circ_crackle] = dvec_t_init_circ_crackle

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
    param_idx_batch = psr_utils.get_nearest_indices_2d_batch(
        accel_new_batch,
        freq_new_batch,
        param_grid_count_init,
        param_limits,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def circ_taylor_fixed_resolve_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
    minimum_snap_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _ = leaf_params_batch.shape
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add

    param_vec_batch = leaf_params_batch[:, :-2]
    f0_batch = leaf_params_batch[:, -2]

    idx_circ_snap, idx_circ_crackle, idx_taylor = get_circ_taylor_mask(
        leaf_params_batch,
        leaf_bases_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    dvec_t_add = np.empty((n_batch, 6), dtype=np.float64)
    if idx_circ_snap.size > 0:
        dvec_t_add_circ_snap = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_snap],
            t0_add - t0_init,
        )
        dvec_t_add[idx_circ_snap] = dvec_t_add_circ_snap

    if idx_circ_crackle.size > 0:
        dvec_t_add_circ_crackle = transforms.shift_taylor_circular_params(
            param_vec_batch[idx_circ_crackle],
            t0_add - t0_init,
            in_hole=True,
        )
        dvec_t_add[idx_circ_crackle] = dvec_t_add_circ_crackle

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
    param_idx_batch = psr_utils.get_nearest_indices_2d_batch(
        accel_new_batch,
        freq_new_batch,
        param_grid_count_init,
        param_limits,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def circ_taylor_transform_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    p_orb_min: float,
    minimum_snap_cells: float,
) -> None:
    """Re-center (in-place) the leaves to the next segment reference time."""
    idx_circ_snap, idx_circ_crackle, idx_taylor = get_circ_taylor_mask(
        leaf_params_batch,
        leaf_bases_batch,
        minimum_snap_cells=minimum_snap_cells,
    )
    delta_t = coord_next[0] - coord_cur[0]
    if idx_circ_snap.size > 0:
        transforms.shift_taylor_circular_params_basis(
            leaf_params_batch[idx_circ_snap],
            leaf_bases_batch[idx_circ_snap],
            delta_t,
            p_orb_min,
        )
    if idx_circ_crackle.size > 0:
        transforms.shift_taylor_circular_params_basis(
            leaf_params_batch[idx_circ_crackle],
            leaf_bases_batch[idx_circ_crackle],
            delta_t,
            p_orb_min,
            in_hole=True,
        )
    if idx_taylor.size > 0:
        transforms.shift_taylor_params_basis(
            leaf_params_batch[idx_taylor],
            leaf_bases_batch[idx_taylor],
            delta_t,
        )


@njit(cache=True, fastmath=True)
def cir_physical_branch_batch(
    param_vec_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: np.ndarray,
    branch_max: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_vec_batch)
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
def poly_circular_resolve_batch(
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    _, _, _ = leaf_batch.shape
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
    param_idx_batch = psr_utils.get_nearest_indices_2d_batch(
        a_new_batch,
        f_new_batch,
        param_grid_count_init,
        param_limits,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def generate_bp_circ_taylor(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: np.ndarray,
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    p_orb_min: float,
    minimum_snap_cells: float,
    use_moving_grid: bool,
    use_cheby_coarsening: bool = True,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor circular pruning search."""
    n_params = len(dparams_lim)
    if n_params != 5:
        msg = "Circular branching pattern requires exactly 5 parameters."
        raise ValueError(msg)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    weights = np.ones(n_freqs, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    leaf_bases = np.zeros((n_freqs, n_params, n_params), dtype=np.float64)
    for i in range(n_freqs):
        for j in range(n_params - 1):
            leaf_bases[i, j, j] = dparams_lim[j]
        # f = f0(1 - v / C) => dv = -(C/f0) * df
        leaf_bases[i, n_params - 1, n_params - 1] = dparams_lim[n_params - 1] * (
            C_VAL / f0_batch[i]
        )
    dparam_cur_batch = np.empty((n_freqs, n_params), dtype=np.float64)
    branched_scales = np.empty((n_freqs, n_params), dtype=np.float64)

    # Track when first snap branching occurs for each frequency
    snap_first_branched = np.zeros(n_freqs, dtype=np.bool_)
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        _, t_obs_minus_t_ref = coord_cur
        for i in range(n_freqs):
            for j in range(n_params):
                # basis diagonal -> current valid extent
                dparam_cur_batch[i, j] = np.abs(leaf_bases[i, j, j])

        dparam_new_batch = psr_utils.poly_taylor_step_d_vec_limited(
            n_params,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f0_batch,
            param_limits,
            t_ref=0,
            use_cheby=use_cheby_coarsening,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            nbins,
            f0_batch,
            t_ref=0,
            use_cheby=use_cheby_coarsening,
        )
        n_branch_snap = np.ones(n_freqs, dtype=np.float64)
        n_branches = np.ones(n_freqs, dtype=np.float64)

        for i in range(n_freqs):
            for j in range(1, n_params):
                if shift_bins_batch[i, j] < (eta - FLOAT_EPSILON):
                    branched_scales[i, j] = 1.0
                    continue
                numerator = dparam_cur_batch[i, j] + FLOAT_EPSILON
                ratio = numerator / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - FLOAT_EPSILON)))
                n_branches[i] *= num_points
                branched_scales[i, j] = 1.0 / num_points
                if j == 1:
                    n_branch_snap[i] = num_points
        # Scale the parent basis. The child tile geometry is the same as the parent
        # but scaled by the branching factor.
        leaf_bases = leaf_bases * branched_scales[:, None, :]

        # Determine validation fraction
        snap_val = param_limits[1, 1]  # Take max
        dsnap = np.abs(leaf_bases[:, 1, 1])
        snap_active_mask = np.abs(snap_val) > (
            minimum_snap_cells * (dsnap + FLOAT_EPSILON)
        )
        # Apply 0.5x if this is the first time snap becomes active
        just_active = snap_active_mask & (~snap_first_branched)
        # Correction factor array
        val_factor = np.ones(n_freqs, dtype=np.float64)
        val_factor[just_active] = 0.5
        n_branches *= val_factor
        snap_first_branched |= just_active

        # Compute average branching factor
        children = np.sum(weights * n_branches)
        parents = np.sum(weights)
        branching_pattern[prune_level - 1] = children / parents
        # Update weights and dparams
        weights *= n_branches

        if use_moving_grid:
            # Transform basis to the next segment
            delta_t = coord_next[0] - coord_cur[0]
            idx_circ_snap = np.flatnonzero(snap_active_mask)
            idx_taylor = np.flatnonzero(~snap_active_mask)
            if idx_circ_snap.size > 0:
                transforms.shift_taylor_circular_basis(
                    leaf_bases[idx_circ_snap],
                    delta_t,
                    p_orb_min,
                )
            if idx_taylor.size > 0:
                transforms.shift_taylor_basis(leaf_bases[idx_taylor], delta_t)

    return branching_pattern
