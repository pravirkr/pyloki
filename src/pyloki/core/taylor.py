from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.core.common import get_leaves
from pyloki.utils import maths, np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL, FLOAT_EPSILON
from pyloki.utils.snail import MiddleOutScheme


@njit(cache=True, fastmath=True)
def poly_taylor_seed(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the seed leaves for Taylor polynomial search.

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
        The seed leaves. Shape is (n_leaves, total_size).

    Notes
    -----
    Conventions for each seed leaf: leaf_params + leaf_basis
    total_size = (n_params + 2) * 2 + (n_params * n_params)
    leaf_params[:-1, 0] -> Taylor polynomial coefficients, [d_poly_order, ..., d_1, d_0]
    leaf_params[:-1, 1] -> Grid size (error) on each coefficient,
    leaf_params[-1, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf_params[-1, 1]  -> Flag to indicate basis change (0: Polynomial, 1: Physical)
    leaf_basis[:] -> Basis matrix, shape is (n_params, n_params)
    """
    _, _ = coord_init
    leaves_taylor = get_leaves(param_arr, dparams)
    n_leaves = len(leaves_taylor)
    n_params = poly_order

    param_rows = n_params + 2
    bo = param_rows * 2
    basis_size = n_params * n_params
    total_size = bo + basis_size
    leaves = np.zeros((n_leaves, total_size), dtype=np.float64)

    f0_batch = leaves_taylor[:, -1, 0]
    df_batch = leaves_taylor[:, -1, 1]

    for i in range(n_leaves):
        params = leaves[i, :bo].reshape(param_rows, 2)
        # Copy till accel
        params[:-3, :] = leaves_taylor[i, :-1, :]
        # f = f0(1 - v / C) => dv = -(C/f0) * df
        params[-3, 0] = 0.0
        params[-3, 1] = df_batch[i] * (C_VAL / f0_batch[i])
        # initialize d0 (measure from t=t_init)
        params[-2, 0] = 0.0  # we never branch on d0
        # f0 and flag
        params[-1, 0] = f0_batch[i]
        params[-1, 1] = 0.0  # polynomial basis

        basis = leaves[i, bo : bo + basis_size].reshape(n_params, n_params)
        for j in range(n_params):
            basis[j, j] = params[j, 1]
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: np.ndarray,
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Parameter sets (leaves) to branch. Shape: (n_leaves, total_size).
        total_size = (n_params + 2) * 2 + (n_params * n_params)
    coord_cur : tuple[float, float]
        Coordinates for the accumulated segment in the current stage.
    nbins : int
        Number of bins in the folded profile.
    eta : float
        Tolerance for the parameter step size in bins.
    poly_order : int
        The order of the Taylor polynomial.
    param_limits : np.ndarray
        The limits for each parameter in Taylor basis (reverse order).
    branch_max : int
        Maximum number of branches that can be generated.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets. Shape: (n_branch, total_size).
    """
    n_batch, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur

    n_params = poly_order
    param_rows = n_params + 2
    bo = param_rows * 2
    basis_size = n_params * n_params
    total_size = bo + basis_size

    dparam_cur_batch = np.empty((n_batch, n_params), dtype=np.float64)
    f0_batch = np.empty(n_batch, dtype=np.float64)
    for i in range(n_batch):
        row = leaves_batch[i]
        f0_batch[i] = row[2 * (param_rows - 1) + 0]
        for j in range(n_params):
            dparam_cur_batch[i, j] = row[(2 * j) + 1]

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
    # Vectorized Padded Branching
    pad_branched_fracs = np.empty((n_batch, n_params, branch_max), dtype=np.float64)
    branched_scales = np.empty((n_batch, n_params), dtype=np.float64)
    branched_counts = np.empty((n_batch, n_params), dtype=np.int64)
    for i in range(n_batch):
        for j in range(n_params):
            # Get purely fractional offsets [-0.5, 0.5]
            scale, count = psr_utils.branch_logical_padded(
                pad_branched_fracs[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
            )
            branched_scales[i, j] = scale
            branched_counts[i, j] = count

    # Vectorized Selection
    for i in range(n_batch):
        for j in range(n_params):
            if shift_bins_batch[i, j] < (eta - FLOAT_EPSILON):
                branched_scales[i, j] = 1.0
                branched_counts[i, j] = 1
                for k in range(branch_max):
                    pad_branched_fracs[i, j, k] = 0.0
    # Optimized Padded Cartesian Product (fractional offsets)
    leaves_branch_fracs_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_fracs,
        branched_counts,
        n_batch,
        n_params,
    )
    total_leaves = batch_origins.shape[0]
    leaves_branch_batch = np.zeros((total_leaves, total_size), dtype=np.float64)
    # Child centers and basis
    for i in range(total_leaves):
        parent_idx = batch_origins[i]
        parent_row = leaves_batch[parent_idx]
        child_row = leaves_branch_batch[i]
        fracs = leaves_branch_fracs_batch[i]

        # Compute physical displacement and write child centers
        # \phys_disp = fracs @ parent_basis.T
        # New center = Old center + Physical Displacement
        for j in range(n_params):
            phys_disp = np.float64(0.0)
            for k in range(n_params):
                phys_disp += fracs[k] * parent_row[bo + (k * n_params) + j]
            child_row[2 * j] = parent_row[2 * j] + phys_disp

        # Scale the child basis. The child tile geometry is the same as the parent
        # but scaled by the branching factor.
        for j in range(n_params):
            scale = branched_scales[parent_idx, j]
            for k in range(n_params):
                idx = bo + (k * n_params) + j
                val = parent_row[idx] * scale
                child_row[idx] = val
                # diagonal entry → parameter extent
                # Store current valid extent (diagonal of child basis) in dparam slot
                if k == j:
                    child_row[(2 * j) + 1] = np.abs(val)

        # Copy over d0, f0, flags...
        idx = 2 * (param_rows - 2)
        child_row[idx] = parent_row[idx]
        idx = 2 * (param_rows - 1)
        child_row[idx] = parent_row[idx]
        child_row[idx + 1] = parent_row[idx + 1]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
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
    param_grid_count_init : np.ndarray
        Number of points in the initial (FFA) grid for the ``coord_add`` segment
        Currently this is simply [n_accel, n_freq].
    param_limits : np.ndarray
        Parameter limits (min, max).
    nbins : int
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
    relative_phase = psr_utils.get_phase_idx(t0_add - t0_init, f0, nbins, delay)
    pset_res = np.empty(2, dtype=np.float64)
    pset_res[0] = accel_new
    pset_res[1] = freq_new
    param_idx = psr_utils.get_nearest_indices_analytical(
        pset_res,
        param_grid_count_init[-2:],
        param_limits[-2:],
    )
    return param_idx, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
    poly_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _ = leaves_batch.shape
    t0_cur, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add

    param_vec_batch = np.empty((n_batch, poly_order + 1), dtype=np.float64)
    f0_batch = np.empty(n_batch, dtype=np.float64)
    for i in range(n_batch):
        row = leaves_batch[i]
        for j in range(poly_order + 1):
            param_vec_batch[i, j] = row[2 * j]
        # f0 row (last param row)
        f0_base = 2 * (poly_order + 1)
        f0_batch[i] = row[f0_base]

    dvec_t_add = transforms.shift_taylor_params(param_vec_batch, t0_add - t0_cur)
    dvec_t_init = transforms.shift_taylor_params(param_vec_batch, t0_init - t0_cur)
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
    # Pass the full param_limits to infer correct n_params
    param_idx_batch = psr_utils.get_nearest_indices_2d_batch(
        accel_new_batch,
        freq_new_batch,
        param_grid_count_init,
        param_limits,
    )
    return param_idx_batch, relative_phase_batch


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    _, _ = leaves_batch.shape
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    dvec_t_add = transforms.shift_taylor_params(param_vec_batch, t0_add - t0_init)
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
def poly_taylor_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    poly_order: int,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    n_batch, _ = leaves_batch.shape
    k = poly_order
    param_rows = k + 2
    param_size = param_rows * 2
    basis_size = k * k
    bo = param_size
    delta_t = coord_next[0] - coord_cur[0]
    leaves_batch_trans = leaves_batch.copy()

    # Construct the transformation matrix
    powers = np.tril(np.arange(k + 1)[:, np.newaxis] - np.arange(k + 1))
    t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))

    # Apply transform to each leaf
    taylor_coeffs = np.empty(k + 1, dtype=np.float64)
    basis_block = np.empty(basis_size, dtype=np.float64)
    for i in range(n_batch):
        row = leaves_batch_trans[i]
        # \Parameters (taylor_coeffs @ t_mat.T)
        for j in range(k + 1):
            taylor_coeffs[j] = row[2 * j]
        for j in range(k + 1):
            acc = np.float64(0.0)
            for m in range(k + 1):
                acc += taylor_coeffs[m] * t_mat[j, m]
            row[2 * j] = acc
        # \Basis (t_mat[:-1,:-1] @ basis_block)
        for r in range(k):
            for c in range(k):
                acc = np.float64(0.0)
                for m in range(k):
                    acc += t_mat[r, m] * row[bo + (m * k) + c]
                basis_block[(r * k) + c] = acc
        for r in range(k):
            for c in range(k):
                row[bo + (r * k) + c] = basis_block[(r * k) + c]
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_taylor_report_batch(leaves: np.ndarray, poly_order: int) -> np.ndarray:
    n_batch, _ = leaves.shape
    param_rows = poly_order + 2
    out = np.empty((n_batch, poly_order, 2), dtype=np.float64)

    for i in range(n_batch):
        row = leaves[i]
        v_base = 2 * (param_rows - 3)  # -3 row
        v_final = row[v_base]
        dv_final = row[v_base + 1]
        f0_base = 2 * (param_rows - 1)  # last row
        f0 = row[f0_base]
        s_factor = 1.0 - v_final / C_VAL

        for j in range(poly_order - 1):
            val = row[2 * j]
            sig = row[(2 * j) + 1]
            val_new = val / s_factor
            sig_new = np.sqrt(
                (sig / s_factor) ** 2
                + ((val / (C_VAL * s_factor * s_factor)) ** 2) * (dv_final * dv_final),
            )
            out[i, j, 0] = val_new
            out[i, j, 1] = sig_new
        out[i, poly_order - 1, 0] = f0 * s_factor
        out[i, poly_order - 1, 1] = f0 * dv_final / C_VAL

    return out


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_report_batch(
    leaves: np.ndarray,
    coord_mid: tuple[float, float],
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Specialized version of shift_taylor_params_d_f_batch for final report."""
    delta_t = coord_mid[0] - coord_init[0]
    param_vec_batch = leaves[:, :-2]
    n_batch, nparams, _ = param_vec_batch.shape
    taylor_param_vec = np.zeros((n_batch, nparams + 1), dtype=param_vec_batch.dtype)
    taylor_param_vec[:, :-2] = param_vec_batch[:, :-1, 0]  # till acceleration
    taylor_param_vec_new = transforms.shift_taylor_params(taylor_param_vec, delta_t)
    s_factor = 1 - taylor_param_vec_new[:, -2] / C_VAL
    param_vec_new = param_vec_batch.copy()
    param_vec_new[:, :-1, 0] = taylor_param_vec_new[:, :-2] / s_factor[:, None]
    param_vec_new[:, -1, 0] = param_vec_batch[:, -1, 0] * s_factor
    return param_vec_new


@njit(cache=True, fastmath=True)
def generate_bp_poly_taylor_approx(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: np.ndarray,
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    itree: int = 0,
    branch_max: int = 256,
) -> np.ndarray:
    """Generate the approximate branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    coord_init = snail_scheme.get_coord(0)
    leaves_init = poly_taylor_seed(param_arr, dparams_lim, poly_order, coord_init)
    leaf = leaves_init[itree : itree + 1]  # shape: (1, total_size)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(prune_level)
        leaves_arr = poly_taylor_branch_batch(
            leaf,
            coord_cur,
            nbins,
            eta,
            poly_order,
            param_limits,
            branch_max,
        )
        branching_pattern[prune_level - 1] = len(leaves_arr)
        leaves_arr_trans = poly_taylor_transform_batch(
            leaves_arr,
            coord_next,
            coord_cur,
            poly_order,
        )
        leaf = leaves_arr_trans[0:1]  # shape: (1, total_size)
    # Check if any branches is truncated due to branch_max
    if np.any(branching_pattern == branch_max):
        msg = "Branching pattern is truncated due to branch_max. Increase branch_max."
        raise ValueError(msg)
    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_bp_poly_taylor(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: np.ndarray,
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    weights = np.ones(n_freqs, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    for prune_level in range(1, nsegments):
        coord_cur = snail_scheme.get_current_coord(prune_level)
        _, t_obs_minus_t_ref = coord_cur
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec_limited(
            poly_order,
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

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - FLOAT_EPSILON)

        for i in range(n_freqs):
            for j in range(poly_order):
                if not needs_branching[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                numerator = dparam_cur_batch[i, j] + FLOAT_EPSILON
                ratio = numerator / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - FLOAT_EPSILON)))
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points
        # Compute average branching factor
        children = np.sum(weights * n_branches)
        parents = np.sum(weights)
        branching_pattern[prune_level - 1] = children / parents
        # Update weights and dparams
        weights *= n_branches

        dparam_cur_batch = dparam_cur_next
    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_bp_poly_taylor_fixed(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: np.ndarray,
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor fixed pruning search."""
    poly_order = len(dparams_lim)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    weights = np.ones(n_freqs, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    for prune_level in range(1, nsegments):
        coord_cur_fixed = snail_scheme.get_current_coord_fixed(prune_level)
        _, t_obs_minus_t_ref = coord_cur_fixed
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec_limited(
            poly_order,
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

        dparam_cur_next = np.empty((n_freqs, poly_order), dtype=np.float64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - FLOAT_EPSILON)

        for i in range(n_freqs):
            for j in range(poly_order):
                if not needs_branching[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                numerator = dparam_cur_batch[i, j] + FLOAT_EPSILON
                ratio = numerator / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - FLOAT_EPSILON)))
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points
        # Compute average branching factor
        children = np.sum(weights * n_branches)
        parents = np.sum(weights)
        branching_pattern[prune_level - 1] = children / parents
        # Update weights and dparams
        weights *= n_branches
        dparam_cur_batch = dparam_cur_next

    return branching_pattern
