from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.core.common import get_leaves
from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL, FLOAT_EPSILON
from pyloki.utils.snail import MiddleOutScheme


@njit(cache=True, fastmath=True)
def poly_taylor_seed(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
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
    tuple[np.ndarray, np.ndarray]
        - leaf_params: The seed leaf centers. Shape: (n_leaves, n_params + 3).
        - leaf_bases: The seed leaf bases. Shape: (n_leaves, n_params, n_params).

    Notes
    -----
    Conventions for each seed leaf:
    leaf_params[:-2] -> Taylor polynomial coefficients, [d_k, ..., d_1, d_0]
    leaf_params[-2]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf_params[-1]  -> Flag to indicate basis change (0: Polynomial, 1: Physical)
    leaf_bases[:] -> Basis matrix, shape is (n_params, n_params)
    leaf_bases[j, j] -> Grid size (error) on the j-th coefficient
    """
    _, _ = coord_init
    leaves_taylor = get_leaves(param_arr, dparams)
    n_leaves = len(leaves_taylor)
    n_params = poly_order

    f0_batch = leaves_taylor[:, -1, 0]
    df_batch = leaves_taylor[:, -1, 1]
    leaf_params = np.zeros((n_leaves, n_params + 3), dtype=np.float64)
    # Copy till accel
    leaf_params[:, :-4] = leaves_taylor[:, :-1, 0]
    # initialize d1 = 0 (measure from t=t_init) for velocity
    leaf_params[:, -4] = 0
    # intialize d0 = 0 (measure from t=t_init) for displacement
    leaf_params[:, -3] = 0.0
    leaf_params[:, -2] = f0_batch
    leaf_params[:, -1] = 0  # polynomial basis

    leaf_bases = np.zeros((n_leaves, n_params, n_params), dtype=np.float64)
    for i in range(n_leaves):
        # Copy till accel
        for j in range(n_params - 1):
            leaf_bases[i, j, j] = leaves_taylor[i, j, 1]
        # f = f0(1 - v / C) => dv = -(C/f0) * df
        leaf_bases[i, n_params - 1, n_params - 1] = df_batch[i] * (C_VAL / f0_batch[i])
    return leaf_params, leaf_bases


@njit(cache=True, fastmath=True)
def poly_taylor_branch_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_cur: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: np.ndarray,
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Branch a batch of tree parameter nodes to leaves.

    Parameters
    ----------
    leaf_params_batch : np.ndarray
        Leaf parameter sets. Shape: (n_leaves, n_params + 3).
    leaf_bases_batch : np.ndarray
        Leaf basis matrices. Shape: (n_leaves, n_params, n_params).
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
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - leaf_params_branch_batch: Array of leaf centers.
          Shape: (n_branch, n_params + 3).
        - leaf_bases_branch_batch: Array of leaf bases.
          Shape: (n_branch, n_params, n_params).
        - batch_origins: Array of original indices.
          Shape: (n_branch,).
    """
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
    n_branch = batch_origins.shape[0]
    leaf_params_branch_batch = np.empty((n_branch, n_params + 3), dtype=np.float64)
    leaf_bases_branch_batch = np.empty((n_branch, n_params, n_params), dtype=np.float64)
    # Child centers and basis
    for i in range(n_branch):
        parent_idx = batch_origins[i]
        parent_param = leaf_params_batch[parent_idx]
        parent_basis = leaf_bases_batch[parent_idx]
        child_param = leaf_params_branch_batch[i]
        child_basis = leaf_bases_branch_batch[i]

        # Compute physical displacement and write child centers
        # \disp = parent_basis @ leaves_branch_fracs_batch[i]
        # New center = Old center + Physical Displacement
        for j in range(n_params):
            phys_disp = np.float64(0.0)
            for k in range(n_params):
                phys_disp += parent_basis[j, k] * leaves_branch_fracs_batch[i, k]
            child_param[j] = parent_param[j] + phys_disp

        # Copy over d0, f0, flags...
        child_param[n_params:] = parent_param[n_params:]

        # Scale the child basis. The child tile geometry is the same as the parent
        # but scaled by the branching factor.
        for j in range(n_params):
            scale = branched_scales[parent_idx, j]
            for k in range(n_params):
                child_basis[k, j] = parent_basis[k, j] * scale

    return leaf_params_branch_batch, leaf_bases_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_batch(
    leaf_params_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift.

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

    param_vec_batch = leaf_params_batch[:, :-2]
    f0_batch = leaf_params_batch[:, -2]

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
    leaf_params_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add

    param_vec_batch = leaf_params_batch[:, :-2]
    f0_batch = leaf_params_batch[:, -2]

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
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> None:
    """Re-center (in-place) the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    transforms.shift_taylor_params_basis(leaf_params_batch, leaf_bases_batch, delta_t)


@njit(cache=True, fastmath=True)
def poly_taylor_report_batch(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    poly_order: int,
) -> np.ndarray:
    n_leaves, _ = leaf_params_batch.shape
    n_params = poly_order
    out = np.empty((n_leaves, n_params, 2), dtype=np.float64)

    for i in range(n_leaves):
        v_final = leaf_params_batch[i, n_params - 1]
        dv_final = np.abs(leaf_bases_batch[i, n_params - 1, n_params - 1])
        f0 = leaf_params_batch[i, n_params + 1]
        s_factor = 1.0 - v_final / C_VAL

        for j in range(n_params - 1):
            val = leaf_params_batch[i, j]
            sig = np.abs(leaf_bases_batch[i, j, j])
            out[i, j, 0] = val / s_factor
            out[i, j, 1] = np.sqrt(
                (sig / s_factor) ** 2
                + ((val / (C_VAL * s_factor * s_factor)) ** 2) * (dv_final * dv_final),
            )
        out[i, n_params - 1, 0] = f0 * s_factor
        out[i, n_params - 1, 1] = f0 * dv_final / C_VAL

    return out


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
    use_moving_grid: bool,
    itree: int = 0,
    branch_max: int = 256,
) -> np.ndarray:
    """Generate the approximate branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    coord_init = snail_scheme.get_coord(0)
    leaf_params_init, leaf_bases_init = poly_taylor_seed(
        param_arr,
        dparams_lim,
        poly_order,
        coord_init,
    )
    leaf_param = leaf_params_init[itree : itree + 1]  # shape: (1, total_size)
    leaf_basis = leaf_bases_init[itree : itree + 1]  # shape: (1, n_params, n_params)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        leaf_params, leaf_bases, _ = poly_taylor_branch_batch(
            leaf_param,
            leaf_basis,
            coord_cur,
            nbins,
            eta,
            poly_order,
            param_limits,
            branch_max,
        )
        branching_pattern[prune_level - 1] = len(leaf_params)
        if use_moving_grid:
            poly_taylor_transform_batch(
                leaf_params,
                leaf_bases,
                coord_next,
                coord_cur,
                poly_order,
            )
        leaf_param = leaf_params[0:1]  # shape: (1, total_size)
        leaf_basis = leaf_bases[0:1]  # shape: (1, n_params, n_params)
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
    use_moving_grid: bool,
    use_cheby_coarsening: bool = True,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    n_params = len(dparams_lim)
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
        n_branches = np.ones(n_freqs, dtype=np.int64)

        for i in range(n_freqs):
            for j in range(n_params):
                if shift_bins_batch[i, j] < (eta - FLOAT_EPSILON):
                    branched_scales[i, j] = 1.0
                    continue
                numerator = dparam_cur_batch[i, j] + FLOAT_EPSILON
                ratio = numerator / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - FLOAT_EPSILON)))
                n_branches[i] *= num_points
                branched_scales[i, j] = 1.0 / num_points
        # Scale the parent basis. The child tile geometry is the same as the parent
        # but scaled by the branching factor.
        leaf_bases = leaf_bases * branched_scales[:, None, :]
        # Compute average branching factor
        children = np.sum(weights * n_branches)
        parents = np.sum(weights)
        branching_pattern[prune_level - 1] = children / parents
        # Update weights and dparams
        weights *= n_branches

        if use_moving_grid:
            # Transform basis to the next segment
            delta_t = coord_next[0] - coord_cur[0]
            transforms.shift_taylor_basis(leaf_bases, delta_t)
    return branching_pattern
