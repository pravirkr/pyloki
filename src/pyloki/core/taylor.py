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
        The seed leaves. Shape is (n_leaves, poly_order + 2, 2).

    Notes
    -----
    Conventions for each seed leaf:
    leaf[:-1, 0] -> Taylor polynomial coefficients,
                    order is [d_poly_order, ..., d_1, d_0]
    leaf[:-1, 1] -> Grid size (error) on each coefficient,
    leaf[-1, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf[-1, 1]  -> Flag to indicate basis change (0: Polynomial, 1: Physical)
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
    leaves[:, -1, 0] = f0_batch
    leaves[:, -1, 1] = 0  # Polynomial basis
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
    """Branch a batch of tree parameter nodes to leaves.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Leaf parameter sets. Shape: (n_leaves, poly_order + 2, 2).
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
    tuple[np.ndarray, np.ndarray]
        - leaves_branch_batch: Array of leaf centers.
          Shape: (n_branch, poly_order + 2, 2).
        - batch_origins: Array of original indices.
          Shape: (n_branch,).
    """
    n_batch, _, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur
    n_params = poly_order

    param_cur_batch = leaves_batch[:, :-2, 0]
    dparam_cur_batch = leaves_batch[:, :-2, 1]
    d0_cur_batch = leaves_batch[:, -2, 0]
    f0_batch = leaves_batch[:, -1, 0]
    basis_flag_batch = leaves_batch[:, -1, 1]

    dparam_new_batch_actual = psr_utils.poly_taylor_step_d_vec(
        n_params,
        t_obs_minus_t_ref,
        nbins,
        eta,
        f0_batch,
        t_ref=0,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
        dparam_cur_batch,
        dparam_new_batch_actual,
        t_obs_minus_t_ref,
        nbins,
        f0_batch,
        t_ref=0,
    )
    dparam_new_batch = psr_utils.poly_taylor_step_d_vec_limited(
        n_params,
        t_obs_minus_t_ref,
        nbins,
        eta,
        f0_batch,
        param_limits,
        t_ref=0,
    )

    # Vectorized Padded Branching
    pad_branched_params = np.empty((n_batch, n_params, branch_max), dtype=np.float64)
    branched_dparams = np.empty((n_batch, n_params), dtype=np.float64)
    branched_counts = np.empty((n_batch, n_params), dtype=np.int64)
    for i in range(n_batch):
        for j in range(n_params):
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
            )
            branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # Vectorized Selection
    for i in range(n_batch):
        for j in range(n_params):
            if shift_bins_batch[i, j] < (eta - FLOAT_EPSILON):
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1

    # Optimized Padded Cartesian Product
    leaf_params_branch_cart, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        n_params,
    )
    n_branch = len(batch_origins)
    leaves_branch_batch = np.zeros((n_branch, n_params + 2, 2), dtype=np.float64)
    leaves_branch_batch[:, :-2, 0] = leaf_params_branch_cart
    leaves_branch_batch[:, :-2, 1] = branched_dparams[batch_origins]
    leaves_branch_batch[:, -2, 0] = d0_cur_batch[batch_origins]
    leaves_branch_batch[:, -1, 0] = f0_batch[batch_origins]
    leaves_branch_batch[:, -1, 1] = basis_flag_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_resolve_batch(
    leaves_batch: np.ndarray,
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
    leaves_batch : np.ndarray
        The leaf parameter set. Shape is (n_leaves, poly_order + 2, 2).
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

    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

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
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
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
def poly_taylor_ascend_resolve_batch(
    leaves_batch: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    t0_cur, _ = coord_cur
    n_leaves = len(leaves_batch)
    nsegments = len(coord_segments)
    n_params = param_limits.shape[0]

    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    param_idx_batch_arr = np.empty((n_leaves, nsegments, n_params), dtype=np.int64)
    relative_phase_batch_arr = np.empty((n_leaves, nsegments), dtype=np.float64)
    for isegment in range(nsegments):
        t0_seg, _ = coord_segments[isegment]
        dvec_t_seg = transforms.shift_taylor_params(param_vec_batch, t0_seg - t0_cur)
        accel_new_batch = dvec_t_seg[:, -3]
        freq_new_batch = f0_batch * (1 - dvec_t_seg[:, -2] / C_VAL)
        delay_batch = dvec_t_seg[:, -1] / C_VAL
        relative_phase_batch = psr_utils.get_phase_idx(
            t0_seg - t0_cur,
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
        param_idx_batch_arr[:, isegment, :] = param_idx_batch
        relative_phase_batch_arr[:, isegment] = relative_phase_batch
    return param_idx_batch_arr, relative_phase_batch_arr


@njit(cache=True, fastmath=True)
def poly_taylor_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
) -> np.ndarray:
    """Re-center (in-place) the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    leaves_batch_trans = np.zeros_like(leaves_batch)
    leaves_batch_trans[:, :-1] = transforms.shift_taylor_full(
        leaves_batch[:, :-1],
        delta_t,
        use_conservative_tile,
    )
    leaves_batch_trans[:, -1] = leaves_batch[:, -1]
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_taylor_report_batch(leaves_batch: np.ndarray) -> np.ndarray:
    param_sets_batch = leaves_batch.copy()
    param_sets_vals = leaves_batch[:, :-3, 0]
    param_sets_sigs = leaves_batch[:, :-3, 1]
    v_final = leaves_batch[:, -3, 0]
    dv_final = leaves_batch[:, -3, 1]
    f0_batch = leaves_batch[:, -1, 0]
    s_factor = 1 - v_final / C_VAL
    # Gauge transform + error propagation
    param_sets_batch[:, :-3, 0] = param_sets_vals / s_factor[:, None]
    param_sets_batch[:, :-3, 1] = np.sqrt(
        (param_sets_sigs / s_factor[:, None]) ** 2
        + ((param_sets_vals / (C_VAL * s_factor[:, None] ** 2)) ** 2)
        * (dv_final[:, None] ** 2),
    )
    param_sets_batch[:, -3, 0] = f0_batch * s_factor
    param_sets_batch[:, -3, 1] = f0_batch * dv_final / C_VAL
    return param_sets_batch


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
    use_conservative_tile: bool,
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
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        leaves_arr, _ = poly_taylor_branch_batch(
            leaf,
            coord_cur,
            nbins,
            eta,
            poly_order,
            param_limits,
            branch_max,
        )
        branching_pattern[prune_level - 1] = len(leaves_arr)
        if use_moving_grid:
            leaves_arr = poly_taylor_transform_batch(
                leaves_arr,
                coord_next,
                coord_cur,
                use_conservative_tile,
            )
        leaf = leaves_arr[0:1]  # shape: (1, total_size)
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
    use_conservative_tile: bool,
    use_cheby_coarsening: bool = True,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    n_params = len(dparams_lim)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    weights = np.ones(n_freqs, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n_freqs, n_params), dtype=np.float64)
    dparam_cur_next = np.empty((n_freqs, n_params), dtype=np.float64)
    dparam_d_vec = np.empty((n_freqs, n_params + 1), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i, :n_params] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, n_params - 1] *= C_VAL / f0_batch

    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        _, t_obs_minus_t_ref = coord_cur

        dparam_new_batch_actual = psr_utils.poly_taylor_step_d_vec(
            n_params,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f0_batch,
            t_ref=0,
            use_cheby=use_cheby_coarsening,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch_actual,
            t_obs_minus_t_ref,
            nbins,
            f0_batch,
            t_ref=0,
            use_cheby=use_cheby_coarsening,
        )
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
        n_branches = np.ones(n_freqs, dtype=np.int64)

        for i in range(n_freqs):
            for j in range(n_params):  # skip d0
                if shift_bins_batch[i, j] < (eta - FLOAT_EPSILON):
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = dparam_cur_batch[i, j] / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - FLOAT_EPSILON)))
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points
        # Compute average branching factor
        children = np.sum(weights * n_branches)
        parents = np.sum(weights)
        branching_pattern[prune_level - 1] = children / parents
        # Update weights and dparams
        weights *= n_branches

        if use_moving_grid:
            # Transform dparams to the next segment
            delta_t = coord_next[0] - coord_cur[0]
            dparam_d_vec[:, :-1] = dparam_cur_next
            dparam_d_vec_new = transforms.shift_taylor_errors(
                dparam_d_vec,
                delta_t,
                use_conservative_tile,
            )
            dparam_cur_batch = dparam_d_vec_new[:, :-1]
        else:
            dparam_cur_batch = dparam_cur_next
    return branching_pattern
