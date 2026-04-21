from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.core.common import get_leaves
from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL, FLOAT_EPSILON
from pyloki.utils.snail import MiddleOutScheme


@njit(cache=True, fastmath=True)
def poly_chebyshev_seed(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the seed leaves for Chebyshev polynomial search.

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
    leaf[:-1, 0] -> Chebyshev polynomial coefficients,
                    order is [alpha_k, ..., alpha_1, alpha_0]
    leaf[:-1, 1] -> Grid size (error) on each coefficient,
    leaf[-1, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf[-1, 1]  -> Flag to indicate basis change (0: Polynomial, 1: Physical)
    """
    _, scale_init = coord_init
    leaves_taylor = get_leaves(param_arr, dparams)
    f0_batch = leaves_taylor[:, -1, 0]
    df_batch = leaves_taylor[:, -1, 1]
    leaves_d = np.zeros((len(leaves_taylor), poly_order + 1, 2), dtype=np.float64)
    # Copy till accel
    leaves_d[:, :-2] = leaves_taylor[:, :-1]
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    leaves_d[:, -2, 0] = 0
    leaves_d[:, -2, 1] = df_batch * (C_VAL / f0_batch)
    # intialize d0 (measure from t=t_init)
    leaves_d[:, -1, 0] = 0  # we never branch on d0

    leaves = np.zeros((len(leaves_taylor), poly_order + 2, 2), dtype=np.float64)
    leaves[:, :-1] = transforms.taylor_to_cheby_full(leaves_d, scale_init)
    # Store f0
    leaves[:, -1, 0] = f0_batch
    leaves[:, -1, 1] = 0  # Polynomial basis
    return leaves


@njit(cache=True, fastmath=True)
def poly_chebyshev_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    nbins: int,
    eta: float,
    poly_order: int,
    param_limits: np.ndarray,
    branch_max: int,
    use_conservative_tile: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Parameter set (leaf) to branch. Shape is (n_leaves, poly_order + 2, 2).
    coord_cur : tuple[float, float]
        Coordinates for the accumulated segment in the current stage.
    coord_prev : tuple[float, float]
        Coordinates for the accumulated segment at the end of the previous stage.
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
    use_conservative_tile : bool
        Whether to use the conservative tile for the domain expansion.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets. Shape is (n_branch, poly_order + 2, 2).
    """
    n_batch, _, _ = leaves_batch.shape
    _, scale_cur = coord_cur
    n_params = poly_order
    f0_batch = leaves_batch[:, -1, 0]
    basis_flag_batch = leaves_batch[:, -1, 1]

    # Transform the parameters to coord_cur domain
    param_set_trans_batch = transforms.shift_cheby_full(
        leaves_batch[:, :-1],
        coord_cur,
        coord_prev,
        use_conservative_tile,
    )
    param_cur_batch = param_set_trans_batch[:, :-1, 0]
    dparam_cur_batch = param_set_trans_batch[:, :-1, 1]
    alpha0_cur_batch = param_set_trans_batch[:, -1, 0]

    dparam_new_batch_actual = psr_utils.poly_cheb_step_vec(
        n_params,
        nbins,
        eta,
        f0_batch,
    )
    shift_bins_batch = psr_utils.poly_cheb_shift_vec(
        dparam_cur_batch,
        dparam_new_batch_actual,
        nbins,
        f0_batch,
    )
    dparam_new_batch = psr_utils.poly_cheb_step_vec_limited(
        n_params,
        scale_cur,
        nbins,
        eta,
        f0_batch,
        param_limits,
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
    leaves_branch_batch[:, -2, 0] = alpha0_cur_batch[batch_origins]
    leaves_branch_batch[:, -1, 0] = f0_batch[batch_origins]
    leaves_branch_batch[:, -1, 1] = basis_flag_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_chebyshev_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the leaf parameters to find the closest grid index and phase shift.

    Parameters
    ----------
    leaves_batch : np.ndarray
        The leaf parameter set. Shape is (n_batch, poly_order + 2, 2).
    coord_add : tuple[float, float]
        The coordinates of the added segment (level cur).
    coord_cur : tuple[float, float]
        The coordinates of the current segment (level cur).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_grid_count_init : np.ndarray
        Number of points in the initial (FFA) grid for the ``coord_add`` segment
        Currently this is simply [n_accel, n_freq].
    param_limits : np.ndarray
        Parameter limits (min, max).
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    Notes
    -----
    leaves_batch is referenced to coord_cur, so we need to shift it to coord_add to get
    the resolved parameters index and relative phase shift. We also need to correct for
    the tree phase offset from coord_init to coord_cur.

    relative_phase_batch is complete phase shift with fractional part.

    """
    t0_cur, scale_cur = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    dvec_t_add = transforms.cheby_to_taylor_param_shift(
        param_vec_batch,
        t0_cur,
        scale_cur,
        t0_add,
    )
    dvec_t_init = transforms.cheby_to_taylor_param_shift(
        param_vec_batch,
        t0_cur,
        scale_cur,
        t0_init,
    )
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
def poly_chebyshev_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur_fixed: tuple[float, float],
    coord_init: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    _, scale_cur_fixed = coord_cur_fixed
    t0_init, _ = coord_init
    t0_add, _ = coord_add

    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    dvec_t_add = transforms.cheby_to_taylor_param_shift(
        param_vec_batch,
        t0_init,
        scale_cur_fixed,
        t0_add,
    )
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
def poly_chebyshev_ascend_resolve_batch(
    leaves_batch: np.ndarray,
    coord_segments: np.ndarray,
    coord_cur: tuple[float, float],
    param_grid_count_init: np.ndarray,
    param_limits: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    t0_cur, scale_cur = coord_cur
    n_leaves = len(leaves_batch)
    nsegments = len(coord_segments)
    n_params = param_limits.shape[0]

    param_vec_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    param_idx_batch_arr = np.empty((n_leaves, nsegments, n_params), dtype=np.int64)
    relative_phase_batch_arr = np.empty((n_leaves, nsegments), dtype=np.float64)

    for isegment in range(nsegments):
        t0_seg, _ = coord_segments[isegment]
        dvec_t_seg = transforms.cheby_to_taylor_param_shift(
            param_vec_batch,
            t0_cur,
            scale_cur,
            t0_seg,
        )
        accel_new_batch = dvec_t_seg[:, -3]
        freq_new_batch = f0_batch * (1 - dvec_t_seg[:, -2] / C_VAL)
        delay_batch = dvec_t_seg[:, -1] / C_VAL
        relative_phase_batch = psr_utils.get_phase_idx(
            t0_seg - t0_cur,
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
        param_idx_batch_arr[:, isegment, :] = param_idx_batch
        relative_phase_batch_arr[:, isegment] = relative_phase_batch
    return param_idx_batch_arr, relative_phase_batch_arr


@njit(cache=True, fastmath=True)
def poly_chebyshev_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    leaves_batch_trans = np.zeros_like(leaves_batch)
    leaves_batch_trans[:, :-1] = transforms.shift_cheby_full(
        leaves_batch[:, :-1],
        coord_next,
        coord_cur,
        use_conservative_tile,
    )
    leaves_batch_trans[:, -1] = leaves_batch[:, -1]
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_chebyshev_report_batch(
    leaves_batch: np.ndarray,
    coord_report: tuple[float, float],
) -> np.ndarray:
    _, scale = coord_report
    param_sets_batch = leaves_batch.copy()
    cheby_coeffs_batch = leaves_batch[:, :-1, :]
    f0_batch = leaves_batch[:, -1, 0]

    param_sets_batch_d = transforms.cheby_to_taylor_full(cheby_coeffs_batch, scale)
    param_sets_vals = param_sets_batch_d[:, :-2, 0]
    param_sets_sigs = param_sets_batch_d[:, :-2, 1]
    param_sets_batch[:, -2, 0] = param_sets_batch_d[:, -1, 0]
    param_sets_batch[:, -2, 1] = param_sets_batch_d[:, -1, 1]
    v_final = param_sets_batch_d[:, -2, 0]
    dv_final = param_sets_batch_d[:, -2, 1]
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
def generate_bp_poly_chebyshev_approx(
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
    """Generate the approximate branching pattern for the Chebyshev pruning search."""
    poly_order = len(dparams_lim)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    coord_init = snail_scheme.get_coord(0)
    leaves_init = poly_chebyshev_seed(param_arr, dparams_lim, poly_order, coord_init)
    leaf = leaves_init[itree : itree + 1]  # shape: (1, total_size)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        coord_prev = snail_scheme.get_previous_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        leaves_arr, _ = poly_chebyshev_branch_batch(
            leaf,
            coord_cur,
            coord_prev,
            nbins,
            eta,
            poly_order,
            param_limits,
            branch_max,
            use_conservative_tile,
        )
        branching_pattern[prune_level - 1] = len(leaves_arr)
        if use_moving_grid:
            leaves_arr = poly_chebyshev_transform_batch(
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
def generate_bp_poly_chebyshev(
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
    use_cheby_coarsening: bool = True,  # noqa: ARG001
) -> np.ndarray:
    """Generate the exact branching pattern for the Chebyshev pruning search."""
    n_params = len(dparams_lim)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa, stride=1)
    coord_init = snail_scheme.get_coord(0)
    _, scale_init = coord_init
    weights = np.ones(n_freqs, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparams_d_full = np.zeros((n_freqs, n_params + 1), dtype=np.float64)
    for i in range(n_freqs):
        dparams_d_full[i, :n_params] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparams_d_full[:, n_params - 1] *= C_VAL / f0_batch
    dparam_cur_batch = transforms.taylor_to_cheby_errors(dparams_d_full, scale_init)
    dparam_cur_next = np.empty((n_freqs, n_params + 1), dtype=np.float64)

    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        coord_prev = snail_scheme.get_previous_coord(
            prune_level,
            moving_grid=use_moving_grid,
        )
        _, scale_cur = coord_cur
        # Transform dparams to the current segment
        dparam_cur_batch = transforms.shift_cheby_errors(
            dparam_cur_batch,
            coord_cur,
            coord_prev,
            use_conservative_tile,
        )
        dparam_new_batch_actual = psr_utils.poly_cheb_step_vec(
            n_params + 1,
            nbins,
            eta,
            f0_batch,
        )
        shift_bins_batch = psr_utils.poly_cheb_shift_vec(
            dparam_cur_batch,
            dparam_new_batch_actual,
            nbins,
            f0_batch,
        )
        dparam_new_batch = psr_utils.poly_cheb_step_vec_limited(
            n_params + 1,
            scale_cur,
            nbins,
            eta,
            f0_batch,
            param_limits,
        )

        n_branches = np.ones(n_freqs, dtype=np.int64)

        for i in range(n_freqs):
            for j in range(n_params):
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
            dparam_cur_next = transforms.shift_cheby_errors(
                dparam_cur_next,
                coord_next,
                coord_cur,
                use_conservative_tile,
            )
        dparam_cur_batch = dparam_cur_next
    return branching_pattern
