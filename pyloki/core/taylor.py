from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core.common import get_leaves, propagate_bp_state_deterministic
from pyloki.detection.scoring import snr_score_func, snr_score_func_complex
from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.snail import MiddleOutScheme
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
    nbins: int,
    eta: float,
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
    nbins : int
        Number of bins in the folded profile.
    eta : float
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
        nbins,
        eta,
        f0,
        t_ref=0,
    )
    shift_bins = psr_utils.poly_taylor_shift_d(
        dparam_cur,
        dparam_new,
        t_obs_minus_t_ref,
        nbins,
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
        if shift_bins[i] >= (eta - eps):
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
    nbins: int,
    eta: float,
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
    needs_branching = shift_bins_batch >= (eta - eps)
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
    leaves_branch_batch[:, -1, 1] = basis_flag_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
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
    param_arr : types.ListType[types.Array]
        Parameter grid array for the ``coord_add`` segment (dim: 2)
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
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
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
def poly_taylor_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
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
    use_conservative_tile: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
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
def generate_bp_poly_taylor_approx(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    isuggest: int = 0,
    use_conservative_tile: bool = False,
) -> np.ndarray:
    """Generate the approximate branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa)
    coord_init = snail_scheme.get_coord(0)
    leaf = poly_taylor_leaves(param_arr, dparams_lim, poly_order, coord_init)[isuggest]
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(prune_level)
        leaves_arr = poly_taylor_branch(
            leaf,
            coord_cur,
            nbins,
            eta,
            poly_order,
            param_limits,
        )
        branching_pattern[prune_level - 1] = len(leaves_arr)
        leaves_arr_trans = poly_taylor_transform_batch(
            leaves_arr,
            coord_next,
            coord_cur,
            use_conservative_tile,
        )
        leaf = leaves_arr_trans[0]
    return np.array(branching_pattern)


@njit(cache=True, fastmath=True)
def generate_bp_poly_taylor(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    use_conservative_tile: bool = False,
    max_track_size: int = 10000,
) -> np.ndarray:
    """Deterministic branching pattern estimator with fixed memory cap."""
    poly_order = len(dparams_lim)
    f0_batch = param_arr[-1]
    n0 = len(f0_batch)
    # Ensure we don't downsample below the initial size
    max_track_size = max(max_track_size, n0)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa)

    # Track velocity
    d1_batch = np.zeros(n0, dtype=np.float64)
    weights = np.ones(n0, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    # Precompute constant parameter ranges for non-velocity parameters
    param_ranges_const = np.empty(poly_order - 1, dtype=np.float64)
    for i in range(poly_order - 1):
        param_ranges_const[i] = (param_limits[i][1] - param_limits[i][0]) / 2

    # Frequency limits for velocity conversion
    f_min, f_max = param_limits[poly_order - 1]

    eps = 1e-12
    for prune_level in range(1, nsegments):
        coord_next = snail_scheme.get_coord(prune_level)
        coord_cur = snail_scheme.get_current_coord(prune_level)
        _, t_obs_minus_t_ref = coord_cur
        f_cur_batch = f0_batch * (1.0 - d1_batch / C_VAL)
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f_cur_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            nbins,
            f_cur_batch,
            t_ref=0,
        )

        nfreqs = len(f0_batch)
        dparam_next_tmp = np.empty((nfreqs, poly_order), dtype=np.float64)
        n_branch_d1 = np.ones(nfreqs, dtype=np.int64)
        n_branch_other = np.ones(nfreqs, dtype=np.int64)

        v_ranges = (f_max - f_min) / f0_batch * C_VAL / 2
        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - eps)

        for i in range(nfreqs):
            for j in range(poly_order):
                # Bounds check
                limit = v_ranges[i] if j == poly_order - 1 else param_ranges_const[j]
                too_large_step = dparam_new_batch[i, j] > (limit + eps)
                if not needs_branching[i, j] or too_large_step:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                if j == poly_order - 1:  # Velocity dimension
                    n_branch_d1[i] = num_points
                else:
                    n_branch_other[i] *= num_points
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

        (f0_batch, d1_batch, weights, dparam_cur_batch_raw, avg_bp) = (
            propagate_bp_state_deterministic(
                f0_batch,
                d1_batch,
                weights,
                dparam_cur_batch,
                dparam_next_tmp,
                n_branch_d1,
                n_branch_other,
                max_track_size,
            )
        )
        branching_pattern[prune_level - 1] = avg_bp

        # Transform dparams to the next segment
        delta_t = coord_next[0] - coord_cur[0]
        dparam_d_vec = np.zeros((len(f0_batch), poly_order + 1), dtype=np.float64)
        dparam_d_vec[:, :-1] = dparam_cur_batch_raw
        dparam_d_vec_new = transforms.shift_taylor_errors(
            dparam_d_vec,
            delta_t,
            use_conservative_tile,
        )
        dparam_cur_batch = dparam_d_vec_new[:, :-1]

    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_bp_poly_taylor_fixed(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    nbins: int,
    eta: float,
    ref_seg: int,
    max_track_size: int = 10000,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor fixed pruning search."""
    poly_order = len(dparams_lim)
    f0_batch = param_arr[-1]
    n0 = len(f0_batch)
    # Ensure we don't downsample below the initial size
    max_track_size = max(max_track_size, n0)
    snail_scheme = MiddleOutScheme(nsegments, ref_seg, tseg_ffa)

    # Track velocity
    d1_batch = np.zeros(n0, dtype=np.float64)
    weights = np.ones(n0, dtype=np.float64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)

    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams_lim
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] * (C_VAL / f0_batch)

    # Precompute constant parameter ranges for non-velocity parameters
    param_ranges_const = np.empty(poly_order - 1, dtype=np.float64)
    for i in range(poly_order - 1):
        param_ranges_const[i] = (param_limits[i][1] - param_limits[i][0]) / 2

    # Frequency limits for velocity conversion
    f_min, f_max = param_limits[poly_order - 1]

    eps = 1e-12
    for prune_level in range(1, nsegments):
        coord_cur_fixed = snail_scheme.get_current_coord_fixed(prune_level)
        _, t_obs_minus_t_ref = coord_cur_fixed
        f_cur_batch = f0_batch * (1.0 - d1_batch / C_VAL)
        dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
            poly_order,
            t_obs_minus_t_ref,
            nbins,
            eta,
            f_cur_batch,
            t_ref=0,
        )
        shift_bins_batch = psr_utils.poly_taylor_shift_d_vec(
            dparam_cur_batch,
            dparam_new_batch,
            t_obs_minus_t_ref,
            nbins,
            f_cur_batch,
            t_ref=0,
        )

        nfreqs = len(f0_batch)
        dparam_next_tmp = np.empty((nfreqs, poly_order), dtype=np.float64)
        n_branch_d1 = np.ones(nfreqs, dtype=np.int64)
        n_branch_other = np.ones(nfreqs, dtype=np.int64)

        v_ranges = (f_max - f_min) / f0_batch * C_VAL / 2
        # Vectorized branching decision
        needs_branching = shift_bins_batch >= (eta - eps)

        for i in range(nfreqs):
            for j in range(poly_order):
                # Bounds check
                limit = v_ranges[i] if j == poly_order - 1 else param_ranges_const[j]
                too_large_step = dparam_new_batch[i, j] > (limit + eps)
                if not needs_branching[i, j] or too_large_step:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                if j == poly_order - 1:  # Velocity dimension
                    n_branch_d1[i] = num_points
                else:
                    n_branch_other[i] *= num_points
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

        (f0_batch, d1_batch, weights, dparam_cur_batch, avg_bp) = (
            propagate_bp_state_deterministic(
                f0_batch,
                d1_batch,
                weights,
                dparam_cur_batch,
                dparam_next_tmp,
                n_branch_d1,
                n_branch_other,
                max_track_size,
            )
        )
        # Compute average branching factor
        branching_pattern[prune_level - 1] = avg_bp

    return branching_pattern
