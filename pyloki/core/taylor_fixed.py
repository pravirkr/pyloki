from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.core.common import get_leaves
from pyloki.core.taylor import get_circular_mask
from pyloki.detection.scoring import snr_score_func, snr_score_func_complex
from pyloki.utils import np_utils, psr_utils, transforms
from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the leaf parameter sets for Fixed-Grid Taylor polynomial search.

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
    leaf[:-2, 0] -> Fixed-Grid Taylor polynomial coefficients,
                    order is [d_poly_order, ..., d_2, f_0]
    leaf[:-2, 1] -> Grid size (error) on each coefficient,
    leaf[-2, 0]  -> Frequency at t_init (f0), (no real use for now)
    leaf[-1, 0]  -> Reference time at t_init (t0_init)
    leaf[-1, 1]  -> Scale at t_init (scale_init)
    """
    t0_init, scale_init = coord_init
    leaves_taylor = get_leaves(param_arr, dparams)
    f0_batch = leaves_taylor[:, -1, 0]
    leaves = np.zeros((len(leaves_taylor), poly_order + 2, 2), dtype=np.float64)
    # Copy as it is
    leaves[:, :-2] = leaves_taylor
    leaves[:, -2, 0] = f0_batch
    leaves[:, -1, 0] = t0_init
    leaves[:, -1, 1] = scale_init
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_suggest(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStruct:
    """Generate a Fixed-Grid Taylor suggestion struct from a fold segment.

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
    param_sets = poly_taylor_fixed_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks, "taylor_fixed")


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_suggest_complex(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStructComplex:
    """Generate a Fixed-Grid Taylor suggestion struct in Fourier domain.

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
    param_sets = poly_taylor_fixed_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks, "taylor_fixed")


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur_fixed: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch, _, _ = leaves_batch.shape
    _, t_obs_minus_t_ref = coord_cur_fixed
    param_cur_batch = leaves_batch[:, :-2, 0]
    dparam_cur_batch = leaves_batch[:, :-2, 1]
    t0_init_batch = leaves_batch[:, -1, 0]
    scale_init_batch = leaves_batch[:, -1, 1]
    fcur_batch = leaves_batch[:, -3, 0]
    f0_batch = leaves_batch[:, -2, 0]

    dparam_new_batch = psr_utils.poly_taylor_step_d_f_vec(
        poly_order,
        t_obs_minus_t_ref,
        fold_bins,
        tol_bins,
        fcur_batch,
        t_ref=0,
    )
    shift_bins_batch = psr_utils.poly_taylor_shift_d_f_vec(
        dparam_cur_batch,
        dparam_new_batch,
        t_obs_minus_t_ref,
        fold_bins,
        fcur_batch,
        t_ref=0,
    )
    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, poly_order, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, poly_order), dtype=np.float64)
    branched_counts = np.empty((n_batch, poly_order), dtype=np.int64)
    for i in range(n_batch):
        for j in range(poly_order):
            param_min, param_max = param_limits[j]
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
    leaves_branch_batch[:, -2, 0] = f0_batch[batch_origins]
    leaves_branch_batch[:, -1, 0] = t0_init_batch[batch_origins]
    leaves_branch_batch[:, -1, 1] = scale_init_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_fixed_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
    _, _ = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    delta_t = t0_add - t0_init
    param_vec_batch = leaves_batch[:, :-2, 0]
    freq_cur_batch = leaves_batch[:, -3, 0]

    param_vec_new_batch, delay_batch = transforms.shift_taylor_params_d_f_batch(
        param_vec_batch,
        delta_t,
    )
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

    circ_mask = get_circular_mask(leaves_batch, snap_threshold=5)
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
def generate_bp_taylor_fixed(
    param_arr: types.ListType,
    dparams_lim: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tseg_ffa: float,
    nsegments: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    poly_order = len(dparams_lim)
    freq_arr = param_arr[-1]
    n0 = len(freq_arr)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nsegments) - ref_seg), kind="mergesort")
    weights = np.ones(n0, dtype=np.int64)
    branching_pattern = np.empty(nsegments - 1, dtype=np.float64)
    t0_init = (ref_seg + 0.5) * tseg_ffa
    scale_init = tseg_ffa / 2

    dparam_cur_batch = np.empty((n0, poly_order), dtype=np.float64)
    for i in range(n0):
        dparam_cur_batch[i] = dparams_lim
    param_ranges = np.array([(p_max - p_min) / 2 for p_min, p_max in param_limits])

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
        n_branch_nonfreq = np.ones(nfreq, dtype=np.int64)

        # Vectorized branching decision
        eps = 1e-12
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > (param_ranges + eps)

        weighted_sum = 0.0
        total_weight = 0.0
        total_freq_branches = 0

        for i in range(nfreq):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_next_tmp[i, j] = dparam_cur_batch[i, j]
                    continue
                ratio = (dparam_cur_batch[i, j] + eps) / dparam_new_batch[i, j]
                num_points = max(1, int(np.ceil(ratio - eps)))
                if j == poly_order - 1:
                    n_branch_freq[i] = num_points
                else:
                    n_branch_nonfreq[i] *= num_points
                dparam_next_tmp[i, j] = dparam_cur_batch[i, j] / num_points

            total_weight += weights[i]
            weighted_sum += weights[i] * (n_branch_nonfreq[i] * n_branch_freq[i])
            total_freq_branches += n_branch_freq[i]

        # Compute average branching factor
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
            else:
                dparam_cur_freq = dparam_cur_batch[i, poly_order - 1]
                delta = 0.25 * dparam_cur_freq
                f = freq_arr[i]
                # Create cfreq evenly spaced frequency points centered around f
                for k in range(cfreq):
                    offset = (k - (cfreq - 1) / 2) * delta
                    freq_arr_next[pos] = f + offset
                    weights_next[pos] = weight
                    dparam_cur_next[pos] = dparam_next_tmp[i]
                    pos += 1

        freq_arr = freq_arr_next
        dparam_cur_batch = dparam_cur_next
        weights = weights_next

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
