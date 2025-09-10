from __future__ import annotations

import numpy as np
from numba import njit, types

from pyloki.core.common import get_leaves
from pyloki.detection.scoring import snr_score_func, snr_score_func_complex
from pyloki.utils import maths, np_utils, psr_utils, transforms
from pyloki.utils.misc import C_VAL
from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@njit(cache=True, fastmath=True)
def poly_chebyshev_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the leaf parameter sets for Chebyshev polynomial search.

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
    leaf[:-1, 0] -> Chebyshev polynomial coefficients,
                    order is [alpha_poly_order, ..., alpha_1, alpha_0]
    leaf[:-1, 1] -> Grid size (error) on each coefficient,
    leaf[-1, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf[-1, 1]  -> Flag to indicate basis change (placeholder for now)
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
    return leaves


@njit(cache=True, fastmath=True)
def poly_chebyshev_suggest(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStruct:
    """Generate a Chebyshev suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_freq, 2, n_bins). Parameter dimensions are first two.
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
    param_arr : types.ListType
        Parameter values for each dimension (accel, freq).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Chebyshev polynomial.
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
    param_sets = poly_chebyshev_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks, "chebyshev")


@njit(cache=True, fastmath=True)
def poly_chebyshev_suggest_complex(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_widths: np.ndarray,
) -> SuggestionStructComplex:
    """Generate a Chebyshev suggestion struct from a fold segment in Fourier domain.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_freq, 2, n_bins_f). Parameter dimensions are first two.
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
    param_arr : types.ListType
        Parameter values for each dimension (accel, freq).
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Chebyshev polynomial.
    score_widths : np.ndarray
        Boxcar widths for the score computation.

    Returns
    -------
    SuggestionStructComplex
        Suggestion struct in Fourier domain
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_chebyshev_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 2), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks, "chebyshev")


@njit(cache=True, fastmath=True)
def taylor_to_chebyshev_limits(taylor_limits: np.ndarray, ts: float) -> np.ndarray:
    """Convert box limits on Taylor coefficients to limits on Chebyshev coefficients.

    Parameters
    ----------
    taylor_limits : np.ndarray
        Input limits for Taylor coefficients [min, max]. Shape (N, nparams, 2).
        Order is [d_kmax, ..., d_1].
    ts : float
        Scale factor (half-span) of the domain.

    Returns
    -------
    np.ndarray
        New bounding box for Chebyshev coefficients [alpha_min, alpha_max].
        Shape (N, nparams, 2).
        Order is [alpha_kmax, ..., alpha_1].
    """
    n_batch, n_params, _ = taylor_limits.shape
    n_params_d = n_params + 1
    d_limits_scaled_min = np.zeros((n_batch, n_params_d), dtype=np.float64)
    d_limits_scaled_max = np.zeros((n_batch, n_params_d), dtype=np.float64)

    k_range = np.arange(n_params_d, dtype=np.int64)
    scale_factor = (ts**k_range) / maths.fact(k_range)
    d_limits_scaled_min[:, 1:] = taylor_limits[:, ::-1, 0] * scale_factor[1:]
    d_limits_scaled_max[:, 1:] = taylor_limits[:, ::-1, 1] * scale_factor[1:]
    s_mat = maths.compute_connection_matrix_s(n_params)
    cheby_limits_min = np.zeros((n_batch, n_params), dtype=np.float64)
    cheby_limits_max = np.zeros((n_batch, n_params), dtype=np.float64)

    for k in range(1, n_params_d):
        for m in range(k, n_params_d):
            s_mk = s_mat[m, k]
            term_min = s_mk * d_limits_scaled_min[:, m]
            term_max = s_mk * d_limits_scaled_max[:, m]
            if s_mk > 0:
                cheby_limits_min[:, k - 1] += term_min
                cheby_limits_max[:, k - 1] += term_max
            else:  # s_mk < 0, so min/max contributions flip
                cheby_limits_min[:, k - 1] += term_max
                cheby_limits_max[:, k - 1] += term_min

    param_limits_cheby = np.zeros((n_batch, n_params, 2), dtype=np.float64)
    param_limits_cheby[:, :, 0] = cheby_limits_min[:, ::-1]
    param_limits_cheby[:, :, 1] = cheby_limits_max[:, ::-1]
    return param_limits_cheby


@njit(cache=True, fastmath=True)
def poly_chebyshev_branch_batch(
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
    conservative_errors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    leaves_batch : np.ndarray
        Parameter set (leaf) to branch. Shape: (n_params + 2, 2).
    coord_cur : tuple[float, float]
        Coordinates for the accumulated segment in the current stage.
    coord_prev : tuple[float, float]
        Coordinates for the accumulated segment at the end of the previous stage.
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
    n_batch, _, _ = leaves_batch.shape
    _, scale_cur = coord_cur
    _, _ = coord_prev
    f0_batch = leaves_batch[:, -1, 0]

    # Transform the parameters to coord_cur domain
    param_set_trans_batch = transforms.shift_cheby_full(
        leaves_batch[:, :-1],
        coord_cur,
        coord_prev,
        conservative_errors,
    )
    cheb_cur_batch = param_set_trans_batch[:, :-1, 0]
    dcheb_cur_batch = param_set_trans_batch[:, :-1, 1]
    alpha0_cur_batch = param_set_trans_batch[:, -1, 0]

    param_limits_d = np.empty((n_batch, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL
    param_limits_cheby = taylor_to_chebyshev_limits(param_limits_d, scale_cur)

    dcheb_new_batch = psr_utils.poly_cheb_step_vec(
        poly_order,
        fold_bins,
        tol_bins,
        f0_batch,
    )
    shift_bins_batch = psr_utils.poly_cheb_shift_vec(
        dcheb_cur_batch,
        dcheb_new_batch,
        fold_bins,
        f0_batch,
    )
    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, poly_order, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, poly_order), dtype=np.float64)
    branched_counts = np.empty((n_batch, poly_order), dtype=np.int64)
    for i in range(n_batch):
        for j in range(poly_order):
            param_min, param_max = param_limits_cheby[i, j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                cheb_cur_batch[i, j],
                dcheb_cur_batch[i, j],
                dcheb_new_batch[i, j],
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
                pad_branched_params[i, j, 0] = cheb_cur_batch[i, j]
                pad_branched_dparams[i, j] = dcheb_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- Optimized Padded Cartesian Product ---
    leaves_branch_cheb_batch, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        poly_order,
    )
    total_leaves = len(batch_origins)
    leaves_branch_batch = np.zeros((total_leaves, poly_order + 2, 2), dtype=np.float64)
    leaves_branch_batch[:, :-2, 0] = leaves_branch_cheb_batch
    leaves_branch_batch[:, :-2, 1] = pad_branched_dparams[batch_origins]
    leaves_branch_batch[:, -2, 0] = alpha0_cur_batch[batch_origins]
    leaves_branch_batch[:, -1, 0] = f0_batch[batch_origins]
    return leaves_branch_batch, batch_origins


@njit(cache=True, fastmath=True)
def poly_chebyshev_validate_batch(
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
def get_circular_mask(
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
def poly_chebyshev_resolve_batch(
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
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
    param_arr : types.ListType
        Parameter grid array for the ``coord_add`` segment (dim: 2)
    fold_bins : int
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
    n_batch, _, _ = leaves_batch.shape
    t0_cur, scale_cur = coord_cur
    t0_init, _ = coord_init
    t0_add, _ = coord_add
    param_set_batch = leaves_batch[:, :-1, 0]
    f0_batch = leaves_batch[:, -1, 0]

    dvec_t_add = transforms.cheby_to_taylor_param_shift(
        param_set_batch,
        t0_cur,
        scale_cur,
        t0_add,
    )
    dvec_t_init = transforms.cheby_to_taylor_param_shift(
        param_set_batch,
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

    circ_mask = get_circular_mask(leaves_batch, scale_cur, snap_threshold=5)
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
def poly_chebyshev_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    leaves_batch_trans = np.zeros_like(leaves_batch)
    leaves_batch_trans[:, :-1] = transforms.shift_cheby_full(
        leaves_batch[:, :-1],
        coord_next,
        coord_cur,
        conservative_errors,
    )
    leaves_batch_trans[:, -1] = leaves_batch[:, -1]
    return leaves_batch_trans


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
    circ_mask = get_circular_mask(leaves_batch, scale_cur, snap_threshold=5)
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
def generate_bp_chebyshev_approx(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    isuggest: int = 0,
    use_conservative_errors: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the approximate branching pattern for the Taylor pruning search."""
    poly_order = len(dparams)
    branch_max = 256
    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nstages + 1) - ref_seg), kind="mergesort")
    coord_init = (ref_seg + 0.5) * tchunk_ffa, tchunk_ffa / 2
    leaf = poly_chebyshev_leaves(param_arr, dparams, poly_order, coord_init)[isuggest]
    branching_pattern = []
    for prune_level in range(1, nstages + 1):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tchunk_ffa, scale * tchunk_ffa
        scheme_till_now_prev = scheme_data[:prune_level]
        ref_prev = (np.min(scheme_till_now_prev) + np.max(scheme_till_now_prev) + 1) / 2
        scale_prev = ref_prev - np.min(scheme_till_now_prev)
        coord_prev = ref_prev * tchunk_ffa, scale_prev * tchunk_ffa
        coord_cur = coord_prev[0], coord_next[1]
        leaves_arr = poly_chebyshev_branch_batch(
            leaf[np.newaxis, :],
            coord_cur,
            fold_bins,
            tol_bins,
            poly_order,
            param_limits,
            branch_max,
            use_conservative_errors,
        )
        branching_pattern.append(len(leaves_arr))
        leaves_arr_trans = poly_chebyshev_transform_batch(
            leaves_arr,
            coord_next,
            coord_cur,
            use_conservative_errors,
        )
        leaf = leaves_arr_trans[0]
    return np.array(branching_pattern)


@njit(cache=True, fastmath=True)
def generate_bp_chebyshev(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    use_conservative_errors: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the exact branching pattern for the Taylor pruning search."""
    poly_order = len(dparams)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nstages + 1) - ref_seg), kind="mergesort")
    coord_init = (ref_seg + 0.5) * tchunk_ffa, tchunk_ffa / 2
    branching_pattern = np.empty(nstages, dtype=np.float64)
    # Keep poly_order+1 for d0
    dparam_cur_batch = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i, :-1] = dparams
    # f = f0(1 - v / C) => dv = -(C/f0) * df
    dparam_cur_batch[:, -2] = dparam_cur_batch[:, -2] * (C_VAL / f0_batch)
    dcheb_cur_batch = transforms.taylor_to_cheby_errors(
        dparam_cur_batch,
        coord_init[1],
    )

    param_limits_d = np.empty((n_freqs, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL

    for prune_level in range(1, nstages + 1):
        # Compute coordinates
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tchunk_ffa, scale * tchunk_ffa
        scheme_till_now_prev = scheme_data[:prune_level]
        ref_prev = (np.min(scheme_till_now_prev) + np.max(scheme_till_now_prev) + 1) / 2
        scale_prev = ref_prev - np.min(scheme_till_now_prev)
        coord_prev = ref_prev * tchunk_ffa, scale_prev * tchunk_ffa
        coord_cur = coord_prev[0], coord_next[1]

        # Transform dparams to the current segment
        dcheb_cur_batch = transforms.shift_cheby_errors(
            dcheb_cur_batch,
            coord_cur,
            coord_prev,
            use_conservative_errors,
        )

        param_limits_cheby_d0 = np.zeros((n_freqs, poly_order + 1, 2), dtype=np.float64)
        param_limits_cheby = taylor_to_chebyshev_limits(param_limits_d, coord_cur[1])
        param_limits_cheby_d0[:, :-1] = param_limits_cheby
        param_ranges = (
            param_limits_cheby_d0[:, :, 1] - param_limits_cheby_d0[:, :, 0]
        ) / 2
        dcheb_new_batch = psr_utils.poly_cheb_step_vec(
            poly_order + 1,
            fold_bins,
            tol_bins,
            f0_batch,
        )
        shift_bins_batch = psr_utils.poly_cheb_shift_vec(
            dcheb_cur_batch,
            dcheb_new_batch,
            fold_bins,
            f0_batch,
        )

        dcheb_cur_next = np.empty((n_freqs, poly_order + 1), dtype=np.float64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dcheb_new_batch > (param_ranges + eps)

        for i in range(n_freqs):
            # skip d0
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dcheb_cur_next[i, j] = dcheb_cur_batch[i, j]
                    continue
                num_points = max(
                    1,
                    int(np.ceil(dcheb_cur_batch[i, j] / dcheb_new_batch[i, j])),
                )
                n_branches[i] *= num_points
                dcheb_cur_next[i, j] = dcheb_cur_batch[i, j] / num_points
        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs

        # Transform dparams to the next segment
        dcheb_cur_batch = transforms.shift_cheby_errors(
            dcheb_cur_next,
            coord_next,
            coord_cur,
            use_conservative_errors,
        )
    return branching_pattern
