from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import maths, np_utils, psr_utils
from pyloki.utils.misc import C_VAL
from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@njit(cache=True, fastmath=True)
def ffa_taylor_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
    bseg_brute: int,
    fold_bins: int,
    tsamp: float,
) -> np.ndarray:
    """Initialize the fold for the FFA search.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series intensity (signal weighted by E/V).
    ts_v : np.ndarray
        Time series variance (E**2/V).
    param_arr : types.ListType[types.Array]
        Parameter grid array for each search parameter dimension.
    bseg_brute : int
        Brute force segment size in bins.
    fold_bins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the data.

    Returns
    -------
    np.ndarray
        Initial fold for the FFA search.
    """
    freq_arr = param_arr[-1]
    nparams = len(param_arr)
    t_ref = 0 if nparams == 1 else bseg_brute * tsamp / 2
    return common.brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        fold_bins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def ffa_taylor_init_complex(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
    bseg_brute: int,
    fold_bins: int,
    tsamp: float,
) -> np.ndarray:
    """Initialize the fold for the FFA search (complex-domain).

    Parameters
    ----------
    ts_e : np.ndarray
        Time series intensity (signal weighted by E/V).
    ts_v : np.ndarray
        Time series variance (E**2/V).
    param_arr : types.ListType[types.Array]
        Parameter grid array for each search parameter dimension.
    bseg_brute : int
        Brute force segment size in bins.
    fold_bins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the data.

    Returns
    -------
    np.ndarray
        Initial fold for the FFA search.
    """
    freq_arr = param_arr[-1]
    nparams = len(param_arr)
    t_ref = 0 if nparams == 1 else bseg_brute * tsamp / 2
    return common.brutefold_start_complex(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        fold_bins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def ffa_taylor_resolve(
    pset_cur: np.ndarray,
    param_arr: types.ListType[types.Array],
    ffa_level: int,
    latter: int,
    tseg_brute: float,
    fold_bins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the params to find the closest index in grid and absolute phase shift.

    Parameters
    ----------
    pset_cur : np.ndarray
        The current iter parameter set to resolve.
    param_arr : types.ListType[types.Array]
        Parameter grid array for the previous iteration (ffa_level - 1).
    ffa_level : int
        Current FFA level (same level as pset_cur).
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tseg_brute : float
        Duration of the brute force segment.
    fold_bins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, float]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    phase_shift is complete phase shift with fractional part.

    Notes
    -----
    delta_t is the time difference between the reference time of the current segment
    (0) and the reference time of the previous segment.
    """
    nparams = len(pset_cur)
    if nparams == 1:
        delta_t = latter * 2 ** (ffa_level - 1) * tseg_brute
        pset_new, delay = pset_cur, 0
    else:
        delta_t = (latter - 0.5) * 2 ** (ffa_level - 1) * tseg_brute
        pset_new, delay = psr_utils.shift_params_taylor_d_f(pset_cur, delta_t)
    relative_phase = psr_utils.get_phase_idx(delta_t, pset_cur[-1], fold_bins, delay)
    pindex_prev = np.zeros(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(param_arr[ip], pset_new[ip])
    return pindex_prev, relative_phase


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
        Order is [..., acc, freq].
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
        - coord_init[0] -> t0 (reference time) measured from t=0
        - coord_init[1] -> scale (half duration of the segment)
    coord_mid : tuple[float, float]
        The mid-point coordinate of the entire data.
        - coord_mid[0] -> t_mid (midpoint time) measured from t=0
        - coord_mid[1] -> scale (half duration of the entire data)

    Returns
    -------
    np.ndarray
        The leaf parameter sets. Shape is (n_param_sets, poly_order + 3, 2).

    Notes
    -----
    Conventions for each leaf parameter set:
    leaf[:-2, 0] -> Taylor polynomial coefficients, order is [..., d2, d1, d0]
    leaf[:-2, 1] -> Grid size (error) on each coefficient,
    leaf[-2, 0]  -> Frequency at t_init (f0), assuming f=f0 at t_init
    leaf[-1, 0]  -> Reference time from the data start (t_c),
    leaf[-1, 1]  -> Scaling, half duration of the segment (t_s).
    """
    t_init, scale = coord_init
    leaves_taylor = common.get_leaves(param_arr, dparams)
    leaves = np.zeros((len(leaves_taylor), poly_order + 3, 2), dtype=np.float64)
    leaves[:, :-4] = leaves_taylor[:, :-1]
    f0_batch = leaves_taylor[:, -1, 0]
    # f = f0(1 - v / C) => dv = -df / f0 * C
    leaves[:, -4, 0] = 0
    leaves[:, -4, 1] = leaves_taylor[:, -1, 1] / f0_batch * C_VAL
    # intialize d0 (measure from t=t_init)
    leaves[:, -3, 0] = 0  # we never branch on d0
    leaves[:, -2, 0] = f0_batch
    leaves[:, -1, 0] = t_init
    leaves[:, -1, 1] = scale
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
        scores[iparam] = scoring.snr_score_func(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 3), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks)


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
        scores[iparam] = scoring.snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 3), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def poly_taylor_branch(
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    param_set : np.ndarray
        Parameter set to branch. Shape: (nparams + 2, 2).
    coord_cur : tuple[float, float]
        The coordinates for the current segment.
    fold_bins : int
        Number of bins in the folded profile.
    tol_bins : float
        Tolerance for the parameter step size in bins.
    poly_order : int
        The order of the Taylor polynomial.
    param_limits : types.ListType[types.Tuple[float, float]]
        The limits for each parameter.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets (shape: (nbranch, nparams + 2, 2)).
    """
    _, t_obs_minus_t_ref = coord_cur
    nparams = poly_order
    param_cur = param_set[0:-2, 0]
    dparam_cur = param_set[0:-2, 1]
    f0 = param_set[-2, 0]
    t0, scale = param_set[-1]

    param_limits_d = np.empty((nparams, 2), dtype=np.float64)
    for i in range(nparams):
        param_limits_d[i, 0] = param_limits[i][0]
        param_limits_d[i, 1] = param_limits[i][1]
    param_limits_d[-1, 0] = (1 - param_limits[nparams - 1][1] / f0) * C_VAL
    param_limits_d[-1, 1] = (1 - param_limits[nparams - 1][0] / f0) * C_VAL

    dparam_new = psr_utils.poly_taylor_step_d(
        nparams,
        t_obs_minus_t_ref,
        fold_bins,
        tol_bins,
        f0,
        t_ref=0,
    )
    shift_bins = psr_utils.poly_taylor_shift_d(
        dparam_cur,
        dparam_new,
        t_obs_minus_t_ref,
        fold_bins,
        f0,
        t_ref=0,
    )

    eps = 1e-6
    total_size = 1
    leaf_params = typed.List.empty_list(types.float64[::1])
    leaf_dparams = np.empty(nparams, dtype=np.float64)
    shapes = np.empty(nparams, dtype=np.int64)
    for i in range(nparams):
        p_min, p_max = param_limits_d[i]
        if shift_bins[i] >= (tol_bins - eps):
            leaf_param_arr, dparam_act = psr_utils.branch_param(
                param_cur[i],
                dparam_cur[i],
                dparam_new[i],
                p_min,
                p_max,
            )
        else:
            leaf_param_arr = np.array([param_cur[i]], dtype=np.float64)
            dparam_act = dparam_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param_arr)
        shapes[i] = len(leaf_param_arr)
        total_size *= shapes[i]

    # Allocate and populate the final array directly
    leaves_taylor = np.empty((total_size, nparams, 2), dtype=np.float64)
    leaves_taylor[:, :, 1] = leaf_dparams
    # Fill column 0 (parameter values) using Cartesian product logic
    for i in range(total_size):
        idx = i
        # last dimension changes fastest
        for j in range(nparams - 1, -1, -1):
            arr = leaf_params[j]
            arr_idx = idx % shapes[j]
            leaves_taylor[i, j, 0] = arr[arr_idx]
            idx //= shapes[j]

    leaves = np.zeros((total_size, poly_order + 2, 2), dtype=np.float64)
    leaves[:, :-2] = leaves_taylor
    leaves[:, -2, 0] = f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_branch_batch(
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_set_batch)
    nparams = poly_order
    _, t_obs_minus_t_ref = coord_cur
    param_cur_batch = param_set_batch[:, :-3, 0]
    dparam_cur_batch = param_set_batch[:, :-3, 1]
    d0_batch = param_set_batch[:, -3, 0]
    f0_batch = param_set_batch[:, -2, 0]
    t0_batch = param_set_batch[:, -1, 0]
    scale_batch = param_set_batch[:, -1, 1]

    param_limits_d = np.empty((n_batch, nparams, 2), dtype=np.float64)
    for i in range(nparams):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[nparams - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[nparams - 1][0] / f0_batch) * C_VAL

    dparam_new_batch = psr_utils.poly_taylor_step_d_vec(
        nparams,
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
    # --- Vectorized Padded Branching ---
    pad_branched_params = np.empty((n_batch, nparams, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, nparams), dtype=np.float64)
    branched_counts = np.empty((n_batch, nparams), dtype=np.int64)
    for i in range(n_batch):
        for j in range(nparams):
            p_min, p_max = param_limits_d[i, j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                param_cur_batch[i, j],
                dparam_cur_batch[i, j],
                dparam_new_batch[i, j],
                p_min,
                p_max,
            )
            pad_branched_dparams[i, j] = dparam_act
            branched_counts[i, j] = count

    # --- Vectorized Selection ---
    # Select based on mask: shape (n_batch, nparams, 1)
    eps = 1e-6  # Small tolerance for floating-point comparison
    mask_2d = shift_bins_batch >= (tol_bins - eps)  # Shape (n_batch, nparams)
    for i in range(n_batch):
        for j in range(nparams):
            if not mask_2d[i, j]:
                pad_branched_params[i, j, :] = 0
                pad_branched_params[i, j, 0] = param_cur_batch[i, j]
                pad_branched_dparams[i, j] = dparam_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- Optimized Padded Cartesian Product ---
    batch_leaves_taylor, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        nparams,
    )
    total_leaves = len(batch_origins)
    batch_leaves = np.zeros((total_leaves, poly_order + 3, 2), dtype=np.float64)
    batch_leaves[:, :-3, 0] = batch_leaves_taylor
    batch_leaves[:, :-3, 1] = pad_branched_dparams[batch_origins]
    batch_leaves[:, -3, 0] = d0_batch[batch_origins]
    batch_leaves[:, -2, 0] = f0_batch[batch_origins]
    batch_leaves[:, -1, 0] = t0_batch[batch_origins]
    batch_leaves[:, -1, 1] = scale_batch[batch_origins]
    return batch_leaves, batch_origins


@njit(cache=True, fastmath=True)
def poly_taylor_validate_batch(
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    p_orb_min: float,
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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered leaf parameter sets (only physically plausible) and their origins.
    """
    eps = 1e-12
    snap = leaves_batch[:, 0, 0]
    dsnap = leaves_batch[:, 0, 1]
    accel = leaves_batch[:, 2, 0]
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
    snap_threshold = 5
    is_significant = np.abs(snap / (dsnap + eps)) > snap_threshold

    # No filtering for each leaf until its snap is significant
    mask = ~is_significant | (is_zero | (is_usable & is_within_omega_limit))
    idx = np.where(mask)[0]
    return leaves_batch[idx], leaves_origins[idx]


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
    param_arr: types.ListType[types.Array],
    fold_bins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the leaf params to find the closest index in grid and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set (shape: (nparams + 2, 2)).
    coord_add : tuple[float, float]
        The coordinates for the added segment (level current).
    coord_cur : tuple[float, float]
        The coordinates for the current pruning suggestion tree.
    coord_init : tuple[float, float]
        The coordinates for the initial pruning suggestion tree.
    param_arr : types.ListType[types.Array]
        Parameter grid array for the ``coord_add`` segment (dim: 2)
    fold_bins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, float]
        The resolved parameter index in the ``param_arr`` and the relative phase shift.

    Notes
    -----
    leaf is referenced to coord_cur, so we need to shift it to coord_add to get the
    resolved parameters index and relative phase shift. We also need to correct for the
    tree phase offset from t_init to t_cur.

    relative_phase is complete phase shift with fractional part.

    """
    t0, _ = coord_cur
    t_init, _ = coord_init
    t_add, _ = coord_add
    f0 = leaf[-2, 0]

    dvec_t_add = psr_utils.shift_params_taylor(leaf[:-2, 0], t_add - t0)
    dvec_t_init = psr_utils.shift_params_taylor(leaf[:-2, 0], t_init - t0)
    accel_new = dvec_t_add[-3]
    vel_new = dvec_t_add[-2] - dvec_t_init[-2]
    freq_new = f0 * (1 - vel_new / C_VAL)
    delay = (dvec_t_add[-1] - dvec_t_init[-1]) / C_VAL
    relative_phase = psr_utils.get_phase_idx(t_add - t_init, f0, fold_bins, delay)
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
    fold_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a batch of leaf params to find the closest grid index and phase shift."""
    n_batch, _, _ = leaves_batch.shape
    t0, _ = coord_cur
    t_init, _ = coord_init
    t_add, _ = coord_add
    f0_batch = leaves_batch[:, -2, 0]

    dvec_t_add = psr_utils.shift_params_taylor(leaves_batch[:, :-2, 0], t_add - t0)
    dvec_t_init = psr_utils.shift_params_taylor(leaves_batch[:, :-2, 0], t_init - t0)
    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2] - dvec_t_init[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = (dvec_t_add[:, -1] - dvec_t_init[:, -1]) / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t_add - t_init,
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
def get_circular_mask(
    leaves_batch: np.ndarray,
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
    snap = leaves_batch[:, 0, 0]
    dsnap = leaves_batch[:, 0, 1]
    accel = leaves_batch[:, 2, 0]

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
    t0, _ = coord_cur
    t_init, _ = coord_init
    t_add, _ = coord_add
    param_vec_batch = leaves_batch[:, :-2, 0]
    f0_batch = leaves_batch[:, -2, 0]
    circ_mask = get_circular_mask(leaves_batch, snap_threshold=5)
    idx_circular = np.where(circ_mask)[0]
    idx_taylor = np.where(~circ_mask)[0]
    dvec_t_add = np.empty((n_batch, 5), dtype=np.float64)
    dvec_t_init = np.empty((n_batch, 5), dtype=np.float64)

    if idx_circular.size > 0:
        dvec_t_add_circ = psr_utils.shift_params_circular_batch(
            param_vec_batch[idx_circular],
            t_add - t0,
        )
        dvec_t_init_circ = psr_utils.shift_params_circular_batch(
            param_vec_batch[idx_circular],
            t_init - t0,
        )
        dvec_t_add[idx_circular] = dvec_t_add_circ
        dvec_t_init[idx_circular] = dvec_t_init_circ

    if idx_taylor.size > 0:
        dvec_t_add_norm = psr_utils.shift_params_taylor(
            param_vec_batch[idx_taylor],
            t_add - t0,
        )
        dvec_t_init_norm = psr_utils.shift_params_taylor(
            param_vec_batch[idx_taylor],
            t_init - t0,
        )
        dvec_t_add[idx_taylor] = dvec_t_add_norm
        dvec_t_init[idx_taylor] = dvec_t_init_norm

    accel_new_batch = dvec_t_add[:, -3]
    vel_new_batch = dvec_t_add[:, -2] - dvec_t_init[:, -2]
    freq_new_batch = f0_batch * (1 - vel_new_batch / C_VAL)
    delay_batch = (dvec_t_add[:, -1] - dvec_t_init[:, -1]) / C_VAL
    relative_phase_batch = psr_utils.get_phase_idx(
        t_add - t_init,
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
def poly_taylor_transform_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    grid_conservative: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    leaves_batch_trans = np.zeros_like(leaves_batch)
    leaves_batch_trans[:, :-2] = psr_utils.shift_leaves_taylor_batch(
        leaves_batch[:, :-2],
        delta_t,
        grid_conservative,
    )
    leaves_batch_trans[:, -2] = leaves_batch[:, -2]
    leaves_batch_trans[:, -1] = leaves_batch[:, -1]
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def poly_taylor_transform_circular_batch(
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    grid_conservative: bool,
) -> np.ndarray:
    """Re-center the leaves to the next segment reference time."""
    delta_t = coord_next[0] - coord_cur[0]
    circ_mask = get_circular_mask(leaves_batch, snap_threshold=5)
    idx_circular = np.where(circ_mask)[0]
    idx_taylor = np.where(~circ_mask)[0]
    leaves_batch_trans = leaves_batch.copy()
    if idx_circular.size > 0:
        leaves_batch_trans[idx_circular, :-2] = psr_utils.shift_leaves_circular_batch(
            leaves_batch[idx_circular, :-2],
            delta_t,
            grid_conservative,
        )
    if idx_taylor.size > 0:
        leaves_batch_trans[idx_taylor, :-2] = psr_utils.shift_leaves_taylor_batch(
            leaves_batch[idx_taylor, :-2],
            delta_t,
            grid_conservative,
        )
    return leaves_batch_trans


@njit(cache=True, fastmath=True)
def generate_branching_pattern_approx(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    isuggest: int = 0,
) -> np.ndarray:
    """Generate the branching pattern for the pruning Taylor search.

    This is a simplified version of the branching pattern that only tracks the
    worst-case branching factor.

    Returns
    -------
    np.ndarray
        Branching pattern for the pruning Taylor search.
    """
    poly_order = len(dparams)
    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nstages + 1) - ref_seg), kind="mergesort")
    coord_init = (ref_seg + 0.5) * tchunk_ffa, tchunk_ffa / 2
    coord_mid = (tchunk_ffa * (nstages + 1) / 2, tchunk_ffa * (nstages + 1) / 2)
    leaf = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init, coord_mid)[
        isuggest
    ]
    branching_pattern = []
    for prune_level in range(1, nstages + 1):
        # Compute coord_cur
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tchunk_ffa, scale * tchunk_ffa
        scheme_till_now_cur = scheme_data[:prune_level]
        ref_cur = (np.min(scheme_till_now_cur) + np.max(scheme_till_now_cur) + 1) / 2
        coord_cur = ref_cur * tchunk_ffa, (scale + 0.5) * tchunk_ffa
        leaves_arr = poly_taylor_branch(
            leaf,
            coord_cur,
            fold_bins,
            tol_bins,
            poly_order,
            param_limits,
        )
        branching_pattern.append(len(leaves_arr))
        leaves_arr_trans = poly_taylor_transform_batch(
            leaves_arr,
            coord_next,
            coord_cur,
        )
        leaf = leaves_arr_trans[0]
    return np.array(branching_pattern)


@njit(cache=True, fastmath=True)
def generate_branching_pattern(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    grid_conservative: bool = False,  # noqa: FBT002
) -> np.ndarray:
    """Generate the branching pattern for the pruning Taylor search.

    This tracks the exact number of branches per node to compute the average
    branching factor.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension.
    dparams : np.ndarray
        Parameter step (grid) sizes for each dimension in a 1D array.
    param_limits : types.ListType[types.Tuple[float, float]]
        Parameter limits for each dimension.
    tchunk_ffa : float
        The duration of the starting segment.
    nstages : int
        The number of stages to generate the branching pattern for.
    fold_bins : int
        The number of bins in the frequency array.
    tol_bins : float
        The tolerance for the branching pattern.
    ref_seg : int
        The reference segment.
    grid_conservative : bool
        Whether to use a conservative grid.

    Returns
    -------
    np.ndarray
        The branching pattern for the pruning Taylor search.
    """
    poly_order = len(dparams)
    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nstages + 1) - ref_seg), kind="mergesort")

    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams
    # f = f0(1 - v / C) => dv = -df / f0 * C
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] / f0_batch * C_VAL

    branching_pattern = np.empty(nstages, dtype=np.float64)

    param_limits_d = np.empty((n_freqs, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL
    param_ranges = (param_limits_d[:, :, 1] - param_limits_d[:, :, 0]) / 2

    for prune_level in range(1, nstages + 1):
        # Compute coord_cur
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tchunk_ffa, scale * tchunk_ffa
        scheme_till_now_cur = scheme_data[:prune_level]
        ref_cur = (np.min(scheme_till_now_cur) + np.max(scheme_till_now_cur) + 1) / 2
        coord_cur = ref_cur * tchunk_ffa, (scale + 0.5) * tchunk_ffa
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
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > param_ranges

        for i in range(n_freqs):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                num_points = max(
                    1,
                    int(np.ceil(dparam_cur_batch[i, j] / dparam_new_batch[i, j])),
                )
                n_branches[i] *= num_points
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points
        # Compute average branching factor
        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs
        if grid_conservative:
            # Transform dparam_cur_next
            delta_t = coord_next[0] - coord_cur[0]
            n_params = poly_order + 1
            # Construct the transformation matrix
            powers = np.tril(np.arange(n_params)[:, np.newaxis] - np.arange(n_params))
            t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
            dparam_d_vec = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
            dparam_d_vec[:, :-1] = dparam_cur_next
            dparam_d_vec_new = np.sqrt((dparam_d_vec**2) @ (t_mat**2).T)
            dparam_cur_batch = dparam_d_vec_new[:, :-1]
        else:
            dparam_cur_batch = dparam_cur_next

    return branching_pattern


@njit(cache=True, fastmath=True)
def generate_branching_pattern_circular(
    param_arr: types.ListType,
    dparams: np.ndarray,
    param_limits: types.ListType[types.Tuple[float, float]],
    tchunk_ffa: float,
    nstages: int,
    fold_bins: int,
    tol_bins: float,
    ref_seg: int,
    grid_conservative: bool = False,  # noqa: FBT002
) -> np.ndarray:
    poly_order = len(dparams)
    if poly_order != 4:
        msg = "Circular branching pattern requires exactly 4 parameters."
        raise ValueError(msg)

    f0_batch = param_arr[-1]
    n_freqs = len(f0_batch)

    # Snail Scheme
    scheme_data = np.argsort(np.abs(np.arange(nstages + 1) - ref_seg), kind="mergesort")

    # Broadcast current dparams to each freq row
    dparam_cur_batch = np.empty((n_freqs, poly_order), dtype=np.float64)
    for i in range(n_freqs):
        dparam_cur_batch[i] = dparams
    # f = f0(1 - v / C) => dv = -df / f0 * C
    dparam_cur_batch[:, -1] = dparam_cur_batch[:, -1] / f0_batch * C_VAL

    # Track when first snap branching occurs for each frequency
    snap_first_branched = np.zeros(n_freqs, dtype=np.bool_)
    branching_pattern = np.empty(nstages, dtype=np.float64)

    param_limits_d = np.empty((n_freqs, poly_order, 2), dtype=np.float64)
    for i in range(poly_order):
        param_limits_d[:, i, 0] = param_limits[i][0]
        param_limits_d[:, i, 1] = param_limits[i][1]
    param_limits_d[:, -1, 0] = (1 - param_limits[poly_order - 1][1] / f0_batch) * C_VAL
    param_limits_d[:, -1, 1] = (1 - param_limits[poly_order - 1][0] / f0_batch) * C_VAL
    param_ranges = (param_limits_d[:, :, 1] - param_limits_d[:, :, 0]) / 2

    for prune_level in range(1, nstages + 1):
        # Compute coord_cur
        scheme_till_now = scheme_data[: prune_level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        coord_next = ref * tchunk_ffa, scale * tchunk_ffa
        scheme_till_now_cur = scheme_data[:prune_level]
        ref_cur = (np.min(scheme_till_now_cur) + np.max(scheme_till_now_cur) + 1) / 2
        coord_cur = ref_cur * tchunk_ffa, (scale + 0.5) * tchunk_ffa
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
        n_branch_vel = np.ones(n_freqs, dtype=np.int64)
        n_branch_accel = np.ones(n_freqs, dtype=np.int64)
        n_branch_jerk = np.ones(n_freqs, dtype=np.int64)
        n_branch_snap = np.ones(n_freqs, dtype=np.int64)
        n_branches = np.ones(n_freqs, dtype=np.int64)

        # Vectorized branching decision
        eps = 1e-6
        needs_branching = shift_bins_batch >= (tol_bins - eps)
        too_large_step = dparam_new_batch > param_ranges

        validation_fractions = np.ones(n_freqs, dtype=np.float64)

        # First pass: determine branching counts, update dparams, compute stats
        for i in range(n_freqs):
            for j in range(poly_order):
                if not needs_branching[i, j] or too_large_step[i, j]:
                    dparam_cur_next[i, j] = dparam_cur_batch[i, j]
                    continue
                num_points = max(
                    1,
                    int(np.ceil(dparam_cur_batch[i, j] / dparam_new_batch[i, j])),
                )
                dparam_cur_next[i, j] = dparam_cur_batch[i, j] / num_points

                if j == 0:
                    n_branch_snap[i] = num_points
                elif j == 1:
                    n_branch_jerk[i] = num_points
                elif j == 2:
                    n_branch_accel[i] = num_points
                else:
                    n_branch_vel[i] = num_points

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
                * float(n_branch_vel[i])
            )
            n_branches[i] = validation_fractions[i] * raw_combinations

        branching_pattern[prune_level - 1] = np.sum(n_branches) / n_freqs
        if grid_conservative:
            # Transform dparam_cur_next
            delta_t = coord_next[0] - coord_cur[0]
            n_params = poly_order + 1
            # Construct the transformation matrix
            powers = np.tril(np.arange(n_params)[:, np.newaxis] - np.arange(n_params))
            t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
            dparam_d_vec = np.zeros((n_freqs, poly_order + 1), dtype=np.float64)
            dparam_d_vec[:, :-1] = dparam_cur_next
            dparam_d_vec_new = np.sqrt((dparam_d_vec**2) @ (t_mat**2).T)
            dparam_cur_batch = dparam_d_vec_new[:, :-1]
        else:
            dparam_cur_batch = dparam_cur_next
    return branching_pattern
