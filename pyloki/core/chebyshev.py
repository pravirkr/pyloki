from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import maths, np_utils, psr_utils
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
        Order is [..., acc, freq].
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
        - coord_init[0] -> t0 (reference time)
        - coord_init[1] -> scale (half duration of the segment)
    coord_mid : tuple[float, float]
        The mid-point coordinate of the entire data.
        - coord_mid[0] -> t_mid (midpoint time) measured from t=0
        - coord_mid[1] -> scale (half duration of the entire data)

    Returns
    -------
    np.ndarray
        The leaf parameter sets. Shape is (n_param_sets, poly_order + 2, 2).

    Notes
    -----
    Conventions for each leaf parameter set:
    leaf[:-2, 0] -> Chebyshev polynomial coefficients, order is [alpha_1, alpha_2, ...]
    leaf[:-2, 1] -> Grid size (error) on each coefficient,
    leaf[-2, 0]  -> Frequency at data mid-point (f0), assuming f=f0 at t_mid
    leaf[-1, 0]  -> Reference time from the data start (t_c),
    leaf[-1, 1]  -> Scaling, half duration of the segment (t_s).
    """
    t_init, scale = coord_init
    leaves_taylor = common.get_leaves(param_arr, dparams)
    f0_batch = leaves_taylor[:, -1, 0]
    # f = f0(1 - v / C) => dv = -df / f0 * C
    leaves_taylor[:, -1, 0] = 0
    leaves_taylor[:, -1, 1] = leaves_taylor[:, -1, 1] / f0_batch * C_VAL
    leaves_taylor_d = np.zeros((len(leaves_taylor), poly_order + 1, 2))
    leaves_taylor_d[:, :-1] = leaves_taylor
    alpha_vec = maths.taylor_to_cheby_full(leaves_taylor_d, scale)

    leaves = np.zeros((len(leaves_taylor), poly_order + 3, 2), dtype=np.float64)
    leaves[:, :-2] = alpha_vec
    leaves[:, -2, 0] = f0_batch
    leaves[:, -1, 0] = t_init
    leaves[:, -1, 1] = scale
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
        scores[iparam] = scoring.snr_score_func(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 3), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks)


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
        scores[iparam] = scoring.snr_score_func_complex(data[iparam], score_widths)
    backtracks = np.zeros((n_param_sets, poly_order + 3), dtype=np.int32)
    return SuggestionStructComplex(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def split_cheb_params(
    cheb_coeffs_cur: np.ndarray,
    dcheb_cur: np.ndarray,
    dcheb_opt: np.ndarray,
    tol: float,
    tsamp: float,
) -> np.ndarray:
    ncoeffs = len(dcheb_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(ncoeffs, dtype=np.float64)

    effective_tol = tol * tsamp * C_VAL
    for i in range(ncoeffs):
        if abs(dcheb_cur[i]) > effective_tol:
            dcheb_opt[i] = min(dcheb_opt[i], 0.5 * dcheb_cur[i])
            leaf_param, dparam_act = psr_utils.branch_param(
                cheb_coeffs_cur[i],
                dcheb_cur[i],
                dcheb_opt[i],
            )
        else:
            leaf_param, dparam_act = np.array([cheb_coeffs_cur[i]]), dcheb_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param)
    return common.get_leaves(leaf_params, leaf_dparams)


@njit(cache=True, fastmath=True)
def effective_degree(pol_coeffs: np.ndarray, eps: float) -> int:
    mask = np.abs(pol_coeffs) > eps
    return np.max(mask * np.arange(len(pol_coeffs)))


@njit(cache=True, fastmath=True)
def poly_chebyshev_branch_batch(
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
    fold_bins: int,
    tol_bins: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
    branch_max: int,
) -> np.ndarray:
    """Branch a batch of parameter sets to leaves."""
    n_batch = len(param_set_batch)
    nparams = poly_order
    _, t_obs_minus_t_ref = coord_cur
    cheb_cur_batch = param_set_batch[:, :-3, 0]
    dcheb_cur_batch = param_set_batch[:, :-3, 1]
    d0_batch = param_set_batch[:, -3, 0]
    f0_batch = param_set_batch[:, -2, 0]
    t0_batch = param_set_batch[:, -1, 0]
    scale_batch = param_set_batch[:, -1, 1]

    dcheb_new_batch = psr_utils.poly_cheb_step_vec(
        nparams,
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
    pad_branched_params = np.empty((n_batch, nparams, branch_max), dtype=np.float64)
    pad_branched_dparams = np.empty((n_batch, nparams), dtype=np.float64)
    branched_counts = np.empty((n_batch, nparams), dtype=np.int64)
    for i in range(n_batch):
        for j in range(nparams):
            p_min, p_max = param_limits_d[i, j]
            dparam_act, count = psr_utils.branch_param_padded(
                pad_branched_params[i, j],
                cheb_cur_batch[i, j],
                dcheb_cur_batch[i, j],
                dcheb_new_batch[i, j],
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
                pad_branched_params[i, j, 0] = cheb_cur_batch[i, j]
                pad_branched_dparams[i, j] = dcheb_cur_batch[i, j]
                branched_counts[i, j] = 1
    # --- Optimized Padded Cartesian Product ---
    batch_leaves_cheb, batch_origins = np_utils.cartesian_prod_padded(
        pad_branched_params,
        branched_counts,
        n_batch,
        nparams,
    )
    total_leaves = len(batch_origins)
    batch_leaves = np.zeros((total_leaves, poly_order + 3, 2), dtype=np.float64)
    batch_leaves[:, :-3, 0] = batch_leaves_cheb
    batch_leaves[:, :-3, 1] = pad_branched_dparams[batch_origins]
    batch_leaves[:, -3, 0] = d0_batch[batch_origins]
    batch_leaves[:, -2, 0] = f0_batch[batch_origins]
    batch_leaves[:, -1, 0] = t0_batch[batch_origins]
    batch_leaves[:, -1, 1] = scale_batch[batch_origins]
    return batch_leaves


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
        The leaf parameter set. Shape is (n_batch, poly_order + 3, 2).
    coord_add : tuple[float, float]
        The coordinates of the added segment (level cur).
    coord_cur : tuple[float, float]
        The coordinates of the current segment (level cur).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    param_arr : types.ListType
        Parameter array containing the parameter values for the incoming segment.
    fold_bins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaves_batch is referenced to t0, so we need to shift it to t_ref_add and t_ref_init
    to get the resolved parameters index and phase shift.

    """
    n_batch, _, _ = leaves_batch.shape
    t0, _ = coord_cur
    t_init, _ = coord_init
    t_add, _ = coord_add
    param_vec = leaves_batch[:, :-2, 0]
    f0_batch = leaves_batch[:, -2, 0]

    eff_deg = effective_degree(param_vec, 1000)
    a_t_add = maths.cheby2taylor(param_vec, t_add, t0, scale, 2, cheb_table, eff_deg)
    v_t_add = maths.cheby2taylor(param_vec, t_add, t0, scale, 1, cheb_table, eff_deg)
    v_t_init = maths.cheby2taylor(param_vec, t_init, t0, scale, 1, cheb_table, eff_deg)
    d_t_add = maths.cheby2taylor(param_vec, t_add, t0, scale, 0, cheb_table, eff_deg)
    d_t_init = maths.cheby2taylor(param_vec, t_init, t0, scale, 0, cheb_table, eff_deg)

    new_a = a_t_add
    new_f = f0 * (1 + (v_t_add - v_t_init) / C_VAL)
    delay = (d_t_add - d_t_init) / C_VAL

    # phase is measured relative to the phase at 0
    relative_phase = psr_utils.get_phase_idx(coord_add[0], f0, nbins, delay)
    idx_a = np_utils.find_nearest_sorted_idx(param_arr[-2], new_a)
    idx_f = np_utils.find_nearest_sorted_idx(param_arr[-1], new_f)
    index_prev = np.empty(len(param_arr), dtype=np.int64)
    index_prev[-1] = idx_f
    index_prev[-2] = idx_a
    return index_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_chebyshev_transform_matrix(
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    poly_order: int,
) -> np.ndarray:
    t0_cur, scale_cur = coord_cur
    t0_prev, scale_prev = coord_prev
    # check the ordering inside the generalized_cheb_pols
    cheb_pols_cur = maths.generalized_cheb_pols(poly_order, t0_cur, scale_cur)
    cheb_pols_prev = maths.generalized_cheb_pols(poly_order, t0_prev, scale_prev)
    return np.dot(cheb_pols_prev, np.linalg.inv(cheb_pols_cur))


@njit(cache=True, fastmath=True)
def poly_chebyshev_transform(
    leaf: np.ndarray,
    coord_ref: tuple[float, float],
    trans_matrix: np.ndarray,
) -> np.ndarray:
    leaf_trans = np.zeros_like(leaf)
    params = leaf[0:-2, 0]
    dparams = leaf[0:-2, 1]
    params *= np.abs(params) > 1e5
    params_new = np.dot(params, trans_matrix)
    # Choose between the volume treatment approaches. each has it own advantages
    # \dparams_new = np.sqrt(np.dot(A**2, dparams**2)) # Conservative approach
    dparams_new = np.diag(trans_matrix) * dparams  # Violent ignorance approach
    leaf_trans[:-2, 0] = params_new
    leaf_trans[:-2, 1] = dparams_new
    leaf_trans[-2] = leaf[-2]
    leaf_trans[-1] = coord_ref
    return leaf_trans


@njit(cache=True, fastmath=True)
def poly_chebyshev_validate(
    leaves: np.ndarray,
    t_ref_add: float,
    t_ref_init: float,
    validation_params: tuple[np.ndarray, np.ndarray, float],
    tseg_ffa: float,
    deriv_bounds: np.ndarray,
    n_valid: int,
    cheb_table: np.ndarray,
    period_bounds: np.ndarray,
) -> np.ndarray:
    nleaves = len(leaves)
    mask = np.zeros(nleaves, np.bool_)
    t0 = leaves[0, -1, 0]
    # Important not to check on the segment edges because quantization effects
    # may cause unphysical derivatives that are later corrected.
    tcheby = t_ref_add - t0
    tzero = t_ref_init - t0
    time_arr = np.linspace(
        tzero + 3 * tseg_ffa / 2,
        tcheby - tseg_ffa / 2,
        n_valid,
    )
    for ileaf in range(nleaves):
        mask[ileaf] = leaf_validate_physical(
            leaves[ileaf],
            time_arr,
            validation_params,
            deriv_bounds,
            cheb_table,
            period_bounds,
        )

    return mask
