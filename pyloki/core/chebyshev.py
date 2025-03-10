from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pyloki.core import common
from pyloki.utils import math, np_utils, psr_utils
from pyloki.utils.misc import C_VAL
from pyloki.utils.suggestion import SuggestionStruct


@njit(cache=True, fastmath=True)
def poly_cheb_step(poly_order: int, tol: float, tsamp: float) -> np.ndarray:
    return np.zeros(poly_order + 1, np.float32) + ((tol * tsamp) * C_VAL)


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
def poly_chebyshev_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """Generate the leaf parameter sets for Chebyshev polynomials.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, period).
    dparams : np.ndarray
        Parameter step sizes for each dimension. Shape is (poly_order,).
        - [f, acc, ...]
    poly_order : int
        The order of the Chebyshev polynomial.
    coord_init : tuple[float, float]
        The coordinate of the starting segment.
        - coord_init[0] -> t0 (reference time with respect to which the Chebyshev
            polynomials are defined).
        - coord_init[1] -> scale (scale of the Chebyshev polynomials).

    Returns
    -------
    np.ndarray
        The leaf parameter sets.

    Notes
    -----
    Conventions for the leaf parameter sets:
    alpha[:-2, 0] -> Chebyshev polynomial coefficients,
    alpha[:-2, 1] -> Tolerance (step size) on each coefficient,
    alpha[-2, 0]  -> pulsar frequency at data start (f0),
    alpha[-1, 0]  -> reference time from the data start (t_c),
    alpha[-1, 1]  -> scaling, so that Chebyshev polys are fixed to [-1,1] (t_s).
    """
    t0, scale = coord_init
    param_cart = np_utils.cartesian_prod(param_arr)
    conversion_matrix = np.linalg.inv(math.generalized_cheb_pols(poly_order, t0, scale))

    params_vec = np.zeros((len(param_cart), poly_order + 1))
    params_vec[:, 2] = param_cart[:, -2] / 2.0  # acc / 2.0

    alpha_vec = np.zeros((len(param_cart), poly_order + 3, 2))
    alpha_vec[:, :-2, 0] = np.dot(conversion_matrix.T, params_vec)
    alpha_vec[:, 0, 0] += (t0 % (1 / param_cart[:, -1])) * C_VAL

    alpha_vec[:, 0, 1] = 0.0
    # f = f0(1 + v / C) => dv = df / f0 * C
    alpha_vec[:, 1, 1] = dparams[0] / param_cart[:, -1] * C_VAL
    alpha_vec[:, 2:-2, 1] = dparams[1:] * np.diag(conversion_matrix)[2:]

    alpha_vec[:, -2, 0] = param_cart[:, -1]  # f0
    alpha_vec[:, -1, 0] = t0
    alpha_vec[:, -1, 1] = scale
    return alpha_vec


@njit(cache=True, fastmath=True)
def poly_chebyshev_suggest(
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    score_func: types.FunctionType,
) -> SuggestionStruct:
    """Generate a suggestion struct from a fold segment.

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
        The order of the Polynomial to use.
    score_func : types.FunctionType
        Function to score the folded data.

    Returns
    -------
    SuggestionStruct
        Suggestion struct
        - param_sets: The parameter sets (n_param_sets, poly_order + 3, 2).
        - data: The folded data for each leaf.
        - scores: The scores for each leaf.
        - backtracks: The backtracks for each leaf.
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    param_sets = poly_chebyshev_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets, dtype=np.float32)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)), dtype=np.int32)
    return SuggestionStruct(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def poly_chebyshev_branch(
    param_set: np.ndarray,
    tol_bins: float,
    tsamp: float,
    poly_order: int,
) -> np.ndarray:
    cheb_coeffs_cur = param_set[0:-2, 0]
    dcheb_cur = param_set[0:-2, 1]
    f0, _ = param_set[-2]
    t0_cur, scale_cur = param_set[-1]

    dcheb_opt = poly_cheb_step(poly_order, tol_bins, tsamp)
    leafs_cheb = split_cheb_params(
        cheb_coeffs_cur,
        dcheb_cur,
        dcheb_opt,
        tol_bins,
        tsamp,
    )
    leaves = np.zeros((len(leafs_cheb), poly_order + 3, 2))
    leaves[:, :-2] = leafs_cheb
    leaves[:, -2, 0] = f0
    leaves[:, -1, 0] = t0_cur
    leaves[:, -1, 1] = scale_cur
    return leaves


@njit(cache=True, fastmath=True)
def poly_chebyshev_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    nbins: int,
    cheb_table: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    coord_add : tuple[float, float]
        The coordinates of the added segment (level cur).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    nbins : int
        Number of bins in the folded profile.
    cheb_table : np.ndarray
        Precomputed Chebyshev polynomials coefficients.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to t0, so we need to shift it to t_ref_add and t_ref_init
    to get the resolved parameters index and phase shift.

    """
    t0, scale = leaf[-1]
    t_add = coord_add[0]
    t_init = coord_init[0]
    param_vec = leaf[:-2, 0]
    f0 = leaf[-2, 0]

    eff_deg = effective_degree(param_vec, 1000)
    a_t_add = math.cheby2taylor(param_vec, t_add, t0, scale, 2, cheb_table, eff_deg)
    v_t_add = math.cheby2taylor(param_vec, t_add, t0, scale, 1, cheb_table, eff_deg)
    v_t_init = math.cheby2taylor(param_vec, t_init, t0, scale, 1, cheb_table, eff_deg)
    d_t_add = math.cheby2taylor(param_vec, t_add, t0, scale, 0, cheb_table, eff_deg)
    d_t_init = math.cheby2taylor(param_vec, t_init, t0, scale, 0, cheb_table, eff_deg)

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
    cheb_pols_cur = math.generalized_cheb_pols(poly_order, t0_cur, scale_cur)
    cheb_pols_prev = math.generalized_cheb_pols(poly_order, t0_prev, scale_prev)
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


@njit(cache=True, fastmath=True)
def leaf_validate_physical(
    leaf: np.ndarray,
    time_arr: np.ndarray,
    validation_params: tuple[np.ndarray, np.ndarray, float],
    deriv_bounds: np.ndarray,
    cheb_table: np.ndarray,
    period_bounds: np.ndarray,
) -> bool:
    t0, scale = leaf[-1]
    param_vec = leaf[:-2, 0]
    p0 = 1 / leaf[-2, 0]
    eff_degree = effective_degree(param_vec, 1000)
    values = math.cheby2taylor(
        param_vec,
        time_arr,
        t0,
        scale,
        0,
        cheb_table,
        eff_degree,
    )
    med = np.median(values)
    st = np.std(values)
    max_diff = np.max(values) - med
    min_diff = med - np.min(values)
    epicycle_err = err_epicycle_fit_fast(
        values,
        validation_params[0],
        validation_params[1],
    )
    is_epicycle_fit = epicycle_err < validation_params[2]
    if max_diff < 3.2 * st and min_diff < 3.2 * st and is_epicycle_fit:
        good = True
        for deriv_index in range(1, len(deriv_bounds)):
            values = math.cheby2taylor(
                param_vec,
                time_arr,
                t0,
                scale,
                deriv_index,
                cheb_table,
                eff_degree,
            )
            if deriv_index == 1:
                if (np.max(values) - np.min(values)) > 2 * deriv_bounds[deriv_index]:
                    good = False
                    break
                if (((1 - np.min(values) / C_VAL) * p0) < period_bounds[0]) or (
                    ((1 - np.max(values) / C_VAL) * p0) > period_bounds[1]
                ):
                    good = False
                    break
            elif np.max(np.abs(values)) > deriv_bounds[deriv_index]:
                good = False
                break
    return good


@njit(cache=True, fastmath=True)
def err_epicycle_fit(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    om_arr: np.ndarray,
) -> float:
    min_err = np.inf
    n = len(time_arr)
    mat = np.zeros((6, n), np.float64)
    mat[6, :] = 1
    mat[1, :] = time_arr
    for om in om_arr:
        mat[2, :] = np.sin(om * time_arr)
        mat[3, :] = np.cos(om * time_arr)
        mat[4, :] = np.sin(2 * om * time_arr)
        mat[5, :] = np.cos(2 * om * time_arr)
        dot_mat = np.linalg.inv(np.dot(mat, mat.T))
        x = np.dot(dot_mat, np.dot(mat, values_arr))
        err = np.max(np.abs(np.dot(mat.T, x) - values_arr))
        min_err = min(err, min_err)
    return min_err


@njit(cache=True, fastmath=True)
def err_epicycle_fit_fast(
    values_arr: np.ndarray,
    mat_list_left: np.ndarray,
    mat_list_right: np.ndarray,
) -> float:
    min_err = np.inf
    for i in range(mat_list_right.shape[0]):
        x = np.dot(mat_list_right[i], values_arr)
        err_vec = np.dot(mat_list_left[i], x) - values_arr
        err = np.max(np.abs(err_vec))
        min_err = min(err, min_err)
    return min_err


@njit(cache=True, fastmath=True)
def prepare_epicyclic_validation_params(
    tcheby: float,
    tseg_ffa: float,
    n_valid: int,
    omega_min: float,
    omega_max: float,
    x_max: float,
    ecc_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    time_arr = np.linspace(3 * tseg_ffa / 2, tcheby - tseg_ffa / 2, n_valid)
    epicycle_bound = 2 * x_max * ecc_max**2 * C_VAL
    d_omega = ecc_max**2 / (time_arr[-1] - time_arr[0])
    omega_arr = np.arange(omega_min, omega_max, d_omega)

    n_omega = len(omega_arr)
    n = len(time_arr)

    fit_mat_left = np.empty((n_omega, n, 6), dtype=np.float64)
    fit_mat_right = np.empty((n_omega, 6, n), dtype=np.float64)

    time_terms = np.column_stack(
        [
            np.ones(n),
            time_arr,
            np.sin(omega_arr[:, np.newaxis] * time_arr),
            np.cos(omega_arr[:, np.newaxis] * time_arr),
            np.sin(2 * omega_arr[:, np.newaxis] * time_arr),
            np.cos(2 * omega_arr[:, np.newaxis] * time_arr),
        ],
    )

    for i in range(n_omega):
        mat = time_terms[i].T
        dot_mat = np.linalg.inv(np.dot(mat, mat.T))
        fit_mat_right[i] = np.dot(dot_mat, mat)
        fit_mat_left[i] = mat.T
    return fit_mat_left, fit_mat_right, epicycle_bound
