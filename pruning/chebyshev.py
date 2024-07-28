from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from pruning import kernels, math, utils


@njit(cache=True, fastmath=True)
def cheb_step(poly_order: int, tsamp: float, tol: int) -> np.ndarray:
    return np.zeros(poly_order + 1, np.float32) + ((tol * tsamp) * utils.c_val)


@njit(cache=True, fastmath=True)
def split_cheb_params(
    cheb_coeffs_cur: np.ndarray,
    dcheb_cur: np.ndarray,
    dcheb_opt: np.ndarray,
    tol_time: float,
) -> np.ndarray:
    ncoeffs = len(dcheb_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(ncoeffs, dtype=np.float64)

    effective_tol = tol_time * utils.c_val
    for i in range(ncoeffs):
        if abs(dcheb_cur[i]) > effective_tol:
            dcheb_opt[i] = min(dcheb_opt[i], 0.5 * dcheb_cur[i])
            leaf_param, dparam_act = kernels.branch_param(
                dcheb_opt[i],
                dcheb_cur[i],
                cheb_coeffs_cur[i],
            )
        else:
            leaf_param = np.array((cheb_coeffs_cur[i],))
            dparam_act = dcheb_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param)
    return kernels.get_leaves(leaf_params, leaf_dparams)


@njit(cache=True, fastmath=True)
def effective_degree(pol_coeffs: np.ndarray, eps: float) -> int:
    mask = np.abs(pol_coeffs) > eps
    return np.max(mask * np.arange(len(pol_coeffs)))


@njit(cache=True, fastmath=True)
def find_small_polys(
    deg: int,
    err: int,
    over_samp: int = 4,
) -> tuple[list, float, float]:
    """Enumerate the phase space of all polynomials up to a certain resolution.

    To find the volume of small polynomials. The volume roughly fits the expected
    volume from using the basis of chebychev polynomials.

    """
    test_range = np.linspace(-1, 1, 128)
    phase_space = math.cartesian_prod(
        [
            np.linspace(-(deg - 1) * err, (deg - 1) * err, 2 * (deg - 1) * over_samp)
            for _ in range(deg)
        ],
    )
    point_volume = (2 * (deg - 1)) ** deg / len(phase_space)
    x_matrix = test_range[:, np.newaxis] ** np.arange(deg)
    poly_values = np.dot(x_matrix, phase_space.T)
    # Find good polynomials
    good_poly_mask = np.sum(np.abs(poly_values) > err, 0) < 12
    good_poly = phase_space[good_poly_mask]
    volume_factor = point_volume * len(good_poly) / 2**deg
    return good_poly, point_volume, volume_factor


@njit(cache=True, fastmath=True)
def generate_chebyshev_polys_table(order_max: int, n_derivs: int) -> np.ndarray:
    """Generate table of Chebyshev polynomials of the first kind and their derivatives.

    Parameters
    ----------
    order_max : int
        The maximum order of the polynomials (T_0, T_1, ..., T_order_max)
    n_derivs : int
        The number of derivatives to generate for each polynomial (0th derivative
        is the polynomial itself)

    Returns
    -------
    np.ndarray
        A 3D array of shape (n_derivs + 1, order_max + 1, order_max + 1) containing the
        coefficients of the polynomials.
    """
    tab = np.zeros((n_derivs + 1, order_max + 1, order_max + 1), dtype=np.float32)
    tab[0, 0, 0] = 1.0
    tab[0, 1, 1] = 1.0

    for jorder in range(2, order_max + 1):
        tab[0, jorder] = 2 * np.roll(tab[0, jorder - 1], 1) - tab[0, jorder - 2]

    factor = np.arange(1, order_max + 2, dtype=np.float32)
    for ideriv in range(1, n_derivs + 1):
        for jorder in range(1, order_max + 1):
            tab[ideriv, jorder] = np.roll(tab[ideriv - 1, jorder], -1) * factor
            tab[ideriv, jorder, -1] = 0

    return tab


@njit(fastmath=True)
def generalized_cheb_pols(poly_order: int, t0: float, scale: float) -> np.ndarray:
    """Generate table of generalized Chebyshev polynomials.

    Parameters
    ----------
    poly_order : int
        The maximum order of the polynomials (upto T_poly_order)
    t0 : float
        Shifted origin of the polynomials.
    scale : float
        Scaling factor for the polynomials.

    Returns
    -------
    np.ndarray
        A 2D array of shape (poly_order + 1, poly_order + 1) containing the coefficients
        of the polynomials.

    Notes
    -----
    The generalized Chebyshev polynomials are defined as:
    T*_n(y) = T_n((x - t0) / scale)
    where :math: y in [-1, 1] and x in [t0 - scale, t0 + scale].
    """
    cheb_pols = generate_chebyshev_polys_table(poly_order, 0)[0]

    # Shift the origin to t0
    cheb_pols_shifted = np.zeros_like(cheb_pols)
    for iorder in range(poly_order + 1):
        iterms = np.arange(iorder + 1, dtype=np.float32)
        shifted = math.nbinom(iorder, iterms) * (-t0 / scale) ** (iorder - iterms)
        cheb_pols_shifted[iorder, : len(shifted)] = shifted
    pols = np.dot(cheb_pols, cheb_pols_shifted)
    # scale the polynomials
    pols *= scale ** (-np.arange(poly_order + 1, dtype=np.float32))
    return pols


@njit(cache=True, fastmath=True)
def gen_transfer_matrix(
    poly_order: int,
    scale0: float,
    t0: float,
    scale1: float,
    t1: float,
) -> np.ndarray:
    cheb_pols1 = generalized_cheb_pols(poly_order, scale0, t0)
    cheb_pol2 = generalized_cheb_pols(poly_order, scale1, t1)
    return np.dot(cheb_pols1, np.linalg.inv(cheb_pol2))


@njit(cache=True, fastmath=True)
def chebychev_poly_evaluate(
    cheb_table: np.ndarray,
    t_minus_t0: float,
    param_set: np.ndarray,
    deriv_index: int,
    eff_deg: int = -3,
) -> np.ndarray:
    """Evaluate a Chebyshev polynomial at a given time.

    Parameters
    ----------
    cheb_table : np.ndarray
        Precomputed Chebyshev polynomials and their derivatives.
    t_minus_t0 : float
        Time at which to evaluate the polynomial.
    param_set : np.ndarray
        Parameter set with Chebyshev coefficients.
    deriv_index : int
        Index of the derivative to evaluate.
    eff_deg : int, optional
        Effective degree of the polynomial, by default -3.

    Returns
    -------
    np.ndarray
        The value of the polynomial at the given time.

    Notes
    -----
    The effective degree is -3 by default as this value reproduces (after +1) the -2
    (which is the last coefficient by convention).
    """
    tab = cheb_table[deriv_index]
    scale = param_set[-1, 1]
    coeffs = param_set[: eff_deg + 1, 0]
    eff_deg = len(coeffs) - 1 if eff_deg < 0 else eff_deg
    pol = np.sum(coeffs[:, np.newaxis] * tab[:, : eff_deg + 1], axis=0)
    polyval = np.polynomial.polynomial.polyval(t_minus_t0 / scale, pol)
    return polyval / scale**deriv_index


@njit(cache=True, fastmath=True)
def get_leaves_chebyshev(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    t0: float,
    scale: float,
) -> np.ndarray:
    """Generate the leaf parameter sets for Chebyshev polynomials.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, period).
    dparams : np.ndarray
        Parameter step sizes for each dimension. Shape is (poly_order,).
    poly_order : int
        The order of the Chebyshev polynomial.
    t0 : float
        The reference time with respect to which the Chebyshev polynomials are defined.
    scale : float
        The scale of the Chebyshev polynomials.

    Returns
    -------
    np.ndarray
        The leaf parameter sets.

    Notes
    -----
    Conventions for the leaf parameter sets:
    alpha[:-2, 0] -> Chebyshev polynomial coefficients,
    alpha[:-2, 1] -> tolerance on each coefficient,
    alpha[-2, 0]  -> pulsar period at data start (p0),
    alpha[-1, 0]  -> reference time (from the data start) (t0),
    alpha[-1, 1]  -> scaling (scales the input to the polynomials, so that
                    the polynomials are defined in the range [-1, 1]).

    """
    param_cart = math.cartesian_prod(param_arr)
    conversion_matrix = np.linalg.inv(generalized_cheb_pols(poly_order, t0, scale))

    params_vec = np.zeros((len(param_cart), poly_order + 1))
    params_vec[:, 2] = param_cart[:, 0] / 2.0  # acc / 2.0

    alpha_vec = np.zeros((len(param_cart), poly_order + 3, 2))
    alpha_vec[:, :-2, 0] = np.dot(conversion_matrix.T, params_vec)
    alpha_vec[:, 0, 0] += (t0 % param_cart[:, 1]) * utils.c_val

    alpha_vec[:, 0, 1] = 0.0
    alpha_vec[:, 1, 1] = dparams[0] / param_cart[:, 1] * utils.c_val
    alpha_vec[:, 2:-2, 1] = dparams[1:] * np.diag(conversion_matrix)[2:]

    alpha_vec[:, -2, 0] = param_cart[:, 1]  # p0
    alpha_vec[:, -1, 0] = t0
    alpha_vec[:, -1, 1] = scale
    return alpha_vec


@njit(cache=True, fastmath=True)
def suggestion_struct_chebyshev(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    t0: float,
    scale: float,
    score_func: types.FunctionType,
) -> kernels.SuggestionStruct:
    """Generate a suggestion struct from a fold segment for Chebyshev polynomials.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_period, 2, n_bins). Parameter dimensions are first two.
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Chebyshev polynomial to use.
    t0 : float
        The reference time with respect to which the Chebyshev polynomials are defined.
    scale : float
        The scale of the Chebyshev polynomials.
    score_func : types.FunctionType
        Function to score the folded data.

    Returns
    -------
    kernels.SuggestionStruct
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    # \n_param_sets = n_accel * n_period
    # \param_sets_shape = [n_param_sets, poly_order + 3, 2]
    param_sets = get_leaves_chebyshev(param_arr, dparams, poly_order, t0, scale)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)))
    return kernels.SuggestionStruct(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def poly_chebychev_branch2leaves(
    param_set: np.ndarray,
    poly_order: int,
    tolerance: float,
    tsamp: float,
) -> np.ndarray:
    # multiplication by 2 is due to the middle out scheme.
    # indexing_distance = 2*abs(new_index - reference_index) + 1
    # total_duration = indexing_distance * duration

    cheb_coeffs_cur = param_set[0:-2, 0]
    dcheb_cur = param_set[0:-2, 1]
    p0 = param_set[-2, 0]
    t0 = param_set[-1, 0]
    scale = param_set[-1, 1]

    dcheb_opt = cheb_step(poly_order, tsamp, tolerance)
    leafs_cheb = split_cheb_params(
        cheb_coeffs_cur,
        dcheb_opt,
        dcheb_cur,
        tolerance * tsamp,
    )
    leaves = np.zeros((len(leafs_cheb), poly_order + 3, 2))
    leaves[:, :-2] = leafs_cheb
    leaves[:, -2, 0] = p0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_chebychev_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    t_ref_cur: float,
    t_ref_init: float,
    nbins: int,
    cheb_table: np.ndarray,
) -> tuple[tuple[int, int], int]:
    """Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    t_ref_cur : float
        The reference time for the current segment.
    t_ref_init : float
        The reference time for the initial segment (pruning level 0).
    nbins : int
        Number of bins in the folded profile.
    cheb_table : np.ndarray
        Precomputed Chebyshev polynomials coefficients.

    Returns
    -------
    tuple[tuple[int, int], int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to t0, so we need to shift it to t_ref_cur and t_ref
    to get the resolved parameters index and phase shift.

    """
    t0 = leaf[-1, 0]
    tcheby = t_ref_cur - t0
    tzero = t_ref_init - t0

    eff_degree = effective_degree(leaf[:-2, 0], 1000)
    a_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 2, eff_degree)
    v_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 1, eff_degree)
    v_tzero = chebychev_poly_evaluate(cheb_table, tzero, leaf, 1, eff_degree)
    d_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 0, eff_degree)
    d_tzero = chebychev_poly_evaluate(cheb_table, tzero, leaf, 0, eff_degree)

    p0 = leaf[-2, 0]
    new_a = a_tcheby
    new_p = p0 * (1 - (v_tcheby - v_tzero) / utils.c_val)
    delay = (d_tcheby - d_tzero) / utils.c_val

    relative_phase = kernels.get_phase_idx(t_ref_cur, p0, nbins, delay)
    idx_a = math.find_nearest_sorted_idx(param_arr[-2], new_a)
    idx_f = math.find_nearest_sorted_idx(param_arr[-1], new_p)
    return (idx_a, idx_f), relative_phase


@njit(cache=True, fastmath=True)
def poly_chebychev_coord_trans_matrix(
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    poly_order: int,
) -> np.ndarray:
    t0_prev, scale_prev = coord_prev
    t0_cur, scale_cur = coord_cur
    return gen_transfer_matrix(poly_order, scale_prev, t0_prev, scale_cur, t0_cur)


@njit(cache=True, fastmath=True)
def poly_chebychev_coord_trans(
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
    leaf_trans[-2, 0] = leaf[-2, 0]
    leaf_trans[-1, 0] = coord_ref[0]
    leaf_trans[-1, 1] = coord_ref[1]
    return leaf_trans


@njit(cache=True, fastmath=True)
def poly_chebychev_physical_validation(
    leaves: np.ndarray,
    tcheby: float,
    tzero: float,
    validation_params: tuple[np.ndarray, np.ndarray, float],
    tsegment_ffa: float,
    derivative_bounds: np.ndarray,
    num_validation: int,
    cheb_table: np.ndarray,
    period_bounds: np.ndarray,
) -> np.ndarray:
    nleaves = len(leaves)
    mask = np.zeros(nleaves, np.bool_)
    t0 = leaves[0, -1, 0]
    # Important not to check on the segment edges because quantization effects
    # may cause unphysical derivatives that are later corrected.
    time_arr = (
        np.linspace(
            tzero + 3 * tsegment_ffa / 2,
            tcheby - tsegment_ffa / 2,
            num_validation,
        )
        - t0
    )
    for ileaf in range(nleaves):
        mask[ileaf] = leaf_validate_physical(
            leaves[ileaf],
            time_arr,
            validation_params,
            derivative_bounds,
            cheb_table,
            period_bounds,
        )

    return mask


@njit(cache=True, fastmath=True)
def leaf_validate_physical(
    leaf: np.ndarray,
    time_arr: np.ndarray,
    validation_params: tuple[np.ndarray, np.ndarray, float],
    derivative_bounds: np.ndarray,
    cheb_table: np.ndarray,
    period_bounds: np.ndarray,
) -> bool:
    eff_degree = effective_degree(leaf[:-2, 0], 1000)
    values = chebychev_poly_evaluate(cheb_table, time_arr, leaf, 0, eff_degree)
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
        p0 = leaf[-2, 0]
        good = True
        for deriv_index in range(1, len(derivative_bounds)):
            values = chebychev_poly_evaluate(
                cheb_table,
                time_arr,
                leaf,
                deriv_index,
                eff_degree,
            )
            if deriv_index == 1:
                if (np.max(values) - np.min(values)) > 2 * derivative_bounds[
                    deriv_index
                ]:
                    good = False
                    break
                if (((1 - np.min(values) / utils.c_val) * p0) < period_bounds[0]) or (
                    ((1 - np.max(values) / utils.c_val) * p0) > period_bounds[1]
                ):
                    good = False
                    break
            elif np.max(np.abs(values)) > derivative_bounds[deriv_index]:
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
        if err < min_err:
            min_err = err
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
        if err < min_err:
            min_err = err
    return min_err


@njit(cache=True, fastmath=True)
def prepare_epicyclic_validation_params(
    tcheby: float,
    tsegment_ffa: float,
    num_validation: int,
    omega_min: float,
    omega_max: float,
    x_max: float,
    ecc_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    time_arr = np.linspace(
        3 * tsegment_ffa / 2,
        tcheby - tsegment_ffa / 2,
        num_validation,
    )
    epicycle_bound = 2 * x_max * ecc_max**2 * utils.c_val
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
