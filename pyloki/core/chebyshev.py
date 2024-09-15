from __future__ import annotations

import numpy as np
from numba import njit, typed, types
from numba.experimental import jitclass

from pyloki.config import PulsarSearchConfig
from pyloki.core import common, defaults
from pyloki.detection import scoring
from pyloki.utils import math, np_utils, psr_utils
from pyloki.utils.misc import C_VAL


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
    phase_space = np_utils.cartesian_prod(
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
def poly_chebyshev_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """
    Generate the leaf parameter sets for Chebyshev polynomials.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, period).
    dparams : np.ndarray
        Parameter step sizes for each dimension. Shape is (poly_order,).
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
    alpha[:-2, 1] -> tolerance on each coefficient,
    alpha[-2, 0]  -> pulsar period at data start (p0),
    alpha[-1, 0]  -> reference time (from the data start) (t0),
    alpha[-1, 1]  -> scaling (scales the input to the polynomials, so that
                    the polynomials are defined in the range [-1, 1]).

    """
    t0, scale = coord_init
    param_cart = np_utils.cartesian_prod(param_arr)
    conversion_matrix = np.linalg.inv(math.generalized_cheb_pols(poly_order, t0, scale))

    params_vec = np.zeros((len(param_cart), poly_order + 1))
    params_vec[:, 2] = param_cart[:, 0] / 2.0  # acc / 2.0

    alpha_vec = np.zeros((len(param_cart), poly_order + 3, 2))
    alpha_vec[:, :-2, 0] = np.dot(conversion_matrix.T, params_vec)
    alpha_vec[:, 0, 0] += (t0 % param_cart[:, 1]) * C_VAL

    alpha_vec[:, 0, 1] = 0.0
    alpha_vec[:, 1, 1] = dparams[0] / param_cart[:, 1] * C_VAL
    alpha_vec[:, 2:-2, 1] = dparams[1:] * np.diag(conversion_matrix)[2:]

    alpha_vec[:, -2, 0] = param_cart[:, 1, 0]  # f0
    alpha_vec[:, -1, 0] = t0
    alpha_vec[:, -1, 1] = scale
    return alpha_vec


@njit(cache=True, fastmath=True)
def poly_chebychev_suggestion_struct(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
    score_func: types.FunctionType,
) -> common.SuggestionStruct:
    """
    Generate a suggestion struct from a fold segment.

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
    coord_init : tuple[float, float]
        The coordinate of the starting segment (level 0).
    score_func : types.FunctionType
        Function to score the folded data.

    Returns
    -------
    common.SuggestionStruct
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    # \n_param_sets = n_accel * n_period
    # \param_sets_shape = [n_param_sets, poly_order + 3, 2]
    param_sets = poly_chebyshev_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)))
    return common.SuggestionStruct(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def poly_chebychev_branch2leaves(
    param_set: np.ndarray,
    tol: float,
    tsamp: float,
    poly_order: int,
) -> np.ndarray:
    cheb_coeffs_cur = param_set[0:-2, 0]
    dcheb_cur = param_set[0:-2, 1]
    p0, _ = param_set[-2]
    t0_cur, scale_cur = param_set[-1]

    dcheb_opt = poly_cheb_step(poly_order, tol, tsamp)
    leafs_cheb = split_cheb_params(
        cheb_coeffs_cur,
        dcheb_cur,
        dcheb_opt,
        tol,
        tsamp,
    )
    leaves = np.zeros((len(leafs_cheb), poly_order + 3, 2))
    leaves[:, :-2] = leafs_cheb
    leaves[:, -2, 0] = p0
    leaves[:, -1, 0] = t0_cur
    leaves[:, -1, 1] = scale_cur
    return leaves


@njit(cache=True, fastmath=True)
def poly_chebychev_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    nbins: int,
    cheb_table: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Resolve the leaf parameters to find the closest param index and phase shift.

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
    Check if the t_ref should be in the middle of the segment.

    """
    t0, _ = leaf[-1]
    tcheby = coord_add[0] - t0
    tzero = coord_init[0] - t0

    eff_deg = effective_degree(leaf[:-2, 0], 1000)
    a_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 2, eff_deg)
    v_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 1, eff_deg)
    v_tzero = chebychev_poly_evaluate(cheb_table, tzero, leaf, 1, eff_deg)
    d_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, leaf, 0, eff_deg)
    d_tzero = chebychev_poly_evaluate(cheb_table, tzero, leaf, 0, eff_deg)

    f0 = leaf[-2, 0]

    new_a = a_tcheby
    new_f = f0 * (1 + (v_tcheby - v_tzero) / C_VAL)
    delay = (d_tcheby - d_tzero) / C_VAL

    # phase is measured relative to the phase at 0
    relative_phase = psr_utils.get_phase_idx(coord_add[0], f0, nbins, delay)
    idx_a = np_utils.find_nearest_sorted_idx(param_arr[-2], new_a)
    idx_f = np_utils.find_nearest_sorted_idx(param_arr[-1], new_f)
    index_prev = np.empty(len(param_arr), dtype=np.int64)
    index_prev[-1] = idx_f
    index_prev[-2] = idx_a
    return index_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_chebychev_transform_matrix(
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    poly_order: int,
) -> np.ndarray:
    t0_cur, scale_cur = coord_cur
    t0_prev, scale_prev = coord_prev
    cheb_pols_cur = math.generalized_cheb_pols(poly_order, t0_cur, scale_cur)
    cheb_pols_prev = math.generalized_cheb_pols(poly_order, t0_prev, scale_prev)
    return np.dot(cheb_pols_prev, np.linalg.inv(cheb_pols_cur))


@njit(cache=True, fastmath=True)
def poly_chebychev_transform(
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
def poly_chebychev_validate(
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
        for deriv_index in range(1, len(deriv_bounds)):
            values = chebychev_poly_evaluate(
                cheb_table,
                time_arr,
                leaf,
                deriv_index,
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


@jitclass(
    spec=[
        ("cfg", PulsarSearchConfig.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tseg_ffa", types.f8),
        ("poly_order", types.i8),
        ("n_derivs", types.i8),
        ("cheb_table", types.f8[:, :]),
    ],
)
class PruningChebychevDPFunctions:
    def __init__(
        self,
        cfg: PulsarSearchConfig,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tseg_ffa: float,
        poly_order: int = 13,
        n_derivs: int = 6,
    ) -> None:
        self.cfg = cfg
        self.param_arr = param_arr
        self.dparams = dparams
        self.tseg_ffa = tseg_ffa
        self.poly_order = poly_order
        self.n_derivs = n_derivs
        self.cheb_table = math.gen_chebyshev_polys_table(poly_order, n_derivs)

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        """
        Load the data for the given segment index from the folded data structure.

        Parameters
        ----------
        fold : np.ndarray
            The folded data structure to load from.
        seg_idx : int
            Segment index to load from the fold.

        Returns
        -------
        np.ndarray
            The data for the given index.

        Notes
        -----
        Future implementations may include:
        - Simply accessing the data from the fold.
        - Calibration (RFI removal + fold_e, fold_v generation) of the data structure.
        - Compute the data structure live (using dynamic programming).
        - Save the calculated data structure to prevent excessive computation.
        - Remove pulsars with known ephemeris to keep the suggestion counts low.
        - Implement it as a class, and pass its loading function here.

        """
        return fold[seg_idx]

    def resolve(
        self,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, int]:
        coord_add = (coord_add[0] * self.tseg_ffa, coord_add[1] * self.tseg_ffa)
        coord_init = (coord_init[0] * self.tseg_ffa, coord_init[1] * self.tseg_ffa)
        return poly_chebychev_resolve(
            leaf,
            self.param_arr,
            coord_add,
            coord_init,
            self.cfg.nbins,
            self.cheb_table,
        )

    def branch(
        self,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        """
        Branch the current parameter set into the finer grid of parameters (leaves).

        Parameters
        ----------
        param_set : np.ndarray
            The current parameter set to branch.

        Returns
        -------
        np.ndarray
            The branched parameter set.
        """
        coord_cur = (coord_cur[0] * self.tseg_ffa, coord_cur[1] * self.tseg_ffa)
        return poly_chebychev_branch2leaves(
            param_set,
            self.cfg.tol,
            self.cfg.tsamp,
            self.poly_order,
        )

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> common.SuggestionStruct:
        """
        Generate an initial suggestion struct for the starting segment.

        Parameters
        ----------
        fold_segment : np.ndarray
            The folded data segment to generate the suggestion for.
        coord_init : tuple[float, float]
            The coordinate of the starting segment.

        Returns
        -------
        common.SuggestionStruct
            The initial suggestion struct for the segment.
        """
        coord_init = (coord_init[0] * self.tseg_ffa, coord_init[1] * self.tseg_ffa)
        return poly_chebychev_suggestion_struct(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.poly_order,
            coord_init,
            self.score,
        )

    def score(self, combined_res: np.ndarray) -> float:
        """
        Calculate the statistical detection score of the combined fold.

        Parameters
        ----------
        combined_res : np.ndarray
            The combined fold to calculate the score for (fold_e, fold_v).

        Returns
        -------
        float
            The statistical detection score of the combined fold.

        Notes
        -----
        - Score units should be log(P(D|pulsar(theta)) / P(D|noise)).
        - Maybe use it to keep track of a family of scores (profile width, etc).
        """
        return scoring.snr_score_func(combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def validate(
        self,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        """
        Validate which of the leaves are physical.

        Parameters
        ----------
        leaves : np.ndarray
            Set of leaves (parameter sets) to validate.
        coord_valid : np.ndarray
            hacky way to pass min and max
        validation_params : np.ndarray
            Pre-computed validation parameters for the physical validation.

        Returns
        -------
        np.ndarray
            Boolean mask indicating which of the leaves are physical.

        Notes
        -----
        - The validation_params are pre-computed to reduce computation.
        - This function should filter out leafs with unphysical derivatives.
        - pruning scans only functions that are physical at position (t/2).
        But same bounds apply everywhere.
        """
        t_ref_add = self.tseg_ffa * coord_valid[1]
        t_ref_init = self.tseg_ffa * coord_valid[0]
        return poly_chebychev_validate(
            leaves,
            t_ref_add,
            t_ref_init,
            validation_params,
            self.tsegment_ffa,
            self.deriv_bounds,
            self.n_valid,
            self.cheb_table,
            self.period_bounds,
        )

    def transform(
        self,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        coord_cur = (self.tseg_ffa * coord_cur[0], self.tseg_ffa * coord_cur[1])
        return poly_chebychev_transform(leaf, coord_cur, trans_matrix)

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        coord_cur = (self.tseg_ffa * coord_cur[0], self.tseg_ffa * coord_cur[1])
        coord_prev = (self.tseg_ffa * coord_prev[0], self.tseg_ffa * coord_prev[1])
        return poly_chebychev_transform_matrix(
            coord_cur,
            coord_prev,
            self.poly_order,
        )

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Prepare the validation parameters for the epicyclic validation.

        Parameters
        ----------
        coord_add : float
            The Chebyshev time to prepare the validation parameters for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            The validation parameters for the epicyclic validation.
        """
        t_ref = self.tseg_ffa * (coord_add[1] - coord_add[0])
        return prepare_epicyclic_validation_params(
            t_ref,
            self.tsegment_ffa,
            self.num_validation,
            self.omega_bounds,
            self.x_max,
            self.ecc_max,
        )
