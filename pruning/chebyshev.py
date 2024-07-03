import numpy as np
from numba import njit, types

from pruning import defaults, kernels, math, scores, utils


@njit
def cheb_step(poly_order: int, tsamp: float, tol: int) -> np.ndarray:
    return np.zeros(poly_order + 1, np.float32) + ((tol * tsamp) * utils.c_val)


@njit
def err_epicycle_fit_fast(
    values: np.ndarray,
    mat_list_left: np.ndarray,
    mat_list_right: np.ndarray,
) -> float:
    min_err = np.inf
    for i in range(mat_list_right.shape[0]):
        x = np.dot(mat_list_right[i], values)
        err_vec = np.dot(mat_list_left[i], x) - values
        err = np.max(np.abs(err_vec))
        if err < min_err:
            min_err = err
    return min_err


@njit
def split_cheb_params(
    cheb_coeffs_cur: np.ndarray,
    dcheb_opt: np.ndarray,
    dcheb_cur: np.ndarray,
    tol: float,
) -> np.ndarray:
    n_coeffs = len(dcheb_cur)

    effective_tol = tol * pulsar_utils.C
    leafs = np.zeros((0, 0, 0))
    for i in range(n_coeffs):
        if abs(dcheb_cur[i]) > effective_tol:
            if dcheb_opt[i] > 0.5 * dcheb_cur[i]:
                dcheb_opt[i] = 0.5 * dcheb_cur[i]
            par_array, act_dpar = branch_param(
                dcheb_opt[i], dcheb_cur[i], cheb_coeffs_cur[i]
            )
        else:
            par_array = np.array((cheb_coeffs_cur[i],))
            act_dpar = dcheb_cur[i]
        leafs = extend_leaf_params(leafs, par_array, act_dpar)
    return leafs


# TODO Make the branch function more efficient (carefull, a very delicate code!)
@njit
def extend_leaf_params(leafs, par_array, act_dpar):
    n_par = len(par_array)
    if len(leafs) == 0:
        new_leafs = np.zeros((len(par_array), 1, 2))
        new_leafs[:, 0, 0] = par_array
        new_leafs[:, 0, 1] = act_dpar
    else:
        new_leafs = np.zeros((len(leafs) * len(par_array), leafs.shape[1] + 1, 2))
        for leaf_ind in range(len(leafs)):
            leaf = leafs[leaf_ind]
            for par_ind, par in enumerate(par_array):
                new_leafs[leaf_ind * n_par + par_ind][:-1] = leaf
                new_leafs[leaf_ind * n_par + par_ind][-1, 0] = par
                new_leafs[leaf_ind * n_par + par_ind][-1, 1] = act_dpar
    return new_leafs


def poly_factory(params):
    return lambda x: np.sum([x**i * params[i] for i in range(len(params))], 0)


@njit
def evaluate_poly(t_arr, pol_coeffs):
    t_pow = t_arr**0
    s = pol_coeffs[0] * t_pow
    for i in range(1, len(pol_coeffs)):
        t_pow *= t_arr
        s += pol_coeffs[i] * t_pow
    return s


@njit
def get_diagonal(A):
    diag = np.zeros(A.shape[0])
    for i in range(len(diag)):
        diag[i] = A[i, i]
    return diag


@njit
def effective_degree(pol_coeffs, eps):
    return np.max((np.abs(pol_coeffs) > eps) * np.arange(len(pol_coeffs)))


# Enumerating the phase space of all polynomials up to a certain resolution, to find the volume of small polynomials
# The volume roughly fits the expected volume from using the basis of chebychev polynomials.
def find_small_polys(deg, err, over_samp=4):
    test_range = np.linspace(-1, 1, 128)
    phase_space = utils.cartesian_prod(
        [
            np.linspace(-(deg - 1) * err, (deg - 1) * err, 2 * (deg - 1) * over_samp)
            for d in range(deg)
        ]
    )
    point_volume = (2 * (deg - 1)) ** deg / float(len(phase_space))
    good_poly = []
    for i, p_params in enumerate(phase_space):
        if i % 10000 == 0:
            print(i, "/", len(phase_space))
        p = poly_factory(p_params)
        if sum(abs(p(test_range)) > err) < 12:
            good_poly.append(p_params)
    print("volume factor:", point_volume * len(good_poly) / 2**deg)
    return good_poly, point_volume


@njit
def generate_chebyshev_polys_table(order_max: int, n_derivs: int) -> np.ndarray:
    """Generate table of Chebyshev polynomials of the first kind and their derivatives.

    Parameters
    ----------
    order_max : int
        The maximum order to generate the polynomials for (T_0, T_1, ..., T_order_max)
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


@njit
def generalized_cheb_pols(poly_order: int, scale: float, t0: float) -> np.ndarray:
    cheb_pols = generate_chebyshev_polys_table(poly_order, 0)[0]

    # Shift the origin to t0
    cheb_pols_shifted = np.zeros_like(cheb_pols)
    for iorder in range(poly_order + 1):
        iterms = np.arange(iorder + 1, dtype=np.float32)
        shifted = math.nbinom(iorder, iterms) * (-t0 / scale) ** (iorder - iterms)
        cheb_pols_shifted[iorder, : shifted.size] = shifted
    pols = np.dot(cheb_pols, cheb_pols_shifted)

    # scale the polynomials
    pols *= scale ** (-np.arange(pols.shape[1], dtype=np.float32))
    return pols


@njit
def get_leaves_chebyshev(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    scale: float,
    t0: float,
) -> np.ndarray:
    conversion_matrix = np.linalg.inv(generalized_cheb_pols(poly_order, scale, t0))
    params_vec = np.zeros(poly_order + 1)
    # Assign param_arr to the correct position in the params_vec
    params_vec[2] = acc / 2.0
    params_vec_conv = np.dot(conversion_matrix.T, params_vec)

    alpha_vec = np.zeros((poly_order + 3, 2))
    alpha_vec[:-2, 0] = params_vec_conv[:]
    alpha_vec[2:-2, 1] = dparams[1:] * np.diag(conversion_matrix)[2:]

    alpha_vec[0, 0] = params_vec_conv[0] + (t0 % p) * utils.c_val
    alpha_vec[0, 1] = 0.0
    alpha_vec[1, 1] = dparams[0] / p * utils.c_val
    # second from last parameter is p0
    alpha_vec[-2, 0] = p
    alpha_vec[-2, 1] = 0
    # The convention is that the last parameter is the reference point
    # (point in time in which t=0) and last parameter "error" is the scale
    alpha_vec[-1, 0] = t0
    alpha_vec[-1, 1] = scale

    return alpha_vec


@njit
def suggestion_struct_chebyshev(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    t_ref: float,
    scale: float,
    score_func: types.FunctionType,
) -> kernels.SuggestionStruct:
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    # \n_param_sets = n_accel * n_period
    # \param_sets_shape = [n_param_sets, poly_order + 3, 2]
    param_sets = get_leaves_chebyshev(param_arr, dparams, poly_order, scale, t_ref)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)))
    return kernels.SuggestionStruct(param_sets, data, scores, backtracks)


@njit
def poly_chebychev_branch(
    param_set: np.ndarray,
    poly_order: int,
    tol_bins: float,
    tsamp: float,
) -> np.ndarray:
    cheb_coeffs_cur = param_set[0:-2, 0]
    dcheb_cur = param_set[0:-2, 1]
    p0 = param_set[-2, 0]
    t0 = param_set[-1, 0]
    scale = param_set[-1, 1]

    dcheb_opt = cheb_step(poly_order, tsamp, tol_bins)
    leafs_cheb = split_cheb_params(
        cheb_coeffs_cur,
        dcheb_opt,
        dcheb_cur,
        tol_bins * tsamp,
    )

    n_cheb = len(leafs_cheb)

    ind = 0
    leaves = np.zeros((n_cheb, poly_order + 3, 2))
    for i_cheb in range(n_cheb):
        leaves[ind][0:-2] = leafs_cheb[i_cheb]
        leaves[ind][-2, 0] = p0
        leaves[ind][-1, 0] = t0
        leaves[ind][-1, 1] = scale
        ind += 1
    return leaves


@njit
def gen_transfer_matrix(
    poly_order: int, scale0: float, t0: float, scale1: float, t1: float
) -> np.ndarray:
    cheb_pols1 = generalized_cheb_pols(poly_order, scale0, t0)
    cheb_pol2 = generalized_cheb_pols(poly_order, scale1, t1)
    return np.dot(cheb_pols1, np.linalg.inv(cheb_pol2))


@njit
def chebychev_poly_evaluate(cheb_table, t_minus_t0, param_set, deriv_index, eff_deg=-3):
    # effective degree is -3 by default as this value reproduces (after +1) the -2 (which is the last coeff by convention)
    tab = cheb_table[deriv_index]
    # t0 = param_set[-1,0]
    scale = param_set[-1, 1]
    # print t_minus_t0 / scale
    coeffs = param_set[0 : eff_deg + 1, 0]
    # pol = tab[0] * param_set[0,0]
    if eff_deg < 0:
        eff_deg = len(coeffs) - 1
    pol = np.zeros(eff_deg + 1)
    for i in range(len(coeffs)):
        pol += coeffs[i] * tab[i][: eff_deg + 1]
    # print pol/pulsar_utils.C
    return (evaluate_poly(t_minus_t0 / scale, pol)) / scale**deriv_index


def poly_chebychev_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    tchunk_ref: float,
    tchunk_cur: float,
    nbins: int,
    cheb_table: np.ndarray,
) -> tuple[np.ndarray, float]:
    t_ref = pset_cur[-1, 0]
    # bad name, this is the time of the next local param prediction to be made
    tcheby = tchunk_cur - t_ref
    # reference time.
    zero_time = tchunk_ref - t_ref

    p0 = pset_cur[-2, 0]
    eff_degree = effective_degree(pset_cur[:-2, 0], 1000)
    new_p = p0 * (
        1
        - (
            chebychev_poly_evaluate(cheb_table, tcheby, pset_cur, 1, eff_degree)
            - chebychev_poly_evaluate(cheb_table, zero_time, pset_cur, 1, eff_degree)
        )
        / utils.C
    )
    new_a = chebychev_poly_evaluate(cheb_table, tcheby, pset_cur, 2, eff_degree)

    delay_tcheby = chebychev_poly_evaluate(cheb_table, tcheby, pset_cur, 0, eff_degree)
    delay_zero = chebychev_poly_evaluate(cheb_table, zero_time, pset_cur, 0, eff_degree)
    delay = (delay_tcheby - delay_zero) / utils.c_val

    relative_phase = kernels.get_phase_idx_period(tchunk_cur, p0, nbins, delay)
    old_a_index = kernels.find_nearest_sorted_idx(parr_prev[0], new_a)
    old_p_index = kernels.find_nearest_sorted_idx(parr_prev[1], new_p)

    return (old_a_index, old_p_index), relative_phase


def poly_chebychev_coord_trans_matrix(
    data_access_scheme: np.ndarray,
    poly_order: int,
    chunk_duration: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    t_ref_old = chunk_duration * (
        (np.min(data_access_scheme[:-1]) + np.max(data_access_scheme[:-1]) + 1) / 2.0
    )
    t_ref_new = chunk_duration * (
        (np.min(data_access_scheme) + np.max(data_access_scheme) + 1) / 2.0
    )
    scale_old = t_ref_old - chunk_duration * np.min(data_access_scheme[:-1])
    scale_new = t_ref_new - chunk_duration * np.min(data_access_scheme)

    transfer_matrix = gen_transfer_matrix(
        poly_order,
        scale_old,
        t_ref_old,
        scale_new,
        t_ref_new,
    )
    return transfer_matrix, (t_ref_new, scale_new)


@njit
def poly_chebychev_coord_trans(
    trans_matrix: np.ndarray,
    coord_params: np.ndarray,
    leaf_param: np.ndarray,
) -> np.ndarray:
    p = leaf_param[-2, 0]
    params = leaf_param[0:-2, 0]
    d_params = leaf_param[0:-2, 1]
    new_suggestion = np.zeros(leaf_param.shape)
    params *= np.abs(params) > 1e5
    params_new = np.dot(params, trans_matrix)
    # Choose between the volume treatment approaches. each has it own advantages
    # \d_params_new = np.sqrt(np.dot(A**2, d_params**2)) # Conservative approach
    d_params_new = np.diag(trans_matrix) * d_params  # Violent ignorance approach
    # \d_params_new = A.diagonal() * d_params
    new_suggestion[:-2, 0] = params_new
    new_suggestion[:-2, 1] = d_params_new
    new_suggestion[-2, 0] = p
    new_suggestion[-2, 1] = 0

    new_suggestion[-1, 0] = coord_params[0]
    new_suggestion[-1, 1] = coord_params[1]
    new_suggestion[0, 0] = params_new[0]  # + (time_shift % p) * pulsar_utils.C
    # new_suggestion[0, 1] = bin_duration
    return new_suggestion


@njit
def poly_chebychev_physical_validation(
    leaves: np.ndarray,
    indices_arr: np.ndarray,
    validation_params,
    chunk_duration,
    derivative_bounds,
    n_validation_points: int,
    cheb_table: np.ndarray,
    period_bounds,
):
    nleaves = len(leaves)
    mask = np.zeros(nleaves, np.bool_)
    t0 = leaves[0, -1, 0]
    total_duration = np.max(indices_arr) * chunk_duration - t0
    zero_time = np.min(indices_arr) * chunk_duration - t0
    # Important not to check on the segment edges because quantization effects
    # may cause unphysical derivatives that are later corrected.
    time_arr = np.linspace(
        zero_time + 3 * chunk_duration / 2.0,
        total_duration - chunk_duration / 2.0,
        n_validation_points,
    )
    for ileaf in range(nleaves):
        param_set = leaves[ileaf]
        eff_degree = effective_degree(param_set[:-2, 0], 1000)
        values = chebychev_poly_evaluate(
            cheb_table,
            time_arr,
            param_set,
            0,
            eff_degree,
        )
        med = np.median(values)
        st = np.std(values)
        max_diff = np.max(values) - med
        min_diff = med - np.min(values)
        if (
            max_diff < 3.2 * st
            and min_diff < 3.2 * st
            and (
                err_epicycle_fit_fast(
                    values, validation_params[0], validation_params[1]
                )
                < validation_params[2]
            )
        ):
            p0 = param_set[-2, 0]
            good = True
            for deriv_index in range(1, len(derivative_bounds)):
                values = chebychev_poly_evaluate(
                    cheb_table,
                    time_arr,
                    param_set,
                    deriv_index,
                    eff_degree,
                )
                if deriv_index == 1:
                    if (np.max(values) - np.min(values)) > 2 * derivative_bounds[
                        deriv_index
                    ]:
                        good = False
                        break
                    if (((1 - np.min(values) / C) * p0) < period_bounds[0]) or (
                        ((1 - np.max(values) / C) * p0) > period_bounds[1]
                    ):
                        good = False
                        break

                elif np.max(np.abs(values)) > derivative_bounds[deriv_index]:
                    good = False
                    break

        mask[ileaf] = good

    return mask


def chebychev_search_pruning(
    dyps,
    dt,
    tolerance_time,
    x_orbit,
    P_orbit_min,
    max_eccentricity=0.2,
    poly_order=12,
    n_max_suggestions=2**15,
    sigma_clip=False,
    n_box_conv=16,
    n_validation_points=32,
    n_validation_derivatives=4,
    pulse_profile="double",
):
    # Regular case, one dynamic programming initialization class per Pruning object
    if type(dyps) == DynamicProgramming:
        dyp = dyps
        N_dyps = 1
    # When data is too long to hold in memory, and the memory reduction performed by dyp (limited period range)
    # is critical, there are several dyps per class instance, each holding a fraction of the data.
    else:
        dyp = dyps[0]
        N_dyps = len(dyps)

    n_chunks = dyp.data_structure.shape[0] * N_dyps
    n_bins = dyp.data_structure.shape[-1]

    period_min, period_max = dyp.param_limits[1]

    param_limits = compute_param_limits_chebychev(x_orbit, P_orbit_min, poly_order)

    chunk_duration = 2**dyp.iter_num * dyp.brute_length * dt
    param_list = dyp.param_list

    omega_orbit_max = (2 * np.pi) / P_orbit_min
    omega_orbit_min = (2 * np.pi) / (P_orbit_min * 20)
    a_max = x_orbit * omega_orbit_max**2 * pulsar_utils.C
    a_min = -a_max

    da = dyp.param_steppings[0]
    da = np.min([da, a_max - a_min])
    dp = dyp.param_steppings[1]
    C = pulsar_utils.C
    # TODO understand why division by 2**(k-1) is not applied (through the trans_matrix) in the conversion from acc to cheb
    # TODO BUG!! Eccentricity did not effect the dparams!
    # TODO BUG!! Seems like the derivatives do not reflect the actual quality of the polynomial approximations. Errors are HUGE, and so is the potential improvement...
    # dparams = np.array([dp, da] + [2*x_orbit * (omega_orbit_max)**k / fact(k) * C for k in range(3,poly_order+1)])
    derivative_bounds = pulsar_utils.find_max_deriv_bounds(
        x_orbit,
        chunk_duration * n_chunks / P_orbit_min * 2 * np.pi,
        max_eccentricity,
        max(poly_order, n_validation_derivatives) + 1,
        omega_orbit_max,
    )[0]
    dparams = np.array(
        [dp, da]
        + [2 * derivative_bounds[k] / fact(k) for k in range(3, poly_order + 1)]
    )
    # derivative_bounds = np.array([2*x_orbit*(omega_orbit_max)**k * C for k in range(0,n_validation_derivatives+1)])
    # print 2,x_orbit, omega_orbit_max,max_eccentricity
    # derivative_bounds = np.array([pulsar_utils.keplerian_derivative_bounds(deriv_index,x_orbit, omega_orbit_max,max_eccentricity)
    #                              for deriv_index in range(0,n_validation_derivatives+1)])

    print("dparams =", dparams)
    print("derivative bounds=", derivative_bounds)

    data_access_scheme = snail_access_scheme(n_chunks, 0)

    if N_dyps == 1:
        data_structure = dyp.data_structure  # ****
    else:
        data_structure = np.concatenate([d.data_structure for d in dyps], 0)  # ****

    effective_param_list = pad_with_inf(dyp.param_list)

    cheb_table = pulsar_utils.generate_chebyshev_polys_table(
        poly_order, n_validation_derivatives
    )

    globals()["YYY"] = [
        chunk_duration,
        derivative_bounds,
        n_validation_points,
        cheb_table,
    ]
    period_bounds = np.array([period_min, period_max])

    # Chebychev representation should allow to work with much fewer suggestions

    target_snr = 12.0
    snr_margin = 2.0

    threshold_scheme_max = (
        target_snr / sqrt(n_chunks) * np.sqrt(np.arange(n_chunks)) - snr_margin
    )

    threshold_scheme = threshold_scheme[: len(threshold_scheme)] + list(
        threshold_scheme_max[len(threshold_scheme) :]
    )
    threshold_scheme = np.array(threshold_scheme[:n_chunks])
    reference_profile = gaussian_pulse_profiles[0]

    prn = Pruning(
        data_structure_loader_func,
        make_suggestion_func,
        get_phase_func,
        resolve_func,
        score_func,
        branch_func,
        physical_validation_func,
        prepare_validation_params_func,
        addition_func,
        shift_and_rebin_func,
        aggregate_score_statistics_func,
        dynamically_update_threshold_func,
        coord_trans_func,
        prepare_coord_transfer_matrix_func,
        log_func,
        n_max_suggestions,
        reference_profile,
    )
    prn.prepare_for_pruning(threshold_scheme, data_access_scheme, data_structure)

    prn.tolerance_time = tolerance_time
    prn.chunk_duration = chunk_duration
    prn.poly_order = poly_order
    prn.derivative_bounds = derivative_bounds
    prn.n_bins = n_bins

    return prn


@jitclass(
    spec=[
        ("params", SearchParams.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tchunk_current", types.f8),
    ]
)
class PruningChebychevDPFunctions:
    def __init__(
        self, params: SearchParams, param_arr, dparams, tchunk_current, cheb_table
    ):
        self.params = params
        self.param_arr = param_arr
        # dparam = np.array([[self.ds, self.dj, self.da, self.dp]])
        self.dparams = dparams
        self.tchunk_current = tchunk_current
        self.cheb_table = cheb_table

    def load(self, fold: np.ndarray, index: int) -> np.ndarray:
        return fold[index]

    def resolve(
        self,
        param_set: np.ndarray,
        idx_dist: int,
        param_ref_ind: int,
    ) -> tuple[np.ndarray, float]:
        tchunk_cur = idx_dist * self.tchunk_ffa
        tchunk_ref = param_ref_ind * self.tchunk_ffa
        return poly_chebychev_resolve(
            param_set,
            self.param_arr,
            tchunk_ref,
            tchunk_cur,
            self.param.nbins,
            self.cheb_table,
        )

    def branch(self, param_set: np.ndarray) -> np.ndarray:
        return poly_chebychev_branch(
            param_set,
            self.params.poly_order,
            self.params.tol_bins,
            self.params.tsamp,
        )

    def suggest(
        self,
        fold_segment: np.ndarray,
        reference_index: int,
    ) -> kernels.SuggestionStruct:
        t0 = self.tchunk_current * (reference_index + 1 / 2.0)
        scale = self.tchunk_current / 2.0
        return suggestion_struct_chebyshev(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.poly_order,
            scale,
            t0,
            self.score_func,
        )

    def score_func(self, combined_res: np.ndarray) -> float:
        return scores.snr_score_func(combined_res)

    def validate(self, data: np.ndarray) -> bool:
        return defaults.validate(data)

    def validation_params(self, indices_arr):
        # prepare_epicyclic_validation_params
        # chunk_duration, n_validation_points, omega_min, omega_max, x_max, ecc_max
        total_duration = (np.max(indices_arr) - np.min(indices_arr)) * chunk_duration
        time_arr = np.linspace(
            3 * chunk_duration / 2.0,
            total_duration - chunk_duration / 2.0,
            n_validation_points,
        )
        fit_mat_left = []
        fit_mat_right = []
        n = len(time_arr)
        epicycle_bound = 2 * x_max * ecc_max**2 * pulsar_utils.C
        d_omega = ecc_max**2 / (time_arr[-1] - time_arr[0])
        omega_arr = np.arange(omega_min, omega_max, d_omega)
        for om in omega_arr:
            mat = np.zeros((6, n), np.float64)
            mat[0, :] = 1
            mat[1, :] = time_arr
            mat[2, :] = np.sin(om * time_arr)
            mat[3, :] = np.cos(om * time_arr)
            mat[4, :] = np.sin(2 * om * time_arr)
            mat[5, :] = np.cos(2 * om * time_arr)
            D = np.dot(mat, mat.T)
            D = np.linalg.inv(D)
            fit_mat_right.append(np.dot(D, mat))
            fit_mat_left.append(mat.T)
        print(
            "shape of fit matrices: ",
            np.array(fit_mat_left).shape,
            np.array(fit_mat_right).shape,
        )
        return np.array(fit_mat_left), np.array(fit_mat_right), epicycle_bound

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def aggregate_stats(self, scores):
        return defaults.aggregate_stats(scores)
