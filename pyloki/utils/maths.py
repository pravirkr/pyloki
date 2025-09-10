from __future__ import annotations

import numpy as np
from numba import njit, vectorize
from numpy import polynomial
from scipy import stats

from pyloki.utils import np_utils


def fact_factory(n_tab_out: int = 100) -> np.ufunc:
    fact_tab = np.ones(n_tab_out)

    @njit(cache=True)
    def _fact(num: int, n_tab: int = n_tab_out) -> int:
        if num < n_tab:
            return fact_tab[num]
        ret = 1
        for nn in range(1, num + 1):
            ret *= nn
        return ret

    for ii in range(n_tab_out):
        fact_tab[ii] = _fact(ii, 0)

    @vectorize(cache=True)
    def fact_vec(num: int) -> int:
        return _fact(num)

    return fact_vec


fact = fact_factory(120)


@njit(cache=True, fastmath=True)
def nbinom(n: int, k: int) -> int:
    """Hybrid integer binomial coefficient."""
    if k < 0 or k > n:
        return 0
    if k in (0, n):
        return 1
    # Use symmetry
    k = min(k, n - k)
    # Use factorial for small values, multiplicative for large
    if n <= 120:
        return fact(n) // (fact(k) * fact(n - k))
    # Multiplicative approach for large n
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def gen_norm_isf_table(max_minus_logsf: float, minus_logsf_res: float) -> np.ndarray:
    x_arr = np.arange(0, max_minus_logsf, minus_logsf_res)
    return stats.norm.isf(np.exp(-x_arr))


def gen_chi_sq_minus_logsf_table(
    df_max: int,
    chi_sq_max: float,
    chi_sq_res: float,
) -> np.ndarray:
    x_arr = np.arange(0, chi_sq_max, chi_sq_res)
    table = np.zeros((df_max + 1, len(x_arr)))
    for i in range(1, df_max + 1):
        table[i] = -stats.chi2.logsf(x_arr, i)
    return table


chi_sq_res = 0.5
chi_sq_max = 300
max_minus_logsf = 400
minus_logsf_res = 0.1
chi_sq_minus_logsf_table = gen_chi_sq_minus_logsf_table(64, chi_sq_max, chi_sq_res)
norm_isf_table = gen_norm_isf_table(max_minus_logsf, minus_logsf_res)


@njit(cache=True, fastmath=True)
def norm_isf_func(minus_logsf: float) -> float:
    pos = minus_logsf / minus_logsf_res
    frac_pos = pos % 1
    if minus_logsf < max_minus_logsf:
        return (
            norm_isf_table[int(pos)] * (1 - frac_pos)
            + norm_isf_table[int(pos) + 1] * frac_pos
        )
    return norm_isf_table[-1] * (minus_logsf / max_minus_logsf) ** 0.5


@njit(cache=True, fastmath=True)
def chi_sq_minus_logsf_func(chi_sq_score: float, df: int) -> float:
    tab_pos = chi_sq_score / chi_sq_res
    frac_pos = tab_pos % 1
    if chi_sq_score < chi_sq_max:
        return (
            chi_sq_minus_logsf_table[df, int(tab_pos)] * (1 - frac_pos)
            + chi_sq_minus_logsf_table[df, int(tab_pos) + 1] * frac_pos
        )
    return chi_sq_minus_logsf_table[df, -1] * chi_sq_score / chi_sq_max


def gen_chebyshev_polys_table_np(order_max: int, n_derivs: int) -> np.ndarray:
    tab = np.zeros((n_derivs + 1, order_max + 1, order_max + 1))
    basis = [polynomial.chebyshev.Chebyshev.basis(i) for i in range(order_max + 1)]

    for ideriv in range(n_derivs + 1):
        for iorder, poly in enumerate(basis):
            poly_coeffs = polynomial.chebyshev.cheb2poly(poly.deriv(ideriv).coef)
            tab[ideriv, iorder, : len(poly_coeffs)] = poly_coeffs
    return tab


@njit(cache=True, fastmath=True)
def gen_chebyshev_polys_table(order_max: int, n_derivs: int) -> np.ndarray:
    """Generate Chebyshev polynomials of the first kind and their derivatives.

    This function uses the recurrence relation for Chebyshev polynomials to efficiently
    generate a table of polynomials up to the specified maximum order, along with
    their derivatives up to the specified order.

    Parameters
    ----------
    order_max : int
        The highest polynomial order to generate (T_0, T_1, ..., T_order_max).
    n_derivs : int
        The number of derivatives to compute for each polynomial (0th derivative
        is the polynomial itself).

    Returns
    -------
    np.ndarray
        A 3D array with shape (n_derivs + 1, order_max + 1, order_max + 1)
        containing the coefficients of the polynomials and their derivatives.
        The array is indexed as [i_deriv, i_order, i_coeff].

    Notes
    -----
    Chebyshev polynomials of the first kind are defined by:
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x) for n > 1

    The derivatives are computed using the chain rule and the properties of Chebyshev
    polynomials.
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


@njit(cache=True, fastmath=True)
def gen_design_matrix_taylor(t_vals: np.ndarray, order_max: int) -> np.ndarray:
    """Generate the design matrix for a Taylor series.

    Parameters
    ----------
    t_val : float
        The value at which to evaluate the Taylor series.
    order_max : int
        The maximum order of the Taylor series.

    Returns
    -------
    np.ndarray
        A 2D array (len(t_vals), order + 1) design matrix.
    """
    t_vals = np.atleast_1d(t_vals)
    n_points = len(t_vals)
    mat = np.zeros((n_points, order_max + 1), dtype=np.float32)
    for i, t_val in enumerate(t_vals):
        for j in range(order_max + 1):
            mat[i, j] = t_val**j / fact(j)
    return mat


@njit(cache=True, fastmath=True)
def generalized_cheb_pols(poly_order: int, t0: float, scale: float) -> np.ndarray:
    """Generate a set of generalized Chebyshev polynomials.

    This function computes the coefficients of generalized Chebyshev polynomials,
    which are defined as Chebyshev polynomials with a shifted and scaled domain.

    Parameters
    ----------
    poly_order : int
        The maximum order of the polynomials to generate (T_0 through T_poly_order).
    t0 : float
        Shift parameter or the center of the domain for the polynomials.
    scale : float
        Scale parameter for the domain of the polynomials.

    Returns
    -------
    np.ndarray
        A 2D array of shape (poly_order + 1, poly_order + 1) containing the coefficients
        of the generalized Chebyshev polynomials.
        The array is indexed as [i_order, i_coeff].

    Notes
    -----
    The generalized Chebyshev polynomials are defined as:
    T*_n(y) = T_n((x - t0) / scale)
    where :math: y in [-1, 1] and x in [t0 - scale, t0 + scale].
    """
    cheb_pols = gen_chebyshev_polys_table(poly_order, 0)[0]

    # scale the polynomials
    scale_factor = (1.0 / scale) ** np.arange(poly_order + 1, dtype=np.float32)
    scaled_pols = cheb_pols * scale_factor

    # Shift the origin to t0
    shifted_pols = np.zeros_like(scaled_pols)
    for iorder in range(poly_order + 1):
        for iterm in range(iorder + 1):
            shifted = nbinom(iorder, iterm) * (-t0 / scale) ** (iorder - iterm)
            shifted_pols[iorder, iterm] = shifted
    return np.dot(scaled_pols, shifted_pols)


def gen_power_series_table_np(order_max: int, n_derivs: int) -> np.ndarray:
    tab = np.zeros((n_derivs + 1, order_max + 1, order_max + 1))

    for order in range(order_max + 1):
        # Create power series polynomial
        coeffs = np.zeros(order + 1)
        coeffs[order] = 1 / np.math.factorial(order)
        poly = polynomial.Polynomial(coeffs)

        for deriv in range(n_derivs + 1):
            # Get coefficients of the derivative
            deriv_coeffs = poly.deriv(deriv).coef
            # Pad with zeros if necessary
            tab[deriv, order, : len(deriv_coeffs)] = deriv_coeffs

    return tab


@njit(cache=True, fastmath=True)
def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1) == 0)


@njit(cache=True, fastmath=True)
def compute_connection_coefficient_s(k: int, m: int) -> float:
    """Compute the connection coefficient S_{k,m}.

    Parameters
    ----------
    k : int
        Power series exponent, x^k.
    m : int
        Order of the Chebyshev polynomial, T_m(x).

    Returns
    -------
    float
        Connection coefficient S_{k,m}.
    """
    # Check if k-m is even and m <= k
    if k < 0 or m < 0 or m > k or (k - m) % 2 != 0:
        return 0.0
    n = (k - m) // 2
    deltam0 = 1 if m == 0 else 0
    # float 2.0 is required for numba
    return 2.0 ** (1 - k - deltam0) * nbinom(k, n)


@njit(cache=True, fastmath=True)
def compute_connection_coefficient_r(k: int, m: int) -> float:
    """Compute the connection coefficient R_{k,m}.

    Parameters
    ----------
    k : int
        Order of the Chebyshev polynomial, T_k(x).
    m : int
        Power series exponent, x^m.

    Returns
    -------
    float
        Connection coefficient R_{k,m}.
    """
    if k < 0 or m < 0 or m > k or (k - m) % 2 != 0:
        return 0.0
    if m == 0 and k % 2 == 0:
        return (-1) ** (k // 2)
    n = (k - m) // 2
    r = (k + m) // 2
    return ((-1) ** n) * (2.0 ** (m - 1)) * (2.0 * k / (k + m)) * nbinom(r, n)


@njit(cache=True, fastmath=True)
def compute_transformation_coefficient_c(n: int, k: int, p: float, q: float) -> float:
    """Compute the transformation coefficient C_{n,k}(p,q).

    Parameters
    ----------
    n : int
        Order of the source Chebyshev polynomial, T_n(x).
    k : int
        Order of the target Chebyshev polynomial, T_k(x).
    p : float
        Scale factor ratio b/d.
    q : float
        Translation factor (a-c)/d.

    Returns
    -------
    float
        Transformation coefficient C_{n,k}(p,q) for Chebyshev basis change.
    """
    if k > n:
        return 0.0
    result = 0.0
    for m in range(k, n + 1):
        r_nm = compute_connection_coefficient_r(n, m)
        inner_sum = 0.0
        for i in range(k, m + 1):
            s_ik = compute_connection_coefficient_s(i, k)
            term = nbinom(m, i) * p**i * q ** (m - i) * s_ik
            inner_sum += term
        result += r_nm * inner_sum
    return result


@njit(cache=True, fastmath=True)
def compute_connection_matrix_s(k_max: int) -> np.ndarray:
    """Compute the connection coefficient matrix S.

    Parameters
    ----------
    k_max : int
        Maximum order for both power series and Chebyshev polynomials.

    Returns
    -------
    np.ndarray
        Matrix S where S[k,m] contains the connection coefficient S_{k,m}.
        Shape is (k_max + 1, k_max + 1).
        Lower triangular matrix since S_{k,m} = 0 for m > k.
    """
    if k_max < 0:
        msg = "k_max must be a non-negative integer."
        raise ValueError(msg)
    s_mat = np.zeros((k_max + 1, k_max + 1), dtype=np.float64)
    for k in range(k_max + 1):
        for m in range(k + 1):
            s_mat[k, m] = compute_connection_coefficient_s(k, m)
    return s_mat


@njit(cache=True, fastmath=True)
def compute_connection_matrix_r(k_max: int) -> np.ndarray:
    """Compute the connection coefficient matrix R.

    Parameters
    ----------
    k_max : int
        Maximum order for both power series and Chebyshev polynomials.

    Returns
    -------
    np.ndarray
        Matrix R where R[k,m] contains the connection coefficient R_{k,m}.
        Shape is (k_max + 1, k_max + 1).
        Lower triangular matrix since R_{k,m} = 0 for m > k.
    """
    r_mat = np.zeros((k_max + 1, k_max + 1), dtype=np.float64)
    for k in range(k_max + 1):
        for m in range(k + 1):
            r_mat[k, m] = compute_connection_coefficient_r(k, m)
    return r_mat


@njit(cache=True, fastmath=True)
def poly_chebyshev_transform_matrix(
    poly_order: int,
    tc1: float,
    ts1: float,
    tc2: float,
    ts2: float,
) -> np.ndarray:
    """Compute the transformation coefficient matrix C.

    Parameters
    ----------
    poly_order : int
        Maximum order for Chebyshev bases (k_max).
    tc1 : float
        Center of the domain for the input Chebyshev polynomials.
    ts1 : float
        Scale factor for the input Chebyshev polynomials.
    tc2 : float
        Center of the domain for the output Chebyshev polynomials.
    ts2 : float
        Scale factor for the output Chebyshev polynomials.

    Returns
    -------
    np.ndarray
        Matrix C where C[n,k] contains the transformation coefficient C_{n,k}(p,q).
        Shape is (k_max + 1, k_max + 1).
    """
    k_max = poly_order
    if k_max < 0:
        msg = "k_max must be a non-negative integer."
        raise ValueError(msg)
    if ts1 <= 0 or ts2 <= 0:
        msg = "ts1 and ts2 must be positive."
        raise ValueError(msg)
    p = ts2 / ts1
    q = (tc2 - tc1) / ts1
    c_mat = np.zeros((k_max + 1, k_max + 1), dtype=np.float64)
    for n in range(k_max + 1):
        for k in range(k_max + 1):
            c_mat[n, k] = compute_transformation_coefficient_c(n, k, p, q)
    return c_mat


@njit(cache=True, fastmath=True)
def find_small_polys(
    degree: int,
    error_bound: int,
    oversampling: int = 4,
    npoints: int = 128,
    max_violations: int = 12,
) -> tuple[list, float, float]:
    """Find small polynomials in the phase space.

    Enumerate the phase space of all polynomials up to a certain resolution to find the
    volume of small polynomials. The volume roughly fits the expected volume from using
    the basis of Chebyshev polynomials.

    Parameters
    ----------
    degree : int
        Maximum degree of the polynomials.
    error_bound : int
        Maximum allowed deviation from zero.
    oversampling : int, optional
        Factor to increase density of coefficient sampling, by default 4.
    npoints : int, optional
        Number of points to evaluate the polynomials, by default 128.
    max_violations : int, optional
        Maximum number of points allowed to exceed the error bound, by default 12.

    Returns
    -------
    tuple[list, float, float]
        - good_poly: List of small polynomials.
        - point_volume: Volume of a single point in the phase space.
        - volume_factor: Factor to scale the point volume to the total volume.
    """
    test_points = np.linspace(-1, 1, npoints)
    phase_space = np_utils.cartesian_prod(
        [
            np.linspace(
                -(degree - 1) * error_bound,
                (degree - 1) * error_bound,
                2 * (degree - 1) * oversampling,
            )
            for _ in range(degree)
        ],
    )
    point_volume = (2 * (degree - 1)) ** degree / len(phase_space)
    x_matrix = test_points[:, np.newaxis] ** np.arange(degree)
    poly_values = np.dot(x_matrix, phase_space.T)
    # Find good polynomials
    good_poly_mask = np.sum(np.abs(poly_values) > error_bound, axis=0) < max_violations
    good_poly = phase_space[good_poly_mask]
    volume_factor = point_volume * len(good_poly) / 2**degree
    return good_poly, point_volume, volume_factor
