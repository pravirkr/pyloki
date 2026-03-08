from __future__ import annotations

import numpy as np
from numba import njit

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def shift_taylor_params(taylor_param_vec: np.ndarray, delta_t: float) -> np.ndarray:
    """Shift the kinematic Taylor parameters to a new reference time.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (n_batch, n_poly) at reference time t_i.
        Ordering is [d_k_max, ..., d_1, d_0] where d_k is coefficient of (t - t_c)^k/k!.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, n_poly) at the new reference time t_j.
    """
    n_batch, n_poly = taylor_param_vec.shape
    poly_order = n_poly - 1
    t_mat = maths.poly_taylor_transform_matrix(poly_order, delta_t)

    # Transform polynomial coefficients
    out = np.empty_like(taylor_param_vec)
    for i_batch in range(n_batch):
        # Parameters transformation (taylor_coeffs @ t_mat)
        for i in range(n_poly):
            acc = 0.0
            for j in range(n_poly):
                acc += taylor_param_vec[i_batch, j] * t_mat[j, i]
            out[i_batch, i] = acc
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_params_1d(taylor_param_vec: np.ndarray, delta_t: float) -> np.ndarray:
    n_poly = taylor_param_vec.shape[-1]
    poly_order = n_poly - 1
    t_mat = maths.poly_taylor_transform_matrix(poly_order, delta_t)

    # Transform polynomial coefficients
    out = np.empty_like(taylor_param_vec)
    # Parameters transformation (taylor_coeffs @ t_mat)
    for i in range(n_poly):
        acc = 0.0
        for j in range(n_poly):
            acc += taylor_param_vec[j] * t_mat[j, i]
        out[i] = acc
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_basis(leaf_bases_batch: np.ndarray, delta_t: float) -> None:
    """Shift (In-place) the Taylor Lattice basis to a new reference time.

    Parameters
    ----------
    leaf_bases_batch : np.ndarray
        Basis matrix of shape (..., n_params, n_params) at reference time t_i.
        Diagonal ordering is [d_k_max, ..., d_1].
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    """
    n_leaves, n_params, _ = leaf_bases_batch.shape
    t_mat = maths.poly_taylor_transform_matrix(n_params, delta_t)

    # Transform basis
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)
    for leaf in range(n_leaves):
        # Basis transformation (t_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = 0.0
                for m in range(n_params):
                    acc += t_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def shift_taylor_params_basis(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    delta_t: float,
) -> None:
    """Shift (In-place) the kinematic Taylor parameters and basis to a new reference.

    Parameters
    ----------
    leaf_params_batch : np.ndarray
        Parameter vector of shape (..., n_params) at reference time t_i.
        Ordering is [d_k_max, ..., d_1, d_0] where d_k is coefficient of (t - t_c)^k/k!.
    leaf_bases_batch : np.ndarray
        Basis matrix of shape (..., n_params, n_params) at reference time t_i.
        Diagonal ordering is [d_k_max, ..., d_1].
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    """
    n_leaves, n_params, _ = leaf_bases_batch.shape
    t_mat = maths.poly_taylor_transform_matrix(n_params, delta_t)

    # Transform polynomial coefficients
    coeffs_tmp = np.empty(n_params + 1, dtype=np.float64)
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)

    for leaf in range(n_leaves):
        # Parameters transformation (taylor_coeffs @ t_mat)
        for i in range(n_params + 1):
            coeffs_tmp[i] = leaf_params_batch[leaf, i]
        for i in range(n_params + 1):
            acc = 0.0
            for j in range(n_params + 1):
                acc += coeffs_tmp[j] * t_mat[j, i]
            leaf_params_batch[leaf, i] = acc

        # Basis transformation (t_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = 0.0
                for m in range(n_params):
                    acc += t_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def shift_taylor_params_d_f(
    param_vec: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, float]:
    """Shift the kinematic Taylor parameters (with frequency) to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector of shape (n_params,) at reference time t_i.
        Ordering is [..., j, a, f]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    tuple[np.ndarray, float]
        Parameter vector of shape (n_params,) at the new reference time t_j
        and the phase delay d in light-seconds.

    """
    n_params = param_vec.shape[-1]
    taylor_param_vec = np.zeros(n_params + 1, dtype=param_vec.dtype)
    taylor_param_vec[:-2] = param_vec[:-1]  # till acceleration
    taylor_param_vec_new = shift_taylor_params_1d(taylor_param_vec, delta_t)
    param_vec_new = param_vec.copy()
    param_vec_new[:-1] = taylor_param_vec_new[:-2]
    param_vec_new[-1] = param_vec[-1] * (1 - taylor_param_vec_new[-2] / C_VAL)
    delay_rel = taylor_param_vec_new[-1] / C_VAL
    return param_vec_new, delay_rel


@njit(cache=True, fastmath=True)
def shift_taylor_params_d_f_batch(
    param_vec_batch: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch version of shift_taylor_params_d_f."""
    n_batch, n_params = param_vec_batch.shape
    taylor_param_vec = np.zeros((n_batch, n_params + 1), dtype=param_vec_batch.dtype)
    taylor_param_vec[:, :-2] = param_vec_batch[:, :-1]  # till acceleration
    taylor_param_vec_new = shift_taylor_params(taylor_param_vec, delta_t)
    param_vec_new = param_vec_batch.copy()
    param_vec_new[:, :-1] = taylor_param_vec_new[:, :-2]
    param_vec_new[:, -1] = param_vec_batch[:, -1] * (
        1 - taylor_param_vec_new[:, -2] / C_VAL
    )
    delay_rel = taylor_param_vec_new[:, -1] / C_VAL
    return param_vec_new, delay_rel


@njit(cache=True, fastmath=True)
def taylor_to_circular_full(param_sets: np.ndarray) -> np.ndarray:
    """Transform the kinematic Taylor coefficients to circular orbit parameters.

    Parameters
    ----------
    param_sets : np.ndarray
        The Taylor parameters to convert. Shape is (nparams, 2).
        params: [snap, jerk, accel, freq]
        dparams: [dsnap, djerk, daccel, dfreq]

    Returns
    -------
    np.ndarray
        The circular parameters. Shape is (nparams, 2).
        params: [omega, freq, x_cos_phi, x_sin_phi]
        dparams: [domega, dfreq, dx_cos_phi, dx_sin_phi]
    """
    snap, jerk, accel, freq = (
        param_sets[:, 0, 0],
        param_sets[:, 1, 0],
        param_sets[:, 2, 0],
        param_sets[:, 3, 0],
    )
    dsnap, djerk, daccel, dfreq = (
        param_sets[:, 0, 1],
        param_sets[:, 1, 1],
        param_sets[:, 2, 1],
        param_sets[:, 3, 1],
    )
    omega_sq = -snap / accel
    out = np.empty_like(param_sets)
    out[:, 0, 0] = np.sqrt(omega_sq)
    out[:, 1, 0] = freq * (1 - (-jerk / (omega_sq)) / C_VAL)
    out[:, 2, 0] = -accel / (omega_sq * C_VAL)
    out[:, 3, 0] = -jerk / (omega_sq * np.sqrt(omega_sq) * C_VAL)
    d_omega_sq = np.sqrt(
        (dsnap / accel) ** 2 + ((snap * daccel) / (accel**2)) ** 2,
    )
    out[:, 0, 1] = 0.5 * d_omega_sq / np.sqrt(omega_sq)
    out[:, 1, 1] = np.sqrt(
        ((1 + jerk / (omega_sq * C_VAL)) * dfreq) ** 2
        + ((freq / (omega_sq * C_VAL)) * djerk) ** 2
        + ((freq * jerk / (omega_sq**2 * C_VAL)) * d_omega_sq) ** 2,
    )
    out[:, 2, 1] = np.sqrt(
        (daccel / (omega_sq * C_VAL)) ** 2
        + ((accel * d_omega_sq) / (omega_sq**2 * C_VAL)) ** 2,
    )
    out[:, 3, 1] = np.sqrt(
        (djerk / (omega_sq * np.sqrt(omega_sq) * C_VAL)) ** 2
        + ((1.5 * jerk * d_omega_sq) / (C_VAL * omega_sq**2.5)) ** 2,
    )
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_params(
    taylor_param_vec: np.ndarray,
    delta_t: float,
    in_hole: bool = False,
) -> np.ndarray:
    """Specialized version of shift_taylor_params for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    in_hole : bool, optional
        If True, snap-accel is not significant but crackle-jerk is.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_j.
    """
    n_batch, n_poly = taylor_param_vec.shape
    n_params = n_poly - 1
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    d5_i = taylor_param_vec[:, 0]
    d4_i = taylor_param_vec[:, 1]
    d3_i = taylor_param_vec[:, 2]
    d2_i = taylor_param_vec[:, 3]
    d1_i = taylor_param_vec[:, 4]
    d0_i = taylor_param_vec[:, 5]

    # Pin-down the orbit
    omega_orb_sq = -d5_i / d3_i if in_hole else -d4_i / d2_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # rotation
    d2_j = d2_i * cos_odt + (d3_i / omega_orb) * sin_odt
    d3_j = d3_i * cos_odt - (d2_i * omega_orb) * sin_odt
    # circular constraints
    d4_j = -omega_orb_sq * d2_j
    d5_j = -omega_orb_sq * d3_j
    # Secular parameters. Integrate to get {v, d}
    d1_circ_i = -d3_i / omega_orb_sq
    d1_circ_j = -d3_j / omega_orb_sq
    d1_j = d1_circ_j + (d1_i - d1_circ_i)
    d0_circ_j = -d2_j / omega_orb_sq
    d0_circ_i = -d2_i / omega_orb_sq
    d0_j = d0_circ_j + (d0_i - d0_circ_i) + (d1_i - d1_circ_i) * delta_t

    out = np.empty((n_batch, n_poly), dtype=taylor_param_vec.dtype)
    out[:, 0] = d5_j
    out[:, 1] = d4_j
    out[:, 2] = d3_j
    out[:, 3] = d2_j
    out[:, 4] = d1_j
    out[:, 5] = d0_j
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_basis(
    leaf_bases_batch: np.ndarray,
    delta_t: float,
    p_orb_min: float,
) -> None:
    n_leaves, n_params, _ = leaf_bases_batch.shape
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)

    l_mat = maths.circ_taylor_transform_matrix(delta_t, p_orb_min)
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)
    for leaf in range(n_leaves):
        # Basis transformation (l_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = np.float64(0.0)
                for m in range(n_params):
                    acc += l_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def shift_taylor_circular_params_basis(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    delta_t: float,
    p_orb_min: float,
    in_hole: bool = False,
) -> None:
    n_leaves, n_params, _ = leaf_bases_batch.shape
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    leaf_params_batch[:, :-2] = shift_taylor_circular_params(
        leaf_params_batch[:, :-2],
        delta_t,
        in_hole=in_hole,
    )

    l_mat = maths.circ_taylor_transform_matrix(delta_t, p_orb_min, 1)
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)
    for leaf in range(n_leaves):
        # Basis transformation (l_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = np.float64(0.0)
                for m in range(n_params):
                    acc += l_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def taylor_to_cheby(taylor_param_vec: np.ndarray, t_s: float) -> np.ndarray:
    """Transform Taylor series coefficients to Chebyshev coefficients.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (..., n_params).
        Ordering is [d_k_max, ..., d_1, d_0] where d_k is coefficient of (t - t_c)^k/k!.
    t_s : float
        Scale factor for the transformation, typically half the time span.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients [alpha_k_max, ..., alpha_1, alpha_0].
    """
    if t_s <= 0:
        msg = "t_s must be a positive."
        raise ValueError(msg)
    n_params = taylor_param_vec.shape[-1]
    k_max = n_params - 1
    s_mat = maths.compute_connection_matrix_s(k_max)
    k_range = np.arange(k_max + 1, dtype=np.int64)
    scale = t_s**k_range / maths.fact(k_range)
    d_scaled = np.ascontiguousarray(taylor_param_vec[..., ::-1]) * scale
    alpha_standard = d_scaled @ s_mat
    return alpha_standard[..., ::-1]


@njit(cache=True, fastmath=True)
def taylor_to_cheby_full(taylor_full_vec: np.ndarray, t_s: float) -> np.ndarray:
    """Transform Taylor series coefficients and their errors to Chebyshev basis.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (..., n_params, 2).
        Ordering is [[d_k_max, dd_k_max], ..., [d_1, dd_1], [d_0, dd_0]]
    t_s : float
        Scale factor for the transformation, typically half the time span.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients and its errors. Ordering is
        [[alpha_k_max, dalpha_k_max], ..., [alpha_1, dalpha_1], [alpha_0, dalpha_0]]
    """
    if t_s <= 0:
        msg = "t_s must be a positive."
        raise ValueError(msg)
    n_params = taylor_full_vec.shape[-2]
    k_max = n_params - 1
    s_mat = maths.compute_connection_matrix_s(k_max)
    k_range = np.arange(k_max + 1, dtype=np.int64)
    scale = t_s**k_range / maths.fact(k_range)
    d_scaled_vals = np.ascontiguousarray(taylor_full_vec[..., ::-1, 0]) * scale
    d_scaled_errs = np.ascontiguousarray(taylor_full_vec[..., ::-1, 1]) * scale
    alpha_vals = d_scaled_vals @ s_mat
    # Propagate errors: var_alpha = d_scaled_errs^2 @ s_mat^2
    alpha_errs = np.sqrt(d_scaled_errs**2 @ s_mat**2)
    result = np.empty_like(taylor_full_vec)
    result[..., :, 0] = alpha_vals[..., ::-1]
    result[..., :, 1] = alpha_errs[..., ::-1]
    return result


@njit(cache=True, fastmath=True)
def taylor_to_cheby_errors(taylor_error_vec: np.ndarray, t_s: float) -> np.ndarray:
    """Transform Taylor series coefficients errors to Chebyshev coefficients errors.

    Parameters
    ----------
    taylor_error_vec : np.ndarray
        Parameter vector of shape (..., n_params).
        Ordering is [dd_k_max, ..., dd_1, dd_0]
    t_s : float
        Scale factor for the transformation, typically half the time span.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients errors [dalpha_k_max, ..., dalpha_1, dalpha_0].
    """
    if t_s <= 0:
        msg = "t_s must be a positive."
        raise ValueError(msg)
    n_params = taylor_error_vec.shape[-1]
    k_max = n_params - 1
    s_mat = maths.compute_connection_matrix_s(k_max)
    k_range = np.arange(k_max + 1, dtype=np.int64)
    scale = t_s**k_range / maths.fact(k_range)
    d_scaled_errs = np.ascontiguousarray(taylor_error_vec[..., ::-1]) * scale
    alpha_errs = np.sqrt(d_scaled_errs**2 @ s_mat**2)
    return alpha_errs[..., ::-1]


@njit(cache=True, fastmath=True)
def cheby_to_taylor(alpha_param_vec: np.ndarray, t_s: float) -> np.ndarray:
    """Transform Chebyshev coefficients to Taylor series coefficients.

    Parameters
    ----------
    alpha_param_vec : np.ndarray
        Parameter vector of shape (..., n_params).
        Ordering is [alpha_k_max, ..., alpha_1, alpha_0]
    t_s : float
        Scale factor for the transformation, typically half the time span.

    Returns
    -------
    np.ndarray
        Taylor series coefficients [d_k_max, ..., d_1, d_0] where d_k is
        coefficient of (t - t_c)^k/k!.
    """
    if t_s <= 0:
        msg = "t_s must be a positive."
        raise ValueError(msg)
    n_params = alpha_param_vec.shape[-1]
    k_max = n_params - 1
    alpha_standard = alpha_param_vec[..., ::-1]
    r_mat = maths.compute_connection_matrix_r(k_max)
    k_range = np.arange(k_max + 1, dtype=np.int64)
    d_standard = np.ascontiguousarray(alpha_standard) @ r_mat
    d_standard = d_standard * maths.fact(k_range) / t_s**k_range
    return d_standard[..., ::-1]


@njit(cache=True, fastmath=True)
def cheby_to_taylor_full(alpha_full_vec: np.ndarray, t_s: float) -> np.ndarray:
    """Transform Chebyshev coefficients and its errors to Taylor series basis.

    Parameters
    ----------
    alpha_full_vec : np.ndarray
        Parameter vector of shape (..., n_params, 2).
        Ordering is [[alpha_k_max, dalpha_k_max], ..., [alpha_1, dalpha_1],
        [alpha_0, dalpha_0]]
    t_s : float
        Scale factor for the transformation, typically half the time span.

    Returns
    -------
    np.ndarray
        Taylor series coefficients and its errors. Ordering is
        [[d_k_max, dd_k_max], ..., [d_1, dd_1], [d_0, dd_0]]
    """
    if t_s <= 0:
        msg = "t_s must be a positive."
        raise ValueError(msg)
    n_params = alpha_full_vec.shape[-2]
    k_max = n_params - 1
    alpha_vals = np.ascontiguousarray(alpha_full_vec[..., ::-1, 0])
    alpha_errs = np.ascontiguousarray(alpha_full_vec[..., ::-1, 1])
    r_mat = maths.compute_connection_matrix_r(k_max)
    d_intermediate_vals = alpha_vals @ r_mat
    # Propagate errors: var_alpha = alpha_errs^2 @ r_mat^2
    var_d_intermediate = alpha_errs**2 @ r_mat**2
    k_range = np.arange(k_max + 1, dtype=np.int64)
    scale = maths.fact(k_range) / t_s**k_range
    d_vals = d_intermediate_vals * scale
    d_errs = np.sqrt(var_d_intermediate * scale**2)
    result = np.empty_like(alpha_full_vec)
    result[..., :, 0] = d_vals[..., ::-1]
    result[..., :, 1] = d_errs[..., ::-1]
    return result


@njit(cache=True, fastmath=True)
def shift_cheby_basis(
    leaf_bases_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    poly_order: int,
) -> None:
    """Shift the kinematic chebyshev errors to a new domain.

    Parameters
    ----------
    alpha_error_vec : np.ndarray
        Chebyshev errors vector of shape (N, n_params) in domain coord_cur.
        [dalpha_k_max, ..., dalpha_1, dalpha_0]
    coord_next : tuple[float, float]
        The coordinate of the new domain.
    coord_cur : tuple[float, float]
        The coordinate of the current domain.
    """
    n_leaves, _, _ = leaf_bases_batch.shape
    n_params = poly_order
    tc1, ts1 = coord_cur
    tc2, ts2 = coord_next
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, tc1, ts1, tc2, ts2, 1)

    # Transform basis
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)
    for leaf in range(n_leaves):
        # Basis transformation (c_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = np.float64(0.0)
                for m in range(n_params):
                    acc += c_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def shift_cheby_params_basis(
    leaf_params_batch: np.ndarray,
    leaf_bases_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    poly_order: int,
) -> None:
    """Shift the kinematic chebyshev parameters and errors to a new domain.

    Parameters
    ----------
    alpha_full_vec : np.ndarray
        Chebyshev coefficients vector of shape (N, n_params, 2) in domain coord_cur.
        Each batch element is a vector of shape (n_params, 2)
        [[alpha_k_max, dalpha_k_max], ..., [alpha_1, dalpha_1], [alpha_0, dalpha_0]]
    coord_next : tuple[float, float]
        The coordinate of the new domain.
    coord_cur : tuple[float, float]
        The coordinate of the current domain.
    """
    n_leaves, _ = leaf_params_batch.shape
    n_params = poly_order
    tc1, ts1 = coord_cur
    tc2, ts2 = coord_next
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, tc1, ts1, tc2, ts2, 1)

    # Transform polynomial coefficients
    coeffs_tmp = np.empty(n_params + 1, dtype=np.float64)
    basis_tmp = np.empty((n_params, n_params), dtype=np.float64)

    for leaf in range(n_leaves):
        # Parameters transformation (cheb_coeffs @ c_mat)
        for i in range(n_params + 1):
            coeffs_tmp[i] = leaf_params_batch[leaf, i]
        for i in range(n_params + 1):
            acc = np.float64(0.0)
            for j in range(n_params + 1):
                acc += coeffs_tmp[j] * c_mat[j, i]
            leaf_params_batch[leaf, i] = acc

        # Basis transformation (c_mat.T @ B)
        for r in range(n_params):
            for c in range(n_params):
                acc = np.float64(0.0)
                for m in range(n_params):
                    acc += c_mat[m, r] * leaf_bases_batch[leaf, m, c]
                basis_tmp[r, c] = acc
        leaf_bases_batch[leaf] = basis_tmp


@njit(cache=True, fastmath=True)
def cheby_to_taylor_param_shift(
    alpha_param_vec: np.ndarray,
    t0: float,
    ts: float,
    t_eval: float,
) -> np.ndarray:
    """Shift the Chebyshev parameters to a new center and return Taylor coefficients.

    Parameters
    ----------
    alpha_param_vec : np.ndarray
        Chebyshev coefficients [alpha_kmax, ..., alpha_0] in domain centered at t0.
        2D batch array (N_batch, n_params).
    t0 : float
        Center of the original Chebyshev domain.
    ts : float
        Half-span (scale factor) of the Chebyshev domain.
    t_eval : float
        Time at which to evaluate the derivatives.

    Returns
    -------
    np.ndarray
        Taylor coefficients [d_kmax, ..., d_0] evaluated at t_eval.
        d_k corresponds to the coefficient of (t - t_eval)^k / k!.
    """
    n_params = alpha_param_vec.shape[-1]
    poly_order = n_params - 1
    # Get transformation matrix for domain shift: t0 -> t_eval
    # Source domain: center=t0, scale=ts; Target domain: center=t_eval, scale=ts
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, t0, ts, t_eval, ts, 1)
    alpha_standard_t0 = np.ascontiguousarray(alpha_param_vec)
    alpha_standard_t_eval = alpha_standard_t0 @ c_mat
    return cheby_to_taylor(alpha_standard_t_eval, ts)


@njit(cache=True, fastmath=True)
def taylor_to_chebyshev_limits_generic(
    taylor_limits: np.ndarray,
    ts: float,
) -> np.ndarray:
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
    k_range = np.arange(n_params_d, dtype=np.int64)
    scale_factor = (ts**k_range) / maths.fact(k_range)

    d_limits_scaled_min = np.zeros((n_batch, n_params_d), dtype=np.float64)
    d_limits_scaled_max = np.zeros((n_batch, n_params_d), dtype=np.float64)
    d_limits_scaled_min[:, 1:] = taylor_limits[:, ::-1, 0] * scale_factor[1:]
    d_limits_scaled_max[:, 1:] = taylor_limits[:, ::-1, 1] * scale_factor[1:]
    s_mat = maths.compute_connection_matrix_s(n_params)
    cheby_limits_min = np.zeros((n_batch, n_params), dtype=np.float64)
    cheby_limits_max = np.zeros((n_batch, n_params), dtype=np.float64)

    # All S_{m,k} coefficients are non-negative,
    for k in range(1, n_params_d):
        for m in range(k, n_params_d):
            s_mk = s_mat[m, k]
            term_min = s_mk * d_limits_scaled_min[:, m]
            term_max = s_mk * d_limits_scaled_max[:, m]
            cheby_limits_min[:, k - 1] += term_min
            cheby_limits_max[:, k - 1] += term_max

    param_limits_cheby = np.zeros((n_batch, n_params, 2), dtype=np.float64)
    param_limits_cheby[:, :, 0] = cheby_limits_min[:, ::-1]
    param_limits_cheby[:, :, 1] = cheby_limits_max[:, ::-1]
    return param_limits_cheby


@njit(cache=True, fastmath=True)
def taylor_to_chebyshev_limits_full(taylor_limits: np.ndarray, ts: float) -> np.ndarray:
    """Convert Taylor limits to Chebyshev limits with explicit unrolling.

    Parameters
    ----------
    taylor_limits : np.ndarray
        Input limits for Taylor coefficients [min, max]. Shape (N, nparams, 2).
        Order is [d_kmax, ..., d_2, d_1].
    ts : float
        Scale factor (half-span) of the domain.

    Returns
    -------
    np.ndarray
        Chebyshev coefficient limits. Shape (N, nparams, 2).
        Order is [alpha_kmax, ..., alpha_2, alpha_1].
    """
    n_batch, n_params, _ = taylor_limits.shape

    if n_params == 2:
        # d2, d1 -> alpha_2, alpha_1
        result = np.zeros((n_batch, 2, 2), dtype=np.float64)
        ts2 = ts * ts
        d1_min = taylor_limits[:, 1, 0] * ts
        d1_max = taylor_limits[:, 1, 1] * ts
        d2_min = taylor_limits[:, 0, 0] * ts2 * 0.5
        d2_max = taylor_limits[:, 0, 1] * ts2 * 0.5
        # alpha_1 = (d1 ts^1/1!)
        result[:, 1, 0] = d1_min
        result[:, 1, 1] = d1_max
        # alpha_2 = 0.5(d2 ts^2/2!)
        result[:, 0, 0] = 0.5 * d2_min
        result[:, 0, 1] = 0.5 * d2_max
        return result

    if n_params == 3:
        # d3, d2, d1 -> alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 3, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts

        d1_min = taylor_limits[:, 2, 0] * ts
        d1_max = taylor_limits[:, 2, 1] * ts
        d2_min = taylor_limits[:, 1, 0] * ts2 * 0.5
        d2_max = taylor_limits[:, 1, 1] * ts2 * 0.5
        d3_min = taylor_limits[:, 0, 0] * ts3 / 6.0
        d3_max = taylor_limits[:, 0, 1] * ts3 / 6.0

        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!)
        result[:, 2, 0] = d1_min + 0.75 * d3_min
        result[:, 2, 1] = d1_max + 0.75 * d3_max
        # alpha_2 = 0.5(d2 ts^2/2!)
        result[:, 1, 0] = 0.5 * d2_min
        result[:, 1, 1] = 0.5 * d2_max
        # alpha_3 = 0.25(d3 ts^3/3!)
        result[:, 0, 0] = 0.25 * d3_min
        result[:, 0, 1] = 0.25 * d3_max
        return result

    if n_params == 4:
        # d4, d3, d2, d1 -> alpha_4, alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 4, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        ts4 = ts3 * ts

        d1_min = taylor_limits[:, 3, 0] * ts
        d1_max = taylor_limits[:, 3, 1] * ts
        d2_min = taylor_limits[:, 2, 0] * ts2 * 0.5
        d2_max = taylor_limits[:, 2, 1] * ts2 * 0.5
        d3_min = taylor_limits[:, 1, 0] * ts3 / 6.0
        d3_max = taylor_limits[:, 1, 1] * ts3 / 6.0
        d4_min = taylor_limits[:, 0, 0] * ts4 / 24.0
        d4_max = taylor_limits[:, 0, 1] * ts4 / 24.0

        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!)
        result[:, 3, 0] = d1_min + 0.75 * d3_min
        result[:, 3, 1] = d1_max + 0.75 * d3_max

        # alpha_2 = 0.5(d2 ts^2/2!) + 0.5(d4 ts^4/4!)
        result[:, 2, 0] = 0.5 * d2_min + 0.5 * d4_min
        result[:, 2, 1] = 0.5 * d2_max + 0.5 * d4_max

        # alpha_3 = 0.25(d3 ts^3/3!)
        result[:, 1, 0] = 0.25 * d3_min
        result[:, 1, 1] = 0.25 * d3_max

        # alpha_4 = 0.125(d4 ts^4/4!)
        result[:, 0, 0] = 0.125 * d4_min
        result[:, 0, 1] = 0.125 * d4_max
        return result

    if n_params == 5:
        # d5, d4, d3, d2, d1 -> alpha_5, alpha_4, alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 5, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        ts4 = ts3 * ts
        ts5 = ts4 * ts

        d1_min = taylor_limits[:, 4, 0] * ts
        d1_max = taylor_limits[:, 4, 1] * ts
        d2_min = taylor_limits[:, 3, 0] * ts2 * 0.5
        d2_max = taylor_limits[:, 3, 1] * ts2 * 0.5
        d3_min = taylor_limits[:, 2, 0] * ts3 / 6.0
        d3_max = taylor_limits[:, 2, 1] * ts3 / 6.0
        d4_min = taylor_limits[:, 1, 0] * ts4 / 24.0
        d4_max = taylor_limits[:, 1, 1] * ts4 / 24.0
        d5_min = taylor_limits[:, 0, 0] * ts5 / 120.0
        d5_max = taylor_limits[:, 0, 1] * ts5 / 120.0

        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!) + 0.625(d5 ts^5/5!)
        result[:, 4, 0] = d1_min + 0.75 * d3_min + 0.625 * d5_min
        result[:, 4, 1] = d1_max + 0.75 * d3_max + 0.625 * d5_max

        # alpha_2 = 0.5(d2 ts^2/2!) + 0.5(d4 ts^4/4!)
        result[:, 3, 0] = 0.5 * d2_min + 0.5 * d4_min
        result[:, 3, 1] = 0.5 * d2_max + 0.5 * d4_max

        # alpha_3 = 0.25(d3 ts^3/3!) + 0.3125(d5 ts^5/5!)
        result[:, 2, 0] = 0.25 * d3_min + 0.3125 * d5_min
        result[:, 2, 1] = 0.25 * d3_max + 0.3125 * d5_max

        # alpha_4 = 0.125(d4 ts^4/4!)
        result[:, 1, 0] = 0.125 * d4_min
        result[:, 1, 1] = 0.125 * d4_max

        # alpha_5 = 0.0625(d5 ts^5/5!)
        result[:, 0, 0] = 0.0625 * d5_min
        result[:, 0, 1] = 0.0625 * d5_max
        return result

    msg = "n_params > 5 not supported in optimized version."
    raise ValueError(msg)


@njit(cache=True, fastmath=True)
def taylor_to_chebyshev_limits_d1(
    d1_limits: np.ndarray,
    taylor_limits: np.ndarray,
    ts: float,
) -> np.ndarray:
    """Convert Taylor limits to Chebyshev limits with explicit unrolling.

    Parameters
    ----------
    d1_limits : np.ndarray
        Input limits for d1 coefficients [min, max]. Shape (N, 2).
    taylor_limits : np.ndarray
        Input limits for Taylor coefficients [min, max]. Shape (nparams, 2).
        Order is [d_kmax, ..., d_2].
    ts : float
        Scale factor (half-span) of the domain.

    Returns
    -------
    np.ndarray
        Chebyshev coefficient for alpha_1. Shape (N, 2).
        Order is [alpha_1_min, alpha_1_max].
    """
    n_params, _ = taylor_limits.shape
    n_batch, _ = d1_limits.shape

    if n_params == 1:
        result = np.zeros((n_batch, 2), dtype=np.float64)
        result[:, 0] = d1_limits[:, 0] * ts
        result[:, 1] = d1_limits[:, 1] * ts
        return result

    if n_params == 2:
        # d3, d2, d1 -> alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        d1_min = d1_limits[:, 0] * ts
        d1_max = d1_limits[:, 1] * ts
        d3_min = taylor_limits[0, 0] * ts3 / 6.0
        d3_max = taylor_limits[0, 1] * ts3 / 6.0

        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!)
        result[:, 0] = d1_min + 0.75 * d3_min
        result[:, 1] = d1_max + 0.75 * d3_max
        return result

    if n_params == 3:
        # d4, d3, d2, d1 -> alpha_4, alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        d1_min = d1_limits[:, 0] * ts
        d1_max = d1_limits[:, 1] * ts
        d3_min = taylor_limits[1, 0] * ts3 / 6.0
        d3_max = taylor_limits[1, 1] * ts3 / 6.0
        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!)
        result[:, 0] = d1_min + 0.75 * d3_min
        result[:, 1] = d1_max + 0.75 * d3_max
        return result

    if n_params == 4:
        # d5, d4, d3, d2, d1 -> alpha_5, alpha_4, alpha_3, alpha_2, alpha_1
        result = np.zeros((n_batch, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        ts4 = ts3 * ts
        ts5 = ts4 * ts
        d1_min = d1_limits[:, 0] * ts
        d1_max = d1_limits[:, 1] * ts
        d3_min = taylor_limits[2, 0] * ts3 / 6.0
        d3_max = taylor_limits[2, 1] * ts3 / 6.0
        d5_min = taylor_limits[0, 0] * ts5 / 120.0
        d5_max = taylor_limits[0, 1] * ts5 / 120.0
        # alpha_1 = (d1 ts^1/1!) + 0.75(d3 ts^3/3!) + 0.625(d5 ts^5/5!)
        result[:, 0] = d1_min + 0.75 * d3_min + 0.625 * d5_min
        result[:, 1] = d1_max + 0.75 * d3_max + 0.625 * d5_max
        return result

    msg = "n_params > 5 not supported in optimized version."
    raise ValueError(msg)


@njit(cache=True, fastmath=True)
def taylor_to_chebyshev_limits_upto_d2(
    taylor_limits: np.ndarray,
    ts: float,
) -> np.ndarray:
    """Convert Taylor limits to Chebyshev limits with explicit unrolling.

    Parameters
    ----------
    taylor_limits : np.ndarray
        Input limits for Taylor coefficients [min, max]. Shape (nparams, 2).
        Order is [d_kmax, ..., d_2].
    ts : float
        Scale factor (half-span) of the domain.

    Returns
    -------
    np.ndarray
        Chebyshev coefficient limits. Shape (nparams - 1, 2).
        Order is [alpha_kmax, ..., alpha_2].
    """
    n_params, _ = taylor_limits.shape

    if n_params == 1:
        # d2 -> alpha_2
        result = np.zeros((1, 2), dtype=np.float64)
        ts2 = ts * ts
        d2_min = taylor_limits[0, 0] * ts2 * 0.5
        d2_max = taylor_limits[0, 1] * ts2 * 0.5
        # alpha_2 = 0.5(d2 ts^2/2!)
        result[0, 0] = 0.5 * d2_min
        result[0, 1] = 0.5 * d2_max
        return result

    if n_params == 2:
        # d3, d2 -> alpha_3, alpha_2
        result = np.zeros((2, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        d2_min = taylor_limits[1, 0] * ts2 * 0.5
        d2_max = taylor_limits[1, 1] * ts2 * 0.5
        d3_min = taylor_limits[0, 0] * ts3 / 6.0
        d3_max = taylor_limits[0, 1] * ts3 / 6.0
        # alpha_2 = 0.5(d2 ts^2/2!)
        result[0, 0] = 0.5 * d2_min
        result[0, 1] = 0.5 * d2_max
        # alpha_3 = 0.25(d3 ts^3/3!)
        result[1, 0] = 0.25 * d3_min
        result[1, 1] = 0.25 * d3_max
        return result

    if n_params == 3:
        # d4, d3, d2 -> alpha_4, alpha_3, alpha_2
        result = np.zeros((3, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        ts4 = ts3 * ts
        d2_min = taylor_limits[2, 0] * ts2 * 0.5
        d2_max = taylor_limits[2, 1] * ts2 * 0.5
        d3_min = taylor_limits[1, 0] * ts3 / 6.0
        d3_max = taylor_limits[1, 1] * ts3 / 6.0
        d4_min = taylor_limits[0, 0] * ts4 / 24.0
        d4_max = taylor_limits[0, 1] * ts4 / 24.0
        # alpha_2 = 0.5(d2 ts^2/2!) + 0.5(d4 ts^4/4!)
        result[0, 0] = 0.5 * d2_min + 0.5 * d4_min
        result[0, 1] = 0.5 * d2_max + 0.5 * d4_max

        # alpha_3 = 0.25(d3 ts^3/3!)
        result[1, 0] = 0.25 * d3_min
        result[1, 1] = 0.25 * d3_max

        # alpha_4 = 0.125(d4 ts^4/4!)
        result[2, 0] = 0.125 * d4_min
        result[2, 1] = 0.125 * d4_max
        return result

    if n_params == 4:
        # d5, d4, d3, d2 -> alpha_5, alpha_4, alpha_3, alpha_2
        result = np.zeros((4, 2), dtype=np.float64)
        ts2 = ts * ts
        ts3 = ts2 * ts
        ts4 = ts3 * ts
        ts5 = ts4 * ts
        d2_min = taylor_limits[3, 0] * ts2 * 0.5
        d2_max = taylor_limits[3, 1] * ts2 * 0.5
        d3_min = taylor_limits[2, 0] * ts3 / 6.0
        d3_max = taylor_limits[2, 1] * ts3 / 6.0
        d4_min = taylor_limits[1, 0] * ts4 / 24.0
        d4_max = taylor_limits[1, 1] * ts4 / 24.0
        d5_min = taylor_limits[0, 0] * ts5 / 120.0
        d5_max = taylor_limits[0, 1] * ts5 / 120.0
        # alpha_2 = 0.5(d2 ts^2/2!) + 0.5(d4 ts^4/4!)
        result[0, 0] = 0.5 * d2_min + 0.5 * d4_min
        result[0, 1] = 0.5 * d2_max + 0.5 * d4_max

        # alpha_3 = 0.25(d3 ts^3/3!) + 0.3125(d5 ts^5/5!)
        result[2, 0] = 0.25 * d3_min + 0.3125 * d5_min
        result[2, 1] = 0.25 * d3_max + 0.3125 * d5_max

        # alpha_4 = 0.125(d4 ts^4/4!)
        result[1, 0] = 0.125 * d4_min
        result[1, 1] = 0.125 * d4_max

        # alpha_5 = 0.0625(d5 ts^5/5!)
        result[0, 0] = 0.0625 * d5_min
        result[0, 1] = 0.0625 * d5_max
        return result

    msg = "n_params > 5 not supported in optimized version."
    raise ValueError(msg)
