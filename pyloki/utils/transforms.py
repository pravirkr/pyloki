from __future__ import annotations

import numpy as np
from numba import njit

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def shift_taylor_params(
    taylor_param_vec: np.ndarray,
    delta_t: float,
    n_out: int = 0,
) -> np.ndarray:
    """Shift the kinematic Taylor parameters to a new reference time.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (..., n_params) at reference time t_i.
        Ordering is [d_k_max, ..., d_1, d_0] where d_k is coefficient of (t - t_c)^k/k!.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    n_out : int, optional
        Number of output parameters from end. If negative, all parameters are returned.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (..., n_out) at the new reference time t_j.
    """
    n_params = taylor_param_vec.shape[-1]
    n_out = n_params if n_out <= 0 else min(n_out, n_params)
    powers = np.tril(np.arange(n_params)[:, np.newaxis] - np.arange(n_params))
    # Calculate the transformation matrix (taylor coefficients)
    t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
    t_mat = t_mat[-n_out:]
    # transform each vector in correct shape: np.dot(t_mat, param_vec)
    return np.ascontiguousarray(taylor_param_vec) @ t_mat.T


@njit(cache=True, fastmath=True)
def shift_taylor_errors(
    taylor_error_vec: np.ndarray,
    delta_t: float,
    use_conservative_tile: bool,
) -> np.ndarray:
    """Shift the kinematic Taylor errors to a new reference time.

    Parameters
    ----------
    taylor_error_vec : np.ndarray
        Error vector of shape (..., nparams) at reference time t_i.
        Ordering is [dd_k_max, ..., dd_1, dd_0] where dd_k is coefficient
        of (t - t_c)^k/k!.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Error vector of shape (..., nparams) at reference time t_j.
    """
    n_params = taylor_error_vec.shape[-1]
    # Construct the transformation matrix
    powers = np.tril(np.arange(n_params)[:, np.newaxis] - np.arange(n_params))
    t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
    # Propagate errors (assuming no covariance)
    if use_conservative_tile:
        return np.sqrt((np.ascontiguousarray(taylor_error_vec) ** 2) @ (t_mat**2).T)
    return np.ascontiguousarray(taylor_error_vec) * np.abs(np.diag(t_mat))


@njit(cache=True, fastmath=True)
def shift_taylor_full(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    use_conservative_tile: bool,
) -> np.ndarray:
    """Shift the kinematic Taylor parameters and errors to a new reference time.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (..., nparams, 2) at reference time t_i.
        Ordering is [[d_k_max, dd_k_max], ..., [d_1, dd_1], [d_0, dd_0]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (..., nparams, 2) at reference time t_j.
    """
    n_params = taylor_full_vec.shape[-2]
    # Construct the transformation matrix
    powers = np.tril(np.arange(n_params)[:, np.newaxis] - np.arange(n_params))
    t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
    # Propagate errors (assuming no covariance)
    leaves_param_new = np.empty_like(taylor_full_vec)
    param_values = np.ascontiguousarray(taylor_full_vec[..., :, 0])
    param_errors = np.ascontiguousarray(taylor_full_vec[..., :, 1])
    leaves_param_new[..., :, 0] = param_values @ t_mat.T
    if use_conservative_tile:
        leaves_param_new[..., :, 1] = np.sqrt((param_errors**2) @ (t_mat**2).T)
    else:
        leaves_param_new[..., :, 1] = param_errors * np.abs(np.diag(t_mat))
    return leaves_param_new


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
    taylor_param_vec_new = shift_taylor_params(taylor_param_vec, delta_t)
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
    n_batch, nparams = param_vec_batch.shape
    taylor_param_vec = np.zeros((n_batch, nparams + 1), dtype=param_vec_batch.dtype)
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
def taylor_fixed_report_batch(
    leaves_batch: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Specialized version of shift_taylor_params_d_f_batch for final report."""
    param_vec_batch = leaves_batch[:, :-2]
    n_batch, nparams, _ = param_vec_batch.shape
    taylor_param_vec = np.zeros((n_batch, nparams + 1), dtype=param_vec_batch.dtype)
    taylor_param_vec[:, :-2] = param_vec_batch[:, :-1, 0]  # till acceleration
    taylor_param_vec_new = shift_taylor_params(taylor_param_vec, delta_t)
    s_factor = 1 - taylor_param_vec_new[:, -2] / C_VAL
    param_vec_new = param_vec_batch.copy()
    param_vec_new[:, :-1, 0] = taylor_param_vec_new[:, :-2] / s_factor[:, None]
    param_vec_new[:, -1, 0] = param_vec_batch[:, -1, 0] * s_factor
    return param_vec_new


@njit(cache=True, fastmath=True)
def shift_taylor_params_circular_d_f_batch(
    param_vec_batch: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialized version of shift_taylor_params_circular_batch for d_f basis."""
    n_batch, nparams = param_vec_batch.shape
    if nparams != 4:
        msg = "4 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = param_vec_batch[:, 0]
    j_i = param_vec_batch[:, 1]
    a_i = param_vec_batch[:, 2]
    f_i = param_vec_batch[:, 3]

    omega_orb_sq = -s_i / a_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Pin-down {s, j, a}
    a_j = a_i * cos_odt + (j_i / omega_orb) * sin_odt
    j_j = j_i * cos_odt - (a_i * omega_orb) * sin_odt
    s_j = -omega_orb_sq * a_j
    # Integrate to get {v, d}
    v_circ_i = -j_i / omega_orb_sq
    v_circ_j = -j_j / omega_orb_sq
    delta_v = v_circ_j - v_circ_i
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    delta_d = d_circ_j - d_circ_i + (0 - v_circ_i) * delta_t
    delay_rel = delta_d / C_VAL
    f_j = f_i * (1 - delta_v / C_VAL)
    out = np.empty((n_batch, 5), dtype=param_vec_batch.dtype)
    out[:, 0] = s_j
    out[:, 1] = j_j
    out[:, 2] = a_j
    out[:, 3] = f_j
    return out, delay_rel


@njit(cache=True, fastmath=True)
def taylor_fixed_circular_report_batch(
    leaves_batch: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Specialized version of shift_taylor_params_circular_d_f_batch for report."""
    param_vec_batch = leaves_batch[:, :-2]
    _, nparams, _ = param_vec_batch.shape
    if nparams != 4:
        msg = "4 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = param_vec_batch[:, 0, 0]
    j_i = param_vec_batch[:, 1, 0]
    a_i = param_vec_batch[:, 2, 0]
    f_i = param_vec_batch[:, 3, 0]

    omega_orb_sq = -s_i / a_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Pin-down {s, j, a}
    a_j = a_i * cos_odt + (j_i / omega_orb) * sin_odt
    j_j = j_i * cos_odt - (a_i * omega_orb) * sin_odt
    s_j = -omega_orb_sq * a_j
    # Integrate to get {v, d}
    v_circ_i = -j_i / omega_orb_sq
    v_circ_j = -j_j / omega_orb_sq
    delta_v = v_circ_j - v_circ_i
    s_factor = 1 - delta_v / C_VAL
    out = param_vec_batch.copy()
    out[:, 0, 0] = s_j / s_factor[:, None]
    out[:, 1, 0] = j_j / s_factor[:, None]
    out[:, 2, 0] = a_j / s_factor[:, None]
    out[:, 3, 0] = f_i * s_factor
    return out


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
) -> np.ndarray:
    """Specialized version of shift_taylor_params for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit. Crackle is redundant and not used for pinning omega.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_j.
    """
    n_batch, n_params = taylor_param_vec.shape
    if n_params != 6:
        msg = "6 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = taylor_param_vec[:, 1]
    j_i = taylor_param_vec[:, 2]
    a_i = taylor_param_vec[:, 3]
    v_i = taylor_param_vec[:, 4]
    d_i = taylor_param_vec[:, 5]

    # Pin-down the orbit using snap and accel
    omega_orb_sq = -s_i / a_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Pin-down {s, j, a}
    a_j = a_i * cos_odt + (j_i / omega_orb) * sin_odt
    j_j = j_i * cos_odt - (a_i * omega_orb) * sin_odt
    s_j = -omega_orb_sq * a_j
    c_j = -omega_orb_sq * j_j
    # Integrate to get {v, d}
    v_circ_i = -j_i / omega_orb_sq
    v_circ_j = -j_j / omega_orb_sq
    v_j = v_circ_j + (v_i - v_circ_i)
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    d_j = d_circ_j + (d_i - d_circ_i) + (v_i - v_circ_i) * delta_t

    out = np.empty((n_batch, 6), dtype=taylor_param_vec.dtype)
    out[:, 0] = c_j
    out[:, 1] = s_j
    out[:, 2] = j_j
    out[:, 3] = a_j
    out[:, 4] = v_j
    out[:, 5] = d_j
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_crackle_params(
    taylor_param_vec: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Specialized version of shift_taylor_params for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit. Crackle is used here as snap/accel is unstable.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 6), ordered [c, s, j, a, v, d] at t_j.
    """
    n_batch, n_params = taylor_param_vec.shape
    if n_params != 6:
        msg = "6 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    c_i = taylor_param_vec[:, 0]
    j_i = taylor_param_vec[:, 2]
    a_i = taylor_param_vec[:, 3]
    v_i = taylor_param_vec[:, 4]
    d_i = taylor_param_vec[:, 5]

    # Pin-down the orbit using crackle and jerk
    omega_orb_sq = -c_i / j_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Pin-down {s, j, a}
    a_j = a_i * cos_odt + (j_i / omega_orb) * sin_odt
    j_j = j_i * cos_odt - (a_i * omega_orb) * sin_odt
    s_j = -omega_orb_sq * a_j
    c_j = -omega_orb_sq * j_j
    # Integrate to get {v, d}
    v_circ_i = -j_i / omega_orb_sq
    v_circ_j = -j_j / omega_orb_sq
    v_j = v_circ_j + (v_i - v_circ_i)
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    d_j = d_circ_j + (d_i - d_circ_i) + (v_i - v_circ_i) * delta_t

    out = np.empty((n_batch, 6), dtype=taylor_param_vec.dtype)
    out[:, 0] = c_j
    out[:, 1] = s_j
    out[:, 2] = j_j
    out[:, 3] = a_j
    out[:, 4] = v_j
    out[:, 5] = d_j
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_errors(
    taylor_error_vec: np.ndarray,
    delta_t: float,
    p_orb_min: float,
    use_conservative_tile: bool,
) -> np.ndarray:
    """Specialized version of shift_taylor_errors for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit. Crackle is redundant and not used for pinning omega.

    Parameters
    ----------
    taylor_error_vec : np.ndarray
        Error vector of shape (n_batch, 6) at reference time t_i.
        Ordering is [dc, ds, dj, da, dv, dd]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Error vector of shape (n_batch, 6) at reference time t_j.
    """
    n_batch, n_params = taylor_error_vec.shape
    if n_params != 6:
        msg = "6 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    sig_d3_i = taylor_error_vec[:, 2]
    sig_d2_i = taylor_error_vec[:, 3]
    sig_d1_i = taylor_error_vec[:, 4]

    omega_orb_max = 2 * np.pi / p_orb_min
    omega_orb_sq_max = omega_orb_max**2
    out = np.empty((n_batch, 6), dtype=taylor_error_vec.dtype)
    if use_conservative_tile:
        msg = "Conservative tile not implemented for circular orbit propagation."
        raise NotImplementedError(msg)
    sig_d2_j = np.sqrt(sig_d2_i**2 + (delta_t * sig_d3_i) ** 2)
    sig_d3_j = np.sqrt(sig_d3_i**2 + sig_d2_i**2 * omega_orb_sq_max)
    sig_d1_j = np.sqrt(
        sig_d1_i**2 + (delta_t * sig_d3_i / 2) ** 2 + (delta_t * sig_d2_i) ** 2,
    )
    out[:, 0] = omega_orb_sq_max * sig_d3_j
    out[:, 1] = omega_orb_sq_max * sig_d2_j
    out[:, 2] = sig_d3_j
    out[:, 3] = sig_d2_j
    out[:, 4] = sig_d1_j
    out[:, 5] = 0
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_full(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    use_conservative_tile: bool,
) -> np.ndarray:
    """Specialized version of shift_taylor_full for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit. Crackle is redundant and not used for pinning omega.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (n_batch, 6, 2) at reference time t_i.
        Ordering is [[c, dc], [s, ds], [j, dj], [a, da], [v, dv], [d, dd]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 6, 2) at reference time t_j.
    """
    n_batch, n_params, _ = taylor_full_vec.shape
    if n_params != 6:
        msg = "6 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    d4_i = taylor_full_vec[:, 1, 0]
    d3_i = taylor_full_vec[:, 2, 0]
    d2_i = taylor_full_vec[:, 3, 0]
    d1_i = taylor_full_vec[:, 4, 0]
    d0_i = taylor_full_vec[:, 5, 0]

    sig_d4_i = taylor_full_vec[:, 1, 1]
    sig_d3_i = taylor_full_vec[:, 2, 1]
    sig_d2_i = taylor_full_vec[:, 3, 1]
    sig_d1_i = taylor_full_vec[:, 4, 1]

    # Pin-down the orbit using snap and accel
    omega_orb_sq = -d4_i / d2_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Precompute some constants for efficiency
    inv_omega_orb = 1.0 / omega_orb
    inv_omega_orb_sq = 1.0 / omega_orb_sq
    sin_odt_inv_omega = sin_odt * inv_omega_orb
    d3_i_sin_odt_inv_omega = d3_i * sin_odt_inv_omega
    d2_i_omega_sin_odt = d2_i * omega_orb * sin_odt
    # Pin-down {s, j, a}
    d2_j = d2_i * cos_odt + d3_i_sin_odt_inv_omega
    d3_j = d3_i * cos_odt - d2_i_omega_sin_odt
    d4_j = -omega_orb_sq * d2_j
    d5_j = -omega_orb_sq * d3_j
    # Integrate to get {v, d}
    v_circ_i = -d3_i * inv_omega_orb_sq
    v_circ_j = -d3_j * inv_omega_orb_sq
    d1_diff = d1_i - v_circ_i
    d1_j = v_circ_j + d1_diff
    d_circ_j = -d2_j * inv_omega_orb_sq
    d_circ_i = -d2_i * inv_omega_orb_sq
    d0_j = d_circ_j + (d0_i - d_circ_i) + d1_diff * delta_t

    out = np.empty((n_batch, 6, 2), dtype=taylor_full_vec.dtype)
    out[:, 0, 0] = d5_j
    out[:, 1, 0] = d4_j
    out[:, 2, 0] = d3_j
    out[:, 3, 0] = d2_j
    out[:, 4, 0] = d1_j
    out[:, 5, 0] = d0_j

    if use_conservative_tile:
        omega_cu = omega_orb_sq * omega_orb
        inv_omega_cu = inv_omega_orb * inv_omega_orb_sq

        # Precompute sigma squared values
        var_d1_i = sig_d1_i**2
        var_d2_i = sig_d2_i**2
        var_d3_i = sig_d3_i**2
        var_d4_i = sig_d4_i**2

        u2 = omega_dt * d3_j * inv_omega_orb_sq - d3_i_sin_odt_inv_omega * inv_omega_orb
        u3 = -omega_dt * d2_j - d2_i * sin_odt
        u4 = -2 * omega_orb * d2_j - omega_orb_sq * u2
        u5 = -2 * omega_orb * d3_j - omega_orb_sq * u3
        u1 = 2 * (d3_j - d3_i) * inv_omega_cu - u3 * inv_omega_orb_sq
        v2 = -omega_orb / (2 * d2_i)
        v4 = omega_orb / (2 * d4_i)

        j52 = omega_cu * sin_odt + u5 * v2
        j53 = -omega_orb_sq * cos_odt
        j54 = u5 * v4
        var_d5_j = j52**2 * var_d2_i + j53**2 * var_d3_i + j54**2 * var_d4_i

        j42 = -omega_orb_sq * cos_odt + u4 * v2
        j43 = -omega_orb * sin_odt
        j44 = u4 * v4
        var_d4_j = j42**2 * var_d2_i + j43**2 * var_d3_i + j44**2 * var_d4_i

        j32 = -omega_orb * sin_odt + u3 * v2
        j33 = cos_odt
        j34 = u3 * v4
        var_d3_j = j32**2 * var_d2_i + j33**2 * var_d3_i + j34**2 * var_d4_i

        j22 = cos_odt + u2 * v2
        j23 = sin_odt / omega_orb
        j24 = u2 * v4
        var_d2_j = j22**2 * var_d2_i + j23**2 * var_d3_i + j24**2 * var_d4_i

        j11 = 1.0
        j12 = sin_odt_inv_omega + u1 * v2
        j13 = (1 - cos_odt) * inv_omega_orb_sq
        j14 = u1 * v4
        var_d1_j = (
            j11**2 * var_d1_i
            + j12**2 * var_d2_i
            + j13**2 * var_d3_i
            + j14**2 * var_d4_i
        )

        out[:, 0, 1] = np.sqrt(var_d5_j)
        out[:, 1, 1] = np.sqrt(var_d4_j)
        out[:, 2, 1] = np.sqrt(var_d3_j)
        out[:, 3, 1] = np.sqrt(var_d2_j)
        out[:, 4, 1] = np.sqrt(var_d1_j)
        out[:, 5, 1] = 0
    else:
        sig_d2_j = np.sqrt(
            (cos_odt * sig_d2_i) ** 2 + (sin_odt * sig_d3_i) ** 2 / omega_orb_sq,
        )
        sig_d3_j = np.sqrt(
            (cos_odt * sig_d3_i) ** 2 + (sin_odt * sig_d2_i) ** 2 * omega_orb_sq,
        )
        sig_d1_j = np.sqrt(
            sig_d1_i**2
            + ((1 - cos_odt) * sig_d3_i / omega_orb_sq) ** 2
            + (sig_d2_i * sin_odt) ** 2 / omega_orb_sq,
        )
        out[:, 0, 1] = omega_orb_sq * sig_d3_j
        out[:, 1, 1] = omega_orb_sq * sig_d2_j
        out[:, 2, 1] = sig_d3_j
        out[:, 3, 1] = sig_d2_j
        out[:, 4, 1] = sig_d1_j
        out[:, 5, 1] = 0
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_circular_crackle_full(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    use_conservative_tile: bool,
) -> np.ndarray:
    """Specialized version of shift_taylor_full for circular-orbit propagation.

    Works only for 6 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit. Crackle is used here as snap/accel is unstable.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (n_batch, 6, 2) at reference time t_i.
        Ordering is [[c, dc], [s, ds], [j, dj], [a, da], [v, dv], [d, dd]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 6, 2) at reference time t_j.
    """
    n_batch, n_params, _ = taylor_full_vec.shape
    if n_params != 6:
        msg = "6 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    d5_i = taylor_full_vec[:, 0, 0]
    d3_i = taylor_full_vec[:, 2, 0]
    d2_i = taylor_full_vec[:, 3, 0]
    d1_i = taylor_full_vec[:, 4, 0]
    d0_i = taylor_full_vec[:, 5, 0]

    sig_d5_i = taylor_full_vec[:, 0, 1]
    sig_d3_i = taylor_full_vec[:, 2, 1]
    sig_d2_i = taylor_full_vec[:, 3, 1]
    sig_d1_i = taylor_full_vec[:, 4, 1]

    # Pin-down the orbit using crackle and jerk
    omega_orb_sq = -d5_i / d3_i
    omega_orb = np.sqrt(omega_orb_sq)
    # Evolve the phase to the new time t_j = t_i + delta_t
    omega_dt = omega_orb * delta_t
    cos_odt = np.cos(omega_dt)
    sin_odt = np.sin(omega_dt)
    # Precompute some constants for efficiency
    inv_omega_orb = 1.0 / omega_orb
    inv_omega_orb_sq = 1.0 / omega_orb_sq
    sin_odt_inv_omega = sin_odt * inv_omega_orb
    d3_i_sin_odt_inv_omega = d3_i * sin_odt_inv_omega
    d2_i_omega_sin_odt = d2_i * omega_orb * sin_odt
    # Pin-down {s, j, a}
    d2_j = d2_i * cos_odt + d3_i_sin_odt_inv_omega
    d3_j = d3_i * cos_odt - d2_i_omega_sin_odt
    d4_j = -omega_orb_sq * d2_j
    d5_j = -omega_orb_sq * d3_j
    # Integrate to get {v, d}
    v_circ_i = -d3_i * inv_omega_orb_sq
    v_circ_j = -d3_j * inv_omega_orb_sq
    d1_diff = d1_i - v_circ_i
    d1_j = v_circ_j + d1_diff
    d_circ_j = -d2_j * inv_omega_orb_sq
    d_circ_i = -d2_i * inv_omega_orb_sq
    d0_j = d_circ_j + (d0_i - d_circ_i) + d1_diff * delta_t

    out = np.empty((n_batch, 6, 2), dtype=taylor_full_vec.dtype)
    out[:, 0, 0] = d5_j
    out[:, 1, 0] = d4_j
    out[:, 2, 0] = d3_j
    out[:, 3, 0] = d2_j
    out[:, 4, 0] = d1_j
    out[:, 5, 0] = d0_j

    if use_conservative_tile:
        omega_cu = omega_orb_sq * omega_orb
        inv_omega_cu = inv_omega_orb * inv_omega_orb_sq

        # Precompute sigma squared values
        var_d1_i = sig_d1_i**2
        var_d2_i = sig_d2_i**2
        var_d3_i = sig_d3_i**2
        var_d5_i = sig_d5_i**2

        u2 = omega_dt * d3_j * inv_omega_orb_sq - d3_i_sin_odt_inv_omega * inv_omega_orb
        u3 = -omega_dt * d2_j - d2_i * sin_odt
        u4 = -2 * omega_orb * d2_j - omega_orb_sq * u2
        u5 = -2 * omega_orb * d3_j - omega_orb_sq * u3
        u1 = 2 * (d3_j - d3_i) * inv_omega_cu - u3 * inv_omega_orb_sq
        v3 = -omega_orb / (2 * d3_i)
        v5 = omega_orb / (2 * d5_i)

        j52 = omega_cu * sin_odt
        j53 = -omega_orb_sq * cos_odt + u5 * v3
        j55 = u5 * v5
        var_d5_j = j52**2 * var_d2_i + j53**2 * var_d3_i + j55**2 * var_d5_i

        j42 = -omega_orb_sq * cos_odt
        j43 = -omega_orb * sin_odt + u4 * v3
        j45 = u4 * v5
        var_d4_j = j42**2 * var_d2_i + j43**2 * var_d3_i + j45**2 * var_d5_i

        j32 = -omega_orb * sin_odt
        j33 = cos_odt + u3 * v3
        j35 = u3 * v5
        var_d3_j = j32**2 * var_d2_i + j33**2 * var_d3_i + j35**2 * var_d5_i

        j22 = cos_odt
        j23 = sin_odt / omega_orb + u2 * v3
        j25 = u2 * v5
        var_d2_j = j22**2 * var_d2_i + j23**2 * var_d3_i + j25**2 * var_d5_i

        j11 = 1.0
        j12 = sin_odt / omega_orb
        j13 = (1 - cos_odt) / omega_orb_sq + u1 * v3
        j15 = u1 * v5
        var_d1_j = (
            j11**2 * var_d1_i
            + j12**2 * var_d2_i
            + j13**2 * var_d3_i
            + j15**2 * var_d5_i
        )

        out[:, 0, 1] = np.sqrt(var_d5_j)
        out[:, 1, 1] = np.sqrt(var_d4_j)
        out[:, 2, 1] = np.sqrt(var_d3_j)
        out[:, 3, 1] = np.sqrt(var_d2_j)
        out[:, 4, 1] = np.sqrt(var_d1_j)
        out[:, 5, 1] = 0
    else:
        sig_d2_j = np.sqrt(
            (cos_odt * sig_d2_i) ** 2 + (sin_odt * sig_d3_i) ** 2 / omega_orb_sq,
        )
        sig_d3_j = np.sqrt(
            (cos_odt * sig_d3_i) ** 2 + (sin_odt * sig_d2_i) ** 2 * omega_orb_sq,
        )
        sig_d1_j = np.sqrt(
            sig_d1_i**2
            + ((1 - cos_odt) * sig_d3_i / omega_orb_sq) ** 2
            + (sig_d2_i * sin_odt) ** 2 / omega_orb_sq,
        )
        out[:, 0, 1] = omega_orb_sq * sig_d3_j
        out[:, 1, 1] = omega_orb_sq * sig_d2_j
        out[:, 2, 1] = sig_d3_j
        out[:, 3, 1] = sig_d2_j
        out[:, 4, 1] = sig_d1_j
        out[:, 5, 1] = 0
    return out


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
def shift_cheby_errors(
    alpha_error_vec: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
) -> np.ndarray:
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
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Chebyshev errors vector of shape (N, n_params) in domain coord_next.
    """
    _, n_params = alpha_error_vec.shape
    poly_order = n_params - 1
    tc1, ts1 = coord_cur
    tc2, ts2 = coord_next
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, tc1, ts1, tc2, ts2)
    alpha_errors = np.ascontiguousarray(alpha_error_vec[..., ::-1])
    if use_conservative_tile:
        alpha_errors_new = np.sqrt((alpha_errors**2) @ (c_mat**2))
    else:
        alpha_errors_new = alpha_errors * np.abs(np.diag(c_mat))
    return alpha_errors_new[..., ::-1]


@njit(cache=True, fastmath=True)
def shift_cheby_full(
    alpha_full_vec: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    use_conservative_tile: bool,
) -> np.ndarray:
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
    use_conservative_tile : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients vector of shape (N, n_params, 2) in domain coord_next.
    """
    n_params = alpha_full_vec.shape[-2]
    poly_order = n_params - 1
    tc1, ts1 = coord_cur
    tc2, ts2 = coord_next
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, tc1, ts1, tc2, ts2)
    alpha_vec_new = np.empty_like(alpha_full_vec)
    alpha_values = np.ascontiguousarray(alpha_full_vec[..., ::-1, 0])
    alpha_errors = np.ascontiguousarray(alpha_full_vec[..., ::-1, 1])
    alpha_values_new = alpha_values @ c_mat
    if use_conservative_tile:
        alpha_errors_new = np.sqrt((alpha_errors**2) @ (c_mat**2))
    else:
        alpha_errors_new = alpha_errors * np.abs(np.diag(c_mat))
    alpha_vec_new[..., :, 0] = alpha_values_new[..., ::-1]
    alpha_vec_new[..., :, 1] = alpha_errors_new[..., ::-1]
    return alpha_vec_new


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
    c_mat = maths.poly_chebyshev_transform_matrix(poly_order, t0, ts, t_eval, ts)
    alpha_standard_t0 = np.ascontiguousarray(alpha_param_vec[..., ::-1])
    alpha_standard_t_eval = alpha_standard_t0 @ c_mat
    return cheby_to_taylor(alpha_standard_t_eval[..., ::-1], ts)


@njit(cache=True, fastmath=True)
def taylor_report_batch(leaves_batch: np.ndarray) -> np.ndarray:
    param_sets_batch = leaves_batch[:, :-2, :]
    v_final = leaves_batch[:, -3, 0]
    dv_final = leaves_batch[:, -3, 1]
    f0_batch = leaves_batch[:, -1, 0]
    s_factor = 1 - v_final / C_VAL
    # Gauge transform + error propagation
    param_sets_vals = param_sets_batch[:, :-1, 0]
    param_sets_sigs = param_sets_batch[:, :-1, 1]
    param_sets_batch[:, :-1, 0] = param_sets_vals / s_factor[:, None]
    param_sets_batch[:, :-1, 1] = np.sqrt(
        (param_sets_sigs / s_factor[:, None]) ** 2
        + ((param_sets_vals / (C_VAL * s_factor[:, None] ** 2)) ** 2)
        * (dv_final[:, None] ** 2),
    )
    param_sets_batch[:, -1, 0] = f0_batch * s_factor
    param_sets_batch[:, -1, 1] = f0_batch * dv_final / C_VAL
    return param_sets_batch


@njit(cache=True, fastmath=True)
def chebyshev_report_batch(
    leaves_batch: np.ndarray,
    coord_mid: tuple[float, float],
) -> np.ndarray:
    cheby_coeffs_batch = leaves_batch[:, :-1, :]
    f0_batch = leaves_batch[:, -1, 0]
    _, scale = coord_mid
    param_sets_batch_d = cheby_to_taylor_full(cheby_coeffs_batch, scale)
    param_sets_batch = param_sets_batch_d[:, :-1]
    v_final = param_sets_batch[:, -1, 0]
    dv_final = param_sets_batch[:, -1, 1]
    s_factor = 1 - v_final / C_VAL
    # Gauge transform + error propagation
    param_sets_vals = param_sets_batch[:, :-1, 0]
    param_sets_sigs = param_sets_batch[:, :-1, 1]
    param_sets_batch[:, :-1, 0] = param_sets_vals / s_factor[:, None]
    param_sets_batch[:, :-1, 1] = np.sqrt(
        (param_sets_sigs / s_factor[:, None]) ** 2
        + ((param_sets_vals / (C_VAL * s_factor[:, None] ** 2)) ** 2)
        * (dv_final[:, None] ** 2),
    )
    param_sets_batch[:, -1, 0] = f0_batch * s_factor
    param_sets_batch[:, -1, 1] = f0_batch * dv_final / C_VAL
    return param_sets_batch


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
