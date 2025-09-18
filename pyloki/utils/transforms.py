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
def shift_taylor_full(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    conservative_errors: bool,
) -> np.ndarray:
    """Shift the kinematic Taylor parameters and errors to a new reference time.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (..., nparams, 2) at reference time t_i.
        Ordering is [[d_k_max, dd_k_max], ..., [d_1, dd_1], [d_0, dd_0]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    conservative_errors : bool
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
    if conservative_errors:
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
def shift_taylor_params_circular_batch(
    taylor_param_vec: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Specialized version of shift_taylor_params for circular-orbit propagation.

    Works only for 5 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit.

    Parameters
    ----------
    taylor_param_vec : np.ndarray
        Parameter vector of shape (n_batch, 5), ordered [s, j, a, v, d] at t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 5), ordered [s, j, a, v, d] at t_j.
    """
    n_batch, n_params = taylor_param_vec.shape
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = taylor_param_vec[:, 0]
    j_i = taylor_param_vec[:, 1]
    a_i = taylor_param_vec[:, 2]
    v_i = taylor_param_vec[:, 3]
    d_i = taylor_param_vec[:, 4]

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
    v_j = v_circ_j + (v_i - v_circ_i)
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    d_j = d_circ_j + (d_i - d_circ_i) + (v_i - v_circ_i) * delta_t

    out = np.empty((n_batch, 5), dtype=taylor_param_vec.dtype)
    out[:, 0] = s_j
    out[:, 1] = j_j
    out[:, 2] = a_j
    out[:, 3] = v_j
    out[:, 4] = d_j
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_full_circular_batch(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    conservative_errors: bool,  # noqa: ARG001
) -> np.ndarray:
    """Specialized version of shift_taylor_full for circular-orbit propagation.

    Works only for 5 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (n_batch, 5, 2) at reference time t_i.
        Ordering is [[s, ds], [j, dj], [a, da], [v, dv], [d, dd]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    conservative_errors : bool
        If True, the errors are propagated conservatively, otherwise unchanged.

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 5, 2) at reference time t_j.
    """
    _, n_params, _ = taylor_full_vec.shape
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = taylor_full_vec[:, 0, 0]
    j_i = taylor_full_vec[:, 1, 0]
    a_i = taylor_full_vec[:, 2, 0]
    v_i = taylor_full_vec[:, 3, 0]
    d_i = taylor_full_vec[:, 4, 0]

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
    v_j = v_circ_j + (v_i - v_circ_i)
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    d_j = d_circ_j + (d_i - d_circ_i) + (v_i - v_circ_i) * delta_t

    out = taylor_full_vec.copy()
    # Unchanged errors, for now (no use of grid conservative here)
    out[:, 0, 0] = s_j
    out[:, 1, 0] = j_j
    out[:, 2, 0] = a_j
    out[:, 3, 0] = v_j
    out[:, 4, 0] = d_j
    return out


@njit(cache=True, fastmath=True)
def shift_taylor_full_circular_batch_jacobian(
    taylor_full_vec: np.ndarray,
    delta_t: float,
    conservative_errors: bool,
) -> np.ndarray:
    """Specialized version of shift_taylor_full for circular-orbit propagation.

    Works only for 5 parameters and batch processing. Input must be guaranteed to be
    a physical circular orbit.

    Parameters
    ----------
    taylor_full_vec : np.ndarray
        Parameter vector of shape (n_batch, 5, 2) at reference time t_i.
        Ordering is [[s, ds], [j, dj], [a, da], [v, dv], [d, dd]]
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    conservative_errors : bool
        If True, propagate errors via the full Jacobian (first-order, no covariances).
        If False, scale only by the absolute diagonal Jacobian terms (ignores mixing).

    Returns
    -------
    np.ndarray
        Parameter vector of shape (n_batch, 5, 2) at reference time t_j.
    """
    _, n_params, _ = taylor_full_vec.shape
    if n_params != 5:
        msg = "5 parameters are needed for circular orbit propagation."
        raise ValueError(msg)
    s_i = taylor_full_vec[:, 0, 0]
    j_i = taylor_full_vec[:, 1, 0]
    a_i = taylor_full_vec[:, 2, 0]
    v_i = taylor_full_vec[:, 3, 0]
    d_i = taylor_full_vec[:, 4, 0]

    ds_i = taylor_full_vec[:, 0, 1]
    dj_i = taylor_full_vec[:, 1, 1]
    da_i = taylor_full_vec[:, 2, 1]
    dv_i = taylor_full_vec[:, 3, 1]
    dd_i = taylor_full_vec[:, 4, 1]

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
    v_j = v_circ_j + (v_i - v_circ_i)
    d_circ_j = -a_j / omega_orb_sq
    d_circ_i = -a_i / omega_orb_sq
    d_j = d_circ_j + (d_i - d_circ_i) + (v_i - v_circ_i) * delta_t

    out = taylor_full_vec.copy()
    out[:, 0, 0] = s_j
    out[:, 1, 0] = j_j
    out[:, 2, 0] = a_j
    out[:, 3, 0] = v_j
    out[:, 4, 0] = d_j

    # Error propagation
    # Common derivatives
    w2 = omega_orb_sq
    w = omega_orb
    dt = delta_t
    # Derivatives of w and w2 wrt inputs
    dw2_ds = -1.0 / a_i
    dw2_da = s_i / (a_i**2)
    dw_ds = -1.0 / (2.0 * w * a_i)
    dw_da = -w / (2.0 * a_i)
    dwdt_ds = dt * dw_ds
    dwdt_da = dt * dw_da

    # Helper for derivatives of a_j
    # A = -a_i * sin(w dt) * dt + (j_i / w) * cos(w dt) * dt - sin(w dt) * j_i / w^2
    aa = -a_i * sin_odt * dt + (j_i / w) * cos_odt * dt - sin_odt * j_i / (w**2)
    daj_ds = aa * dw_ds
    daj_da = cos_odt + aa * dw_da
    daj_dj = sin_odt / w

    # Derivatives for j_j
    djj_dj = cos_odt
    djj_ds = (
        -j_i * sin_odt * dwdt_ds - a_i * sin_odt * dw_ds - a_i * w * cos_odt * dwdt_ds
    )
    djj_da = (
        -w * sin_odt
        - j_i * sin_odt * dwdt_da
        - a_i * sin_odt * dw_da
        - a_i * w * cos_odt * dwdt_da
    )

    # Derivatives for s_j = -w2 * a_j
    dsj_dj = -w * sin_odt
    dsj_ds = -dw2_ds * a_j - w2 * daj_ds
    dsj_da = -dw2_da * a_j - w2 * daj_da

    # Derivatives for v_j = v_i + (j_i - j_j) / w2
    dvj_dv = 1.0
    dvj_dj = (1.0 - djj_dj) / w2
    dvj_ds = -(j_i - j_j) * (dw2_ds / (w2**2)) - djj_ds / w2
    dvj_da = -(j_i - j_j) * (dw2_da / (w2**2)) - djj_da / w2

    # Derivatives for d_j = -a_j/w2 + d_i + a_i/w2 + (v_i + j_i/w2) dt
    dinv_w2_ds = 1.0 / (a_i * (w2**2))
    dinv_w2_da = -s_i / (a_i**2 * (w2**2))
    ddj_dd = 1.0
    ddj_dv = dt
    ddj_dj = dt / w2 - sin_odt / (w**3)
    ddj_ds = -(1.0 / w2) * daj_ds + (j_i * dt - a_j) * dinv_w2_ds
    ddj_da = (
        -(1.0 / w2) * daj_da
        + (1.0 / w2)
        + (j_i * dt - a_j) * dinv_w2_da
        - s_i / (a_i * (w2**2))
    )

    if conservative_errors:
        # Full Jacobian (diagonal covariance assumption)
        var_s = (dsj_ds * ds_i) ** 2 + (dsj_dj * dj_i) ** 2 + (dsj_da * da_i) ** 2
        var_j = (djj_ds * ds_i) ** 2 + (djj_dj * dj_i) ** 2 + (djj_da * da_i) ** 2
        var_a = (daj_ds * ds_i) ** 2 + (daj_dj * dj_i) ** 2 + (daj_da * da_i) ** 2
        var_v = (
            (dvj_ds * ds_i) ** 2
            + (dvj_dj * dj_i) ** 2
            + (dvj_da * da_i) ** 2
            + (dvj_dv * dv_i) ** 2
        )
        var_d = (
            (ddj_ds * ds_i) ** 2
            + (ddj_dj * dj_i) ** 2
            + (ddj_da * da_i) ** 2
            + (ddj_dv * dv_i) ** 2
            + (ddj_dd * dd_i) ** 2
        )
        out[:, 0, 1] = np.sqrt(var_s)
        out[:, 1, 1] = np.sqrt(var_j)
        out[:, 2, 1] = np.sqrt(var_a)
        out[:, 3, 1] = np.sqrt(var_v)
        out[:, 4, 1] = np.sqrt(var_d)
    else:
        # Diagonal-only scaling (non-conservative)
        out[:, 0, 1] = np.abs(dsj_ds) * ds_i
        out[:, 1, 1] = np.abs(djj_dj) * dj_i
        out[:, 2, 1] = np.abs(daj_da) * da_i
        out[:, 3, 1] = dv_i  # ∂v_j/∂v_i = 1
        out[:, 4, 1] = dd_i  # ∂d_j/∂d_i = 1

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
    conservative_errors: bool,
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
    conservative_errors : bool
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
    if conservative_errors:
        alpha_errors_new = np.sqrt((alpha_errors**2) @ (c_mat**2))
    else:
        alpha_errors_new = alpha_errors * np.abs(np.diag(c_mat))
    return alpha_errors_new[..., ::-1]


@njit(cache=True, fastmath=True)
def shift_cheby_full(
    alpha_full_vec: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
    conservative_errors: bool,
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
    conservative_errors : bool
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
    if conservative_errors:
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
