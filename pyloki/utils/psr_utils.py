from __future__ import annotations

import math

import numpy as np
from numba import njit, vectorize

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL


@vectorize(nopython=True, cache=True)
def get_phase_idx(proper_time: float, freq: float, nbins: int, delay: float) -> float:
    """Compute the absolute phase index for a periodic signal.

    The phase is calculated as the fractional part of the total cycles and then scaled
    by the number of bins. Handles negative time and delay.

    Parameters
    ----------
    proper_time : float
        Proper time of the signal in time units (arrival time).
    freq : float
        Frequency of the signal in Hz. Must be positive.
    nbins : int
        Number of bins in the folded profile. Must be positive.
    delay : float
        Signal delay due to pulsar binary motion in time units.

    Returns
    -------
    float
        Phase bin index as a float in the range [0, nbins).
    """
    if freq <= 0:
        msg = "Frequency must be positive."
        raise ValueError(msg)
    if nbins <= 0:
        msg = "Number of bins must be positive."
        raise ValueError(msg)
    phase = (proper_time - delay) * freq
    phase -= math.floor(phase)
    iphase = phase * nbins
    if iphase >= nbins:
        iphase = 0.0
    return iphase


@vectorize(nopython=True, cache=True)
def get_phase_idx_int(proper_time: float, freq: float, nbins: int, delay: float) -> int:
    shifts = get_phase_idx(proper_time, freq, nbins, delay)
    shifts_float = np.float32(shifts)
    return round(shifts_float) % nbins


@njit(cache=True, fastmath=True)
def poly_taylor_step_f(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid size for frequency and its derivatives {f_k, ..., f}.

    Parameters
    ----------
    nparams : int
        Number of parameters in the Taylor expansion.
    tobs : float
        Total observation time in seconds.
    nbins : int
        Number of bins in the folded profile.
    eta : float, optional
        Tolerance parameter, eta in bins, by default 1.
    t_ref : float, optional
        Reference time in segment e.g. tobs/2, etc., by default 0.

    Returns
    -------
    float
        Optimal frequency and its derivative step size in reverse order.
    """
    dphi = eta / nbins
    k = np.arange(nparams)
    dparams_f = dphi * maths.fact(k + 1) / (tobs - t_ref) ** (k + 1)
    dparams_f_opt = 2**k * dparams_f
    return dparams_f_opt[::-1]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_f(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref)
    dparams = np.zeros(nparams, dtype=np.float64)
    dparams[:-1] = dparams_f[:-1] * C_VAL / f_max
    dparams[-1] = dparams_f[-1]
    return dparams


@njit(cache=True, fastmath=True)
def poly_taylor_step_d(
    poly_order: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, nbins, eta, t_ref)
    return dparams_f * C_VAL / f_max


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_vec(
    poly_order: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, nbins, eta, t_ref)
    return dparams_f[np.newaxis, :] * C_VAL / f_max[:, np.newaxis]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_f_vec(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref)
    dparams = np.zeros((len(f_max), nparams), dtype=np.float64)
    dparams[:, :-1] = dparams_f[:-1][np.newaxis, :] * C_VAL / f_max[:, np.newaxis]
    dparams[:, -1] = dparams_f[-1]
    return dparams


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d_f_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    nbins: int,
    f_cur: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Compute the bin shift for parameters {d_k,... d_2, f} (vector)."""
    nbatch, nparams = dparam_old.shape
    k = np.arange(nparams - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    factors_opt = factors / 2**k
    factors_broadcast = np.empty((nbatch, nparams), dtype=dparam_old.dtype)
    for i in range(nbatch):
        factors_broadcast[i, :] = factors_opt
    # For all but last param, scale by f_cur / C_VAL
    scale = (f_cur / C_VAL)[:, np.newaxis]
    factors_broadcast[:, :-1] *= scale
    return np.abs(dparam_old - dparam_new) * factors_broadcast


@njit
def split_f(
    df_old: float,
    df_new: float,
    tobs_new: float,
    k: int,
    nbins: float,
    eta: float,
    t_ref: float = 0,
) -> bool:
    """Check if a parameter {f_k} should be split."""
    factor = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    factor_opt = factor / 2**k
    eps = 1e-6
    return abs(df_old - df_new) * factor_opt > (eta - eps)


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    nbins: int,
    f_cur: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (scalar)."""
    n_params = len(dparam_old)
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    factors_opt = factors / 2**k
    factors_opt *= f_cur / C_VAL
    return np.abs(dparam_old - dparam_new) * factors_opt


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    nbins: int,
    f_cur: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (vector)."""
    n_batch, n_params = dparam_old.shape
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    factors_opt = factors / 2**k
    factors_broadcast = np.empty((n_batch, n_params), dtype=dparam_old.dtype)
    for i in range(n_batch):
        factors_broadcast[i, :] = factors_opt
    scale = (f_cur / C_VAL)[:, np.newaxis]
    factors_broadcast *= scale
    return np.abs(dparam_old - dparam_new) * factors_broadcast


@njit
def period_step(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit(cache=True, fastmath=True)
def poly_cheb_step_vec(
    nparams: int,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
) -> np.ndarray:
    dphi = eta / nbins
    dparams_f = np.zeros(nparams, np.float64) + dphi
    return dparams_f[np.newaxis, :] * C_VAL / f_max[:, np.newaxis]


@njit(cache=True, fastmath=True)
def poly_cheb_shift_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    nbins: int,
    f_cur: np.ndarray,
) -> np.ndarray:
    scale_factors = nbins * (f_cur / C_VAL)[:, np.newaxis]
    return np.abs(dparam_old - dparam_new) * scale_factors


@njit(cache=True, fastmath=True)
def branch_param(
    param_cur: float,
    dparam_cur: float,
    dparam_new: float,
    param_min: float = -np.inf,
    param_max: float = np.inf,
) -> tuple[np.ndarray, float]:
    """Refine a parameter range around a current value with a finer step size.

    This function creates a new array of parameter values centered on a specified
    current value with a desired new step size.

    Parameters
    ----------
    param_cur : float
        The current parameter value (center of the range).
    dparam_cur : float
        The current parameter step size (half-width of current range).
    dparam_new : float
        The desired new parameter step size (half-width of finer range).
    param_min : float, optional
        The minimum allowable parameter value, by default -np.inf.
    param_max : float, optional
        The maximum allowable parameter value, by default np.inf.

    Returns
    -------
    tuple[np.ndarray, float]
        Array of new parameter values and the actual new parameter step size used.

    Raises
    ------
    ValueError
        If the input parameters are invalid.
    """
    eps = 1e-12
    if dparam_cur <= eps or dparam_new <= eps:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    if param_max <= param_min + eps:
        msg = "param_max must be greater than param_min."
        raise ValueError(msg)
    param_range = (param_max - param_min) / 2.0
    if dparam_new > (param_range + eps):
        # If the desired new step size is too large, return the current value
        return np.array([param_cur]), dparam_new
    # Compute number of intervals with conservative ceil logic
    num_points = int(np.ceil(((dparam_cur + eps) / dparam_new) - eps))
    if num_points <= 0:
        msg = "Invalid input: ensure dparam_cur > dparam_new."
        raise ValueError(msg)
    # Confidence-based symmetric range shrinkage
    # 0.5 < confidence_const < 1
    confidence_const = 0.5 * (1 + 1 / num_points)
    half_range = confidence_const * dparam_cur
    n = num_points + 2  # Total number of points including outer ends
    param_arr_new = np.linspace(param_cur - half_range, param_cur + half_range, n)[1:-1]
    dparam_new_actual = dparam_cur / num_points
    return param_arr_new, dparam_new_actual


@njit(cache=True, fastmath=True)
def branch_param_padded(
    out_values: np.ndarray,  # Slice to write into (shape MAX_BRANCH_VALS,)
    param_cur: float,
    dparam_cur: float,
    dparam_new: float,
    param_min: float = -np.inf,
    param_max: float = np.inf,
) -> tuple[float, int]:
    count = 0
    dparam_new_actual = dparam_new  # Default if no branching occurs or edge cases
    eps = 1e-12
    if dparam_cur <= eps or dparam_new <= eps:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    if param_max <= param_min + eps:
        msg = "param_max must be greater than param_min."
        raise ValueError(msg)

    param_range = (param_max - param_min) / 2.0
    if dparam_new > (param_range + eps):
        # If the desired new step size is too large, return the current value
        out_values[0] = param_cur
        return dparam_new, 1
    # Compute number of intervals with conservative ceil logic
    num_points = int(np.ceil(((dparam_cur + eps) / dparam_new) - eps))
    if num_points <= 0:
        msg = "Invalid input: ensure dparam_cur > dparam_new."
        raise ValueError(msg)
    # Confidence-based symmetric range shrinkage
    # 0.5 < confidence_const < 1
    confidence_const = 0.5 * (1 + 1 / num_points)
    half_range = confidence_const * dparam_cur
    start = param_cur - half_range
    stop = param_cur + half_range
    num_intervals = num_points + 1
    step = (stop - start) / num_intervals

    # Generate points and fill the start of the padded array
    count = min(num_points, len(out_values))
    for i in range(count):
        out_values[i] = start + (i + 1) * step

    # Calculate actual dparam based on generated points
    dparam_new_actual = dparam_cur / num_points
    return dparam_new_actual, count


@njit(cache=True, fastmath=True)
def range_param(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """Generate an evenly spaced array of values between vmin and vmax.

    Endpoints are excluded. Spacing is uniform, though not guaranteed to be
    exactly dv if (vmax - vmin) is not a multiple of dv. It ensures symmetry
    at the cost of a slightly different spacing than dv.

    Parameters
    ----------
    vmin : float
        Minimum value of the parameter range.
    vmax : float
        Maximum value of the parameter range.
    dv : float
        Desired step size. Actual spacing may differ slightly.

    Returns
    -------
    np.ndarray
        Array of parameter values uniformly spaced between vmin and vmax.
    """
    eps = 1e-12
    if not (vmin < (vmax - eps) and dv > eps):
        msg = "Invalid input: ensure vmin < vmax and dv > 0."
        raise ValueError(msg)
    if dv > ((vmax - vmin) / 2 + eps):
        return np.array([(vmax + vmin) / 2])
    npoints = int((vmax - vmin) / dv)
    return np.linspace(vmin, vmax, npoints + 2)[1:-1]
