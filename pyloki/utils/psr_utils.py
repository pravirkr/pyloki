from __future__ import annotations

import numpy as np
from numba import njit, vectorize

from pyloki.utils import math
from pyloki.utils.misc import C_VAL


@vectorize(nopython=True, cache=True)
def get_phase_idx(proper_time: float, freq: float, nbins: int, delay: float) -> int:
    """Calculate the phase index of the proper time in the folded profile.

    Parameters
    ----------
    proper_time : float
        Proper time of the signal in time units.
    freq : float
        Frequency of the signal in Hz. Must be positive.
    nbins : int
        Number of bins in the folded profile. Must be positive.
    delay : float
        Signal delay due to pulsar binary motion in time units.

    Returns
    -------
    int
        Phase bin index of the proper time in the folded profile.
    """
    if freq <= 0:
        msg = "Frequency must be positive."
        raise ValueError(msg)
    if nbins <= 0:
        msg = "Number of bins must be positive."
        raise ValueError(msg)
    phase = ((proper_time + delay) * freq) % 1
    # phase is in [0, 1). Round and wrap to ensure it is in [0, nbins).
    iphase = int(phase * nbins + 0.5)
    if iphase == nbins:
        return 0
    return iphase


@njit
def param_step(
    tobs: float,
    tsamp: float,
    deriv: int,
    tol: float,
    t_ref: float = 0,
) -> float:
    """
    Calculate the parameter step size for polynomial search.

    Parameters
    ----------
    tobs : float
        Total observation time of the segment in seconds.
    tsamp : float
        Sampling time of the segment in seconds.
    deriv : int
        Derivative of the parameter (2: acceleration, 3: jerk, etc.)
    tol : float
        Tolerance parameter for the polynomial search (in bins).
    t_ref : float, optional
        Reference time in segment e.g. start, middle, etc. (default: 0)

    Returns
    -------
    float
        Optimal parameter step size
    """
    if deriv < 2:
        msg = "deriv must be >= 2"
        raise ValueError(msg)
    dparam = tsamp * math.fact(deriv) * C_VAL / (tobs - t_ref) ** deriv
    return tol * dparam


@njit
def param_step_shift(
    dparam_1: float,
    dparam_2: float,
    tobs: float,
    tsamp: float,
    deriv: int,
    tol: float,
    t_ref: float = 0,
) -> float:
    factor = (tobs - t_ref) ** deriv / (tol * tsamp * math.fact(deriv) * C_VAL)
    return abs(dparam_1 - dparam_2) * factor


@njit
def freq_step(tobs: int, nbins: int, f_max: float, tol: float) -> float:
    m_cycle = tobs * f_max
    tsamp_min = 1 / (f_max * nbins)
    return tol * f_max**2 * tsamp_min / (m_cycle - 1)


@njit
def freq_step_approx(tobs: int, f_max: float, tsamp: float, tol: float) -> float:
    m_cycle = tobs * f_max
    return tol * f_max**2 * tsamp / (m_cycle - 1)


@njit
def freq_step_shift(
    df_1: float,
    df_2: float,
    tobs: float,
    tsamp: float,
    f_cur: float,
    tol: float,
) -> float:
    m_cycle = tobs * f_cur
    factor = (m_cycle - 1) / (tol * f_cur**2 * tsamp)
    return abs(df_1 - df_2) * factor


@njit
def period_step_init(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit
def period_step(tobs: float, tsamp: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    return tol * (tsamp * 2) / (m_cycle - 1)


@njit
def period_step_shift(
    dp_1: float,
    dp_2: float,
    tobs: float,
    tsamp: float,
    p_cur: float,
    tol: float,
) -> float:
    m_cycle = tobs / p_cur
    factor = (m_cycle - 1) / (tol * (tsamp * 2))
    return abs(dp_1 - dp_2) * factor


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
    current value with a desired new step size. The new range is guaranteed to be
    within the specified minimum and maximum parameter values.

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
    if dparam_cur <= 0 or dparam_new <= 0:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    if param_cur < param_min or param_cur > param_max:
        msg = f"param_cur must be within [param_min, param_max], got {param_cur}."
        raise ValueError(msg)
    if dparam_new > (param_max - param_min) / 2:
        # If the desired new step size is too large, return the current value
        return np.array([param_cur]), dparam_new
    n = 2 + int(np.ceil(dparam_cur / dparam_new))
    if n < 3:
        msg = "Invalid input: ensure dparam_cur > dparam_new."
        raise ValueError(msg)
    # 0.5 < confidence_const < 1
    confidence_const = 0.5 * (1 + 1 / (n - 2))
    half_range = confidence_const * dparam_cur
    param_arr_new = np.linspace(param_cur - half_range, param_cur + half_range, n)[1:-1]
    dparam_new_actual = dparam_cur / (n - 2)
    return param_arr_new, dparam_new_actual


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
    if not (vmin < vmax and dv > 0):
        msg = "Invalid input: ensure vmin < vmax and dv > 0."
        raise ValueError(msg)
    if dv > (vmax - vmin) / 2:
        return np.array([(vmax + vmin) / 2])
    npoints = int((vmax - vmin) / dv)
    return np.linspace(vmin, vmax, npoints + 2)[1:-1]
