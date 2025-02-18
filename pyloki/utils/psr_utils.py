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


@njit(cache=True, fastmath=True)
def poly_taylor_step_f(
    nparams: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid size for frequency and its derivatives {f_k, ..., f}.

    Parameters
    ----------
    nparams : int
        Number of parameters in the Taylor expansion.
    tobs : float
        Total observation time in seconds.
    fold_bins : int
        Number of bins in the folded profile.
    tol_bins : float, optional
        Tolerance parameter, eta in bins, by default 1.
    t_ref : float, optional
        Reference time in segment e.g. tobs/2, etc., by default 0.

    Returns
    -------
    float
        Optimal frequency and its derivative step size in reverse order.
    """
    dparams_f = np.zeros(nparams, dtype=np.float64)
    dphi = tol_bins / fold_bins
    k = np.arange(nparams)
    dparams_f = dphi * math.fact(k + 1) / (tobs - t_ref) ** (k + 1)
    return dparams_f[::-1]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d(
    nparams: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    f_max: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion."""
    dparams_f = poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref)
    dparams = np.zeros(nparams, dtype=np.float64)
    dparams[-1] = dparams_f[-1]
    dparams[:-1] = dparams_f[:-1] * C_VAL / f_max
    return dparams


@njit
def split_f(
    df_old: float,
    df_new: float,
    tobs_new: float,
    k: int,
    fold_bins: float,
    tol_bins: float,
    t_ref: float = 0,
) -> bool:
    """Check if a parameter {f_k} should be split."""
    factor = (tobs_new - t_ref) ** (k + 1) * fold_bins / math.fact(k + 1)
    return abs(df_old - df_new) * factor > tol_bins


@njit
def poly_taylor_shift_d(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    fold_bins: int,
    f_cur: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Compute the bin shift for parameters {d_k,... d_2, f}."""
    nparams = len(dparam_old)
    k = np.arange(nparams - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * fold_bins / math.fact(k + 1)
    factors[:-1] *= f_cur / C_VAL
    return np.abs(dparam_old - dparam_new) * factors


@njit
def period_step(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit(cache=True, fastmath=True)
def shift_params(param_vec: np.ndarray, delta_t: float) -> np.ndarray:
    """Shift the kinematic taylor parameters to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., a, v, d] at reference time t_i.
        Could also be a 2D array of shape (N, n) where N is the number of vectors
        and n is the number of parameters.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    np.ndarray
        Parameter vector at the new reference time t_j.
    """
    nparams = param_vec.shape[-1]
    powers = np.tril(np.arange(nparams)[:, np.newaxis] - np.arange(nparams))
    # Calculate the transformation matrix (taylor coefficients)
    t_mat = delta_t**powers / math.fact(powers) * np.tril(np.ones_like(powers))
    # transform each vector in correct shape: np.dot(t_mat, param_vec)
    return param_vec @ t_mat.T


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
