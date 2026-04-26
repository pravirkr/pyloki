from __future__ import annotations

import math

import numpy as np
from numba import njit, vectorize

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL, FLOAT_EPSILON, ZERO_EPSILON


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
    # iphase ∈ [0, nbins), half-up rounding is intentional and deterministic
    ibin = int(np.float32(shifts) + np.float32(0.5))
    if ibin == nbins:
        ibin = 0
    return ibin


@njit(cache=True, fastmath=True)
def poly_taylor_step_f(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Grid size for frequency and its derivatives {f_k, ..., f_0}.

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
    use_cheby: bool, optional
        Whether to use Chebyshev coarsening factor, by default True.

    Returns
    -------
    float
        Optimal frequency and its derivative step size in reverse order.
    """
    dphi = eta / nbins
    k = np.arange(nparams)
    dparams_f = dphi * maths.fact(k + 1) / (tobs - t_ref) ** (k + 1)
    if use_cheby:
        dparams_f = 2**k * dparams_f
    return dparams_f[::-1]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_f(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: float,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref, use_cheby)
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
    use_cheby: bool = True,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, nbins, eta, t_ref, use_cheby)
    return dparams_f * C_VAL / f_max


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_vec(
    poly_order: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, nbins, eta, t_ref, use_cheby)
    return dparams_f[np.newaxis, :] * C_VAL / f_max[:, np.newaxis]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_f_vec(
    nparams: int,
    tobs: float,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, nbins, eta, t_ref, use_cheby)
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
    use_cheby: bool = True,
) -> np.ndarray:
    """Compute the bin shift for parameters {d_k,... d_2, f} (vector)."""
    nbatch, nparams = dparam_old.shape
    k = np.arange(nparams - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    if use_cheby:
        factors = factors / 2**k
    factors_broadcast = np.empty((nbatch, nparams), dtype=dparam_old.dtype)
    for i in range(nbatch):
        factors_broadcast[i, :] = factors
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
    use_cheby: bool = True,
) -> bool:
    """Check if a parameter {f_k} should be split."""
    factor = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    if use_cheby:
        factor = factor / 2**k
    return abs(df_old - df_new) * factor > (eta - FLOAT_EPSILON)


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    nbins: int,
    f_cur: float,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (scalar)."""
    n_params = len(dparam_old)
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    if use_cheby:
        factors = factors / 2**k
    factors *= f_cur / C_VAL
    return np.abs(dparam_old - dparam_new) * factors


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    nbins: int,
    f_cur: np.ndarray,
    t_ref: float = 0,
    use_cheby: bool = True,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (vector)."""
    n_batch, n_params = dparam_old.shape
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * nbins / maths.fact(k + 1)
    if use_cheby:
        factors = factors / 2**k
    factors_broadcast = np.empty((n_batch, n_params), dtype=dparam_old.dtype)
    for i in range(n_batch):
        factors_broadcast[i, :] = factors
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
    poly_order: int,
    nbins: int,
    eta: float,
    f_max: np.ndarray,
) -> np.ndarray:
    dphi = eta / nbins
    dparams_f = np.zeros(poly_order, np.float64) + dphi
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
) -> tuple[np.ndarray, float]:
    """Perfectly sub-divide a parent parameter cell into contiguous child cells.

    DESIGN NOTE: Exact Contiguous Splitting
    This physically partitions the parent cell into `num_points` equal sub-cells.
    The outermost edges of the extreme child cells sit perfectly flush with the
    boundaries of the parent cell, ensuring zero overlap and zero gaps between
    adjacent branches in the hierarchical tree.

    The function assumes that parameter values outside the allowed search
    domain will be handled elsewhere (e.g. in the FFA init step). Therefore
    it does not enforce parameter limits internally.

    Parameters
    ----------
    param_cur : float
        Current parameter value (centre of the parent cell).
    dparam_cur : float
        Current grid spacing of the parameter dimension. Should be trunctaed as per
        search range.
    dparam_new : float
        Desired grid spacing for the refined search stage. The actual spacing
        used may differ slightly in order to maintain symmetry.

    Returns
    -------
    tuple[np.ndarray, float]
        param_arr_new : np.ndarray
            Array of refined parameter centres generated around ``param_cur``.

        dparam_new_actual : float
            Actual spacing between the returned parameter centres.

    Raises
    ------
    ValueError
        If the provided spacings are not positive.
    """
    if dparam_cur <= ZERO_EPSILON or dparam_new <= ZERO_EPSILON:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    # How many target spans fit inside the parent span?
    # If dparam_new >= dparam_cur, then num_points = 1
    ratio = dparam_cur / dparam_new
    num_points = max(1, math.ceil(ratio - FLOAT_EPSILON))
    # Exact physical span of the newly created child cells
    dparam_new_actual = dparam_cur / num_points

    # Find the absolute minimum boundary of the parent cell
    parent_min = param_cur - (dparam_cur / 2.0)

    # Place the center of the first child cell exactly half a sub-span inward
    first_center = parent_min + (dparam_new_actual / 2.0)
    # Generate all child centers
    out_values = np.zeros(num_points, dtype=np.float64)
    for i in range(num_points):
        out_values[i] = first_center + i * dparam_new_actual

    return out_values, dparam_new_actual


@njit(cache=True, fastmath=True)
def branch_param_padded(
    out_values: np.ndarray,  # Slice to write into (shape MAX_BRANCH_VALS,)
    param_cur: float,
    dparam_cur: float,
    dparam_new: float,
) -> tuple[float, int]:
    """Generate parameters as `branch_param`, but for padded arrays."""
    if dparam_cur <= ZERO_EPSILON or dparam_new <= ZERO_EPSILON:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    # How many target spans fit inside the parent span?
    # If dparam_new >= dparam_cur, then num_points = 1
    ratio = dparam_cur / dparam_new
    num_points = max(1, math.ceil(ratio - FLOAT_EPSILON))
    branch_max = len(out_values)
    if num_points > branch_max:
        msg = "Invalid input: increase branch_max."
        raise ValueError(msg)

    # Exact physical span of the newly created child cells
    dparam_new_actual = dparam_cur / num_points

    # Find the absolute minimum boundary of the parent cell
    parent_min = param_cur - (dparam_cur / 2.0)

    # Place the center of the first child cell exactly half a sub-span inward
    first_center = parent_min + (dparam_new_actual / 2.0)
    # Generate all child centers
    for i in range(num_points):
        out_values[i] = first_center + i * dparam_new_actual

    return dparam_new_actual, num_points


@njit(cache=True, fastmath=True)
def branch_dparam_crackle(
    dparam_cur: float,
    dparam_new: float,
    branch_max: int,
) -> float:
    if dparam_cur <= ZERO_EPSILON or dparam_new <= ZERO_EPSILON:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    # Compute number of intervals with conservative ceil logic
    ratio = dparam_cur / dparam_new
    num_points = max(1, math.ceil(ratio - FLOAT_EPSILON))
    if num_points > branch_max:
        msg = "Invalid input: increase branch_max."
        raise ValueError(msg)
    # Calculate actual dparam based on generated points
    return dparam_cur / num_points


@njit(cache=True, fastmath=True)
def range_param_count(vmin: float, vmax: float, dv: float) -> int:
    """Calculate the number of points generated by range_param.

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
    int
        Number of points in the parameter range.
    """
    if not (vmin < vmax and dv > 0.0):
        msg = "Invalid input: ensure vmin < vmax and dv > 0.0."
        raise ValueError(msg)
    if dv >= (vmax - vmin):
        return 1
    return math.ceil((vmax - vmin) / dv)


@njit(cache=True, fastmath=True)
def range_param(vmin: float, vmax: float, dv: float) -> np.ndarray:
    """Generate an array of cell centres that perfectly tile the parameter space.

    DESIGN NOTE: Exact Outset Gridding
    Older versions used `np.linspace(..., n+2)[1:-1]`, which created an "inset"
    grid that left unsearched gaps at `vmin` and `vmax`.

    This updated logic recalculates the actual step size (`dv_actual`) based on
    the ceiling count. It places the first centre exactly `dv_actual / 2` away
    from `vmin`, and the last centre `dv_actual / 2` away from `vmax`. When combined
    with their spans during hierarchical stitching, the physical edges of the
    outermost cells align flawlessly with `vmin` and `vmax`,
    ensuring 100% space coverage.

    Parameters
    ----------
    vmin : float
        Minimum boundary of the parameter space.
    vmax : float
        Maximum boundary of the parameter space.
    dv : float
        Desired step size. Actual spacing will be <= dv to ensure perfect tiling.

    Returns
    -------
    np.ndarray
        Array of parameter cell centres uniformly spaced.
    """
    if not (vmin < vmax and dv > 0):
        msg = "Invalid input: ensure vmin < vmax and dv > 0."
        raise ValueError(msg)
    if dv >= (vmax - vmin):
        return np.array([(vmax + vmin) / 2.0])
    npoints = math.ceil((vmax - vmin) / dv)
    dv_actual = (vmax - vmin) / npoints
    return np.linspace(vmin + (dv_actual / 2.0), vmax - (dv_actual / 2.0), npoints)


@njit(cache=True, fastmath=True)
def get_nearest_indices_analytical(
    param_set: np.ndarray,
    param_grid_count: np.ndarray,
    param_limits: np.ndarray,
) -> np.ndarray:
    """Calculate the nearest index in the parameter grid for a given parameter set.

    Parameters
    ----------
    param_set : np.ndarray
        The parameter set to calculate the nearest index for.
    param_grid_count : np.ndarray
        The number of points in the parameter grid.
    param_limits : np.ndarray
        The limits of the parameter grid, shape (nparams, 2).

    Returns
    -------
    np.ndarray
        The nearest index in the parameter grid for each parameter in the set.

    Notes
    -----
    Grid: grid[i] = vmin + (i + 0.5) * step, step = range / n_grid.
    Bin index via direct truncation: idx = int(n_grid * (val - vmin) / range).
    """
    nparams = len(param_set)
    pindex = np.zeros(nparams, dtype=np.int64)

    for ip in range(nparams):
        n_grid = param_grid_count[ip]
        # Guard against single-point or zero-point grids
        if n_grid <= 1:
            pindex[ip] = 0
            continue
        val = param_set[ip]
        vmin = param_limits[ip][0]
        vmax = param_limits[ip][1]
        # Map value to exact bin index
        raw_idx = n_grid * (val - vmin) / (vmax - vmin)
        # Pure truncation to map floating position into integer bin
        idx = int(raw_idx + FLOAT_EPSILON)
        if idx < 0:
            idx = 0
        elif idx >= n_grid:
            idx = n_grid - 1
        pindex[ip] = idx
    return pindex


@njit(cache=True, fastmath=True)
def get_nearest_indices_2d_batch(
    accel_batch: np.ndarray,
    freq_batch: np.ndarray,
    param_grid_count: np.ndarray,
    param_limits: np.ndarray,
) -> np.ndarray:
    n_batch = accel_batch.shape[0]
    n_params = param_limits.shape[0]
    n_accel = param_grid_count[-2]
    n_freq = param_grid_count[-1]
    accel_min, accel_max = param_limits[-2]
    freq_min, freq_max = param_limits[-1]
    out = np.zeros((n_batch, n_params), dtype=np.int64)
    step_inv_accel = n_accel / (accel_max - accel_min) if accel_max > accel_min else 0.0
    step_inv_freq = n_freq / (freq_max - freq_min) if freq_max > freq_min else 0.0
    for i in range(n_batch):
        if step_inv_accel > 0.0:
            raw_accel_idx = (accel_batch[i] - accel_min) * step_inv_accel
            accel_idx = int(raw_accel_idx + FLOAT_EPSILON)
            if accel_idx < 0:
                accel_idx = 0
            elif accel_idx >= n_accel:
                accel_idx = n_accel - 1
            out[i, -2] = accel_idx
        else:
            out[i, -2] = 0
        if step_inv_freq > 0.0:
            raw_freq_idx = (freq_batch[i] - freq_min) * step_inv_freq
            freq_idx = int(raw_freq_idx + FLOAT_EPSILON)
            if freq_idx < 0:
                freq_idx = 0
            elif freq_idx >= n_freq:
                freq_idx = n_freq - 1
            out[i, -1] = freq_idx
        else:
            out[i, -1] = 0
    return out
