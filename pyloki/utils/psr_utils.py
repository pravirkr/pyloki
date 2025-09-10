from __future__ import annotations

import attrs
import numpy as np
from numba import njit, vectorize

from pyloki.utils import maths, transforms
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
    phase = ((proper_time - delay) * freq) % 1.0
    iphase = phase * float(nbins)
    # Clamp to ensure iphase ∈ [0, nbins)
    eps = 1e-12
    if iphase >= (nbins - eps):
        return iphase - nbins
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
    dphi = tol_bins / fold_bins
    k = np.arange(nparams)
    dparams_f = dphi * maths.fact(k + 1) / (tobs - t_ref) ** (k + 1)
    dparams_f_opt = 2**k * dparams_f
    return dparams_f_opt[::-1]


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_f(
    nparams: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    f_max: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref)
    dparams = np.zeros(nparams, dtype=np.float64)
    dparams[:-1] = dparams_f[:-1] * C_VAL / f_max
    dparams[-1] = dparams_f[-1]
    return dparams


@njit(cache=True, fastmath=True)
def poly_taylor_step_d(
    poly_order: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    f_max: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, fold_bins, tol_bins, t_ref)
    return dparams_f * C_VAL / f_max


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_vec(
    poly_order: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    f_max: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Parameter grid for {d_k_max,... d_2, d_1} as per Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(poly_order, tobs, fold_bins, tol_bins, t_ref)
    return dparams_f[np.newaxis, :] * C_VAL / f_max[:, np.newaxis]


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
    factor = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
    factor_opt = factor / 2**k
    eps = 1e-6
    return abs(df_old - df_new) * factor_opt > (tol_bins - eps)


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    fold_bins: int,
    f_cur: float,
    t_ref: float = 0,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (scalar)."""
    n_params = len(dparam_old)
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
    factors_opt = factors / 2**k
    factors_opt *= f_cur / C_VAL
    return np.abs(dparam_old - dparam_new) * factors_opt


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    fold_bins: int,
    f_cur: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Bin shift for parameters {d_k_max,... d_2, d_1} (vector)."""
    n_batch, n_params = dparam_old.shape
    k = np.arange(n_params - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
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
    fold_bins: int,
    tol_bins: float,
    f_max: np.ndarray,
) -> np.ndarray:
    dphi = tol_bins / fold_bins
    dparams_f = np.zeros(nparams, np.float64) + dphi
    return dparams_f[np.newaxis, :] * C_VAL / f_max[:, np.newaxis]


@njit(cache=True, fastmath=True)
def poly_cheb_shift_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    fold_bins: int,
    f_cur: np.ndarray,
) -> np.ndarray:
    scale_factors = fold_bins * (f_cur / C_VAL)[:, np.newaxis]
    return np.abs(dparam_old - dparam_new) * scale_factors


@njit(cache=True, fastmath=True)
def report_leaves_taylor_batch(leaves_batch: np.ndarray) -> np.ndarray:
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
def report_leaves_chebyshev_batch(
    leaves_batch: np.ndarray,
    coord_mid: tuple[float, float],
) -> np.ndarray:
    cheby_coeffs_batch = leaves_batch[:, :-1, :]
    f0_batch = leaves_batch[:, -1, 0]
    _, scale = coord_mid
    param_sets_batch_d = transforms.cheby_to_taylor_full(cheby_coeffs_batch, scale)
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
    param_range = (param_max - param_min) / 2
    if dparam_new > (param_range + eps):
        # If the desired new step size is too large, return the current value
        out_values[0] = param_cur
        dparam_new_actual = dparam_cur
        return dparam_new_actual, 1
    n = 2 + int(np.ceil(dparam_cur / dparam_new))
    num_points = n - 2  # Actual number of branched points to generate
    if num_points <= 0:
        msg = "Invalid input: ensure dparam_cur > dparam_new."
        raise ValueError(msg)
    # Calculate the actual branched values
    # 0.5 < confidence_const < 1
    confidence_const = 0.5 * (1 + 1 / num_points)
    half_range = confidence_const * dparam_cur
    start = param_cur - half_range
    stop = param_cur + half_range
    num_intervals = n - 1
    step = (stop - start) / num_intervals

    # Generate points and fill the start of the padded array
    current_val = start + step
    count = min(num_points, len(out_values))
    for i in range(count):
        out_values[i] = current_val
        current_val += step

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


@attrs.frozen(auto_attribs=True, kw_only=True)
class SnailScheme:
    """A utility class for indexing segments in a hierarchical pruning algorithm.

    The scheme allow for "middle-out" enumeration of the segments.

    Parameters
    ----------
    nseg : int
        The total number of segments in the hierarchical search scheme.
    ref_idx : int
        The reference (starting) segment index for pruning.
    tseg : float, optional
        The duration of each segment in seconds, default is 1.0.
    """

    nseg: int = attrs.field(
        validator=[
            attrs.validators.instance_of((int, np.integer)),
            attrs.validators.gt(0),
        ],
    )
    ref_idx: int = attrs.field(
        validator=[
            attrs.validators.instance_of((int, np.integer)),
            attrs.validators.ge(0),
        ],
    )
    tseg: float = attrs.field(default=1.0, validator=attrs.validators.ge(0))
    data: np.ndarray = attrs.field(init=False)

    @ref_idx.validator
    def check_ref_idx(self, attribute: attrs.Attribute, value: int) -> None:  # noqa: ARG002
        if value >= self.nseg:
            msg = f"ref_idx must be less than nseg ({self.nseg}), got {value}."
            raise ValueError(msg)

    def __attrs_post_init__(self) -> None:
        data = np.argsort(np.abs(np.arange(self.nseg) - self.ref_idx), kind="stable")
        object.__setattr__(self, "data", data)

    @property
    def ref(self) -> float:
        """Reference time at the middle of the reference segment in seconds."""
        return (self.ref_idx + 0.5) * self.tseg

    def get_idx(self, level: int) -> int:
        """Get the segment index at the specified hierarchical level.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        int
            The segment index at the given level.
        """
        if level < 0 or level >= self.nseg:
            msg = f"level must be in [0, {self.nseg}), got {level}."
            raise ValueError(msg)
        return self.data[level]

    def get_coord(self, level: int) -> tuple[float, float]:
        """Get the current coord (ref and scale) at the given level.

        The reference time is the center of the time interval covered by all segments
        from level 0 to the specified level. The scale is the half-width of this
        interval.

        Parameters
        ----------
        level : int
            The current hierarchical level, where 0 is the reference segment.

        Returns
        -------
        tuple[float, float]
            The reference and scale for the current level in seconds.
        """
        if level < 0 or level >= self.nseg:
            msg = f"level must be in [0, {self.nseg - 1}], got {level}."
            raise ValueError(msg)
        scheme_till_now = self.data[: level + 1]
        ref = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        scale = ref - np.min(scheme_till_now)
        return ref * self.tseg, scale * self.tseg

    def get_seg_coord(self, level: int) -> tuple[float, float]:
        """Get the ref and scale for the segment (to be added) at the given level.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        tuple[float, float]
            The reference and scale for the segment at the given level in seconds.
        """
        if level < 0 or level >= self.nseg:
            msg = f"level must be in [0, {self.nseg - 1}], got {level}."
            raise ValueError(msg)
        ref = (self.get_idx(level) + 0.5) * self.tseg
        scale = 0.5 * self.tseg
        return ref, scale

    def get_current_coord(self, level: int) -> tuple[float, float]:
        """Get adaptive reference time and maximum distance from reference."""
        if level < 0 or level >= self.nseg:
            msg = f"level must be in [0, {self.nseg - 1}], got {level}."
            raise ValueError(msg)

        if level == 0:
            return self.get_coord(level)
        prev_ref, _ = self.get_coord(level - 1)
        _, cur_scale = self.get_coord(level)
        return prev_ref, cur_scale

    def get_valid(self, prune_level: int) -> tuple[float, float]:
        scheme_till_now = self.data[:prune_level]
        return np.min(scheme_till_now), np.max(scheme_till_now)

    def get_delta(self, level: int) -> float:
        """Get the difference between the current coord and the reference.

        This measures the shift of the current coord from the reference coord.

        Parameters
        ----------
        level : int
            The hierarchical level, where 0 is the reference segment.

        Returns
        -------
        float
            The difference between the current coord and the reference in seconds.
        """
        return self.get_coord(level)[0] - self.ref
