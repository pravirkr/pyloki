from __future__ import annotations

import attrs
import numpy as np
from numba import njit, vectorize

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL


@vectorize(nopython=True, cache=True)
def get_phase_idx_int(proper_time: float, freq: float, nbins: int, delay: float) -> int:
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


@vectorize(nopython=True, cache=True)
def get_phase_idx(proper_time: float, freq: float, nbins: int, delay: float) -> float:
    """Calculate the (unrounded) phase index of the proper time."""
    if freq <= 0:
        msg = "Frequency must be positive."
        raise ValueError(msg)
    if nbins <= 0:
        msg = "Number of bins must be positive."
        raise ValueError(msg)
    phase = ((proper_time + delay) * freq) % 1
    iphase = phase * nbins
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
    dphi = tol_bins / fold_bins
    k = np.arange(nparams)
    dparams_f = dphi * maths.fact(k + 1) / (tobs - t_ref) ** (k + 1)
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
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (scalar)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref)
    dparams = np.zeros(nparams, dtype=np.float64)
    dparams[:-1] = dparams_f[:-1] * C_VAL / f_max
    dparams[-1] = dparams_f[-1]
    return dparams


@njit(cache=True, fastmath=True)
def poly_taylor_step_d_vec(
    nparams: int,
    tobs: float,
    fold_bins: int,
    tol_bins: float,
    f_max: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Grid for parameters {d_k,... d_2, f} based on the Taylor expansion (vector)."""
    dparams_f = poly_taylor_step_f(nparams, tobs, fold_bins, tol_bins, t_ref)
    dparams = np.zeros((len(f_max), nparams), dtype=np.float64)
    dparams[:, :-1] = dparams_f[:-1][np.newaxis, :] * C_VAL / f_max[:, np.newaxis]
    dparams[:, -1] = dparams_f[-1]
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
    factor = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
    return abs(df_old - df_new) * factor > tol_bins


@njit(cache=True, fastmath=True)
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
    factors = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
    factors[:-1] *= f_cur / C_VAL
    return np.abs(dparam_old - dparam_new) * factors


@njit(cache=True, fastmath=True)
def poly_taylor_shift_d_vec(
    dparam_old: np.ndarray,
    dparam_new: np.ndarray,
    tobs_new: float,
    fold_bins: int,
    f_cur: np.ndarray,
    t_ref: float = 0,
) -> np.ndarray:
    """Compute the bin shift for parameters {d_k,... d_2, f} (vector)."""
    nbatch, nparams = dparam_old.shape
    k = np.arange(nparams - 1, -1, -1)
    factors = (tobs_new - t_ref) ** (k + 1) * fold_bins / maths.fact(k + 1)
    factors_broadcast = np.empty((nbatch, nparams), dtype=dparam_old.dtype)
    for i in range(nbatch):
        factors_broadcast[i, :] = factors
    # For all but last param, scale by f_cur / C_VAL
    scale = (f_cur / C_VAL)[:, np.newaxis]
    factors_broadcast[:, :-1] *= scale
    return np.abs(dparam_old - dparam_new) * factors_broadcast


@njit
def period_step(tobs: float, nbins: int, p_min: float, tol: float) -> float:
    m_cycle = tobs / p_min
    tsamp_min = p_min / nbins
    return tol * tsamp_min / (m_cycle - 1)


@njit(cache=True, fastmath=True)
def shift_params_d(
    param_vec: np.ndarray,
    delta_t: float,
    n_out: int = 3,
) -> np.ndarray:
    """Shift the kinematic taylor parameters to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., a, v, d] at reference time t_i.
        Could also be a 2D array of shape (N, n) where N is the number of vectors
        and n is the number of parameters.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.
    n_out : int, optional
        Number of output parameters from end, by default 3.

    Returns
    -------
    np.ndarray
        Parameter vector at the new reference time t_j.
        Shape is (..., n_out).
    """
    nparams = param_vec.shape[-1]
    n_out = min(n_out, nparams)
    powers = np.tril(np.arange(nparams)[:, np.newaxis] - np.arange(nparams))
    # Calculate the transformation matrix (taylor coefficients)
    t_mat = delta_t**powers / maths.fact(powers) * np.tril(np.ones_like(powers))
    t_mat = t_mat[-n_out:]
    # transform each vector in correct shape: np.dot(t_mat, param_vec)
    return param_vec @ t_mat.T


@njit(cache=True, fastmath=True)
def shift_params(param_vec: np.ndarray, delta_t: float) -> tuple[np.ndarray, float]:
    """Shift the search parameters vector to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., j, a, f] at reference time t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    tuple[np.ndarray, float]
        Parameter vector at the new reference time t_j and the phase delay d.

    Notes
    -----
    phase delay is given in units of seconds.
    """
    nparams = param_vec.shape[-1]
    dvec_cur = np.zeros(nparams + 1, dtype=param_vec.dtype)
    # Copy till acceleration
    dvec_cur[:-2] = param_vec[:-1]
    dvec_new = shift_params_d(dvec_cur, delta_t, n_out=nparams + 1)
    param_vec_new = param_vec.copy()
    param_vec_new[:-1] = dvec_new[:-2]
    param_vec_new[-1] = param_vec[-1] * (1 + dvec_new[-2] / C_VAL)
    delay_rel = dvec_new[-1] / C_VAL
    return param_vec_new, delay_rel


@njit(cache=True, fastmath=True)
def shift_params_batch(
    param_vec_batch: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialized version of shift_params for batch processing.

    Parameters
    ----------
    param_vec_batch : np.ndarray
        Parameter vector of shape (size, nparams, 2) at reference time t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Array of transformed search parameters vector at the new reference time t_j
        and the phase delay d.
    """
    size, nparams, _ = param_vec_batch.shape
    dvec_cur = np.zeros((size, nparams + 1), dtype=param_vec_batch.dtype)
    # Copy till acceleration
    dvec_cur[:, :-2] = param_vec_batch[:, :-1, 0]
    dvec_new = shift_params_d(dvec_cur, delta_t, n_out=nparams + 1)
    param_vec_new = param_vec_batch.copy()
    param_vec_new[:, :-1, 0] = dvec_new[:, :-2]
    param_vec_new[:, -1, 0] = param_vec_batch[:, -1, 0] * (1 + dvec_new[:, -2] / C_VAL)
    delay_rel = dvec_new[:, -1] / C_VAL
    return param_vec_new, delay_rel


@njit(cache=True, fastmath=True)
def shift_params_circular_batch(
    param_vec_batch: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialized version of shift_params for circular orbit batch processing.

    Parameters
    ----------
    param_vec_batch : np.ndarray
        Parameter vector of shape (size, nparams, 2) at reference time t_i.
    delta_t : float
        The time difference (t_j - t_i) to shift the parameters by.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Array of transformed search parameters vector at the new reference time t_j
        and the phase delay d.
    """
    size, nparams, _ = param_vec_batch.shape
    if nparams != 4:
        msg = "4 parameters are needed for circular orbit resolve."
        raise ValueError(msg)
    minus_omega_sq = param_vec_batch[:, 0, 0] / param_vec_batch[:, 2, 0]
    omega_batch = np.sqrt(-minus_omega_sq)
    required_order = min(int(max(omega_batch) * abs(delta_t) * np.e + 10), 100)

    dvec_cur = np.zeros((size, required_order + 1), dtype=param_vec_batch.dtype)
    # Copy till acceleration
    dvec_cur[:, -5:-2] = param_vec_batch[:, :-1, 0]
    # Now fill all the higher order derivatives
    powers = np.arange(5, required_order + 1)
    even_mask = powers % 2 == 0
    odd_mask = ~even_mask
    if np.any(even_mask):
        even_powers = powers[even_mask]
        even_exponents = (even_powers - 2) // 2
        even_coefs = (
            minus_omega_sq[:, None] ** even_exponents[None, :]
        ) * param_vec_batch[:, 2, 0][:, None]
        dvec_cur[:, -even_powers - 1] = even_coefs

    if np.any(odd_mask):
        odd_powers = powers[odd_mask]
        odd_exponents = (odd_powers - 3) // 2
        odd_coefs = (
            minus_omega_sq[:, None] ** odd_exponents[None, :]
        ) * param_vec_batch[:, 1, 0][:, None]
        dvec_cur[:, -odd_powers - 1] = odd_coefs

    dvec_new = shift_params_d(dvec_cur, delta_t, n_out=nparams + 1)
    param_vec_new = param_vec_batch.copy()
    param_vec_new[:, :-1, 0] = dvec_new[:, :-2]
    param_vec_new[:, -1, 0] = param_vec_batch[:, -1, 0] * (1 + dvec_new[:, -2] / C_VAL)
    delay_rel = dvec_new[:, -1] / C_VAL
    return param_vec_new, delay_rel


@njit(cache=True, fastmath=True)
def convert_taylor_to_circular(param_sets: np.ndarray) -> np.ndarray:
    """Convert the Taylor parameters to circular parameters.

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
    if dparam_cur <= 0 or dparam_new <= 0:
        msg = "Both dparam_cur and dparam_new must be positive."
        raise ValueError(msg)
    if param_cur < param_min or param_cur > param_max:
        msg = f"param_cur must be within [param_min, param_max], got {param_cur}."
        raise ValueError(msg)
    if dparam_new > (param_max - param_min) / 2:
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
    if not (vmin < vmax and dv > 0):
        msg = "Invalid input: ensure vmin < vmax and dv > 0."
        raise ValueError(msg)
    if dv > (vmax - vmin) / 2:
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
        data = np.argsort(np.abs(np.arange(self.nseg) - self.ref_idx))
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
