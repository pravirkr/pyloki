"""Global Fourier and coherent P-FFA cube orbital parameter fitting.

Provides:
- Global linear least-squares circular and multi-harmonic eccentric phase-curve
  fitting (``fit_phase_curve_fourier`` -> ``FourierOrbitFit``).
- Numba-accelerated coherent P-FFA fold-cube shift-and-add orbital grid search
  and inspection (``ParameterRecovery``), consuming
  ``KeplerianParamLimits.generate_grid`` and supporting multi-epoch cubes.
- Outlier-resistant Winsorized Doppler frequency-curve fitting
  (``fit_keplerian_periods_winsorized``) and multi-start exact Keplerian
  phase-delay polishing (``fit_keplerian_multistart``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np
from numba import njit, prange
from scipy import optimize

from pyloki import kepler
from pyloki.detection import scoring
from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import typing as npt


@attrs.frozen(auto_attribs=True, kw_only=True)
class FourierOrbitFit:
    """Result of ``fit_phase_curve_fourier``."""

    p_orb: float
    omega_orb: float
    x_orb: float
    phi_orb: float
    ecc_approx: float
    df_over_f: float
    rms_residual: float
    amplitudes: npt.NDArray[np.float64]
    phases: npt.NDArray[np.float64]
    coeffs: npt.NDArray[np.float64]
    coeff_uncertainties: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    omega_grid: npt.NDArray[np.float64]
    rms_curve: npt.NDArray[np.float64]
    t_center: float


@attrs.frozen(auto_attribs=True, kw_only=True)
class OrbitInspection:
    """Per-segment fold extraction along a trial Keplerian orbit."""

    params: npt.NDArray[np.float64]
    snr: float
    integrated_fold: npt.NDArray[np.float32]
    segment_folds: npt.NDArray[np.float32]
    a_indices: npt.NDArray[np.int64]
    f_indices: npt.NDArray[np.int64]
    phase_shifts: npt.NDArray[np.int64]


@attrs.frozen(auto_attribs=True, kw_only=True)
class KeplerianFitResult:
    """Result of nonlinear Keplerian phase-delay or Doppler frequency fitting."""

    f0: float
    p_orb: float
    x_orb: float
    ecc: float
    aop: float
    phi: float
    loss: float
    rms_residual: float
    residuals: npt.NDArray[np.float64]
    success: bool


def _fourier_design_matrix(
    t_centered: npt.NDArray[np.float64],
    t_scale: float,
    omega: float,
    n_harmonics: int,
) -> npt.NDArray[np.float64]:
    n = t_centered.size
    mat = np.empty((n, 2 + 2 * n_harmonics), dtype=np.float64)
    mat[:, 0] = 1.0
    mat[:, 1] = t_centered / t_scale
    for k in range(1, n_harmonics + 1):
        phase = (k * omega) * t_centered
        mat[:, 2 + 2 * (k - 1)] = np.cos(phase)
        mat[:, 2 + 2 * (k - 1) + 1] = np.sin(phase)
    return mat


def _solve_fourier_omega(
    t_centered: npt.NDArray[np.float64],
    t_scale: float,
    y: npt.NDArray[np.float64],
    omega: float,
    n_harmonics: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    design = _fourier_design_matrix(t_centered, t_scale, omega, n_harmonics)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coeffs
    rms = math.sqrt(float(np.mean(residuals * residuals)))
    return design, coeffs, rms


def fit_phase_curve_fourier(
    times: npt.ArrayLike,
    delays: npt.ArrayLike,
    p_orb_range: tuple[float, float],
    *,
    fit_eccentric: bool = False,
    n_harmonics: int = 6,
    fit_mask: npt.NDArray[np.bool_] | None = None,
    n_omega: int | None = None,
    om_tol: float = 0.05,
    polish: bool = True,
) -> FourierOrbitFit:
    r"""Fit unwrapped orbital arrival-time delays via global linear least squares.

    Sweeps trial orbital frequencies ``omega`` across ``p_orb_range`` solving
    each trial via QR/SVD ``np.linalg.lstsq`` on mean-centered, unit-scaled
    times (avoiding normal-equation ill-conditioning), then optionally polishes
    ``omega`` via 1D bounded Brent optimization.

    Parameters
    ----------
    times : array_like
        Segment midpoints in seconds.
    delays : array_like
        Unwrapped orbital time delays :math:`\tau_i = z(t_i) / c` in seconds.
    p_orb_range : tuple[float, float]
        Orbital period search interval ``(p_orb_min, p_orb_max)`` in seconds.
    fit_eccentric : bool, optional
        If ``True``, fit ``n_harmonics`` (default 6) Fourier harmonics to
        capture eccentric orbits and estimate ``ecc_approx``; if ``False``,
        fit the 1-harmonic circular model.
    n_harmonics : int, optional
        Number of harmonics when ``fit_eccentric=True``, by default 6.
    fit_mask : NDArray[np.bool_] | None, optional
        Optional boolean mask selecting valid (non-eclipsed) segments.
    n_omega : int | None, optional
        Explicit number of frequency grid points. If ``None``, derived from
        ``om_tol / t_span``.
    om_tol : float, optional
        Phase step in radians across the observation span when ``n_omega`` is
        ``None``, by default 0.05.
    polish : bool, optional
        Whether to refine the winning ``omega`` with bounded Brent minimization,
        by default ``True``.

    Returns
    -------
    FourierOrbitFit
        Best-fit circular/Fourier orbital elements, uncertainties, and RMS curve.
    """
    t_all = np.asarray(times, dtype=np.float64)
    y_all = np.asarray(delays, dtype=np.float64)
    if t_all.ndim != 1 or t_all.shape != y_all.shape:
        msg = "times and delays must be 1D arrays of matching shape"
        raise ValueError(msg)
    p_min, p_max = p_orb_range
    if p_min <= 0.0 or p_max < p_min:
        msg = f"Invalid p_orb_range: {p_orb_range}"
        raise ValueError(msg)

    k_harm = n_harmonics if fit_eccentric else 1
    if k_harm < 1:
        msg = f"n_harmonics must be >= 1, got {k_harm}"
        raise ValueError(msg)

    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=np.bool_)
        if mask.shape != t_all.shape:
            msg = "fit_mask must match times shape"
            raise ValueError(msg)
        t_fit = t_all[mask]
        y_fit = y_all[mask]
    else:
        t_fit = t_all
        y_fit = y_all

    n_params = 2 + 2 * k_harm
    if t_fit.size < n_params:
        msg = f"Need at least {n_params} valid segments for fit, got {t_fit.size}"
        raise ValueError(msg)

    t_center = float(np.mean(t_fit))
    t_centered = t_fit - t_center
    t_span = float(np.max(t_centered) - np.min(t_centered))
    if t_span <= 0.0:
        msg = "Valid times must span a positive duration"
        raise ValueError(msg)
    t_scale = max(float(np.max(np.abs(t_centered))), 1.0)

    om_lo = 2.0 * math.pi / p_max
    om_hi = 2.0 * math.pi / p_min
    if om_lo == om_hi:
        omega_grid = np.array([om_lo], dtype=np.float64)
        d_omega = 0.0
    elif n_omega is not None:
        if n_omega < 1:
            msg = f"n_omega must be >= 1, got {n_omega}"
            raise ValueError(msg)
        omega_grid = np.linspace(om_lo, om_hi, n_omega, dtype=np.float64)
        d_omega = (om_hi - om_lo) / max(n_omega - 1, 1)
    else:
        d_omega = max(om_tol / t_span, 1e-12)
        n_pts = max(2, math.ceil((om_hi - om_lo) / d_omega) + 1)
        omega_grid = np.linspace(om_lo, om_hi, n_pts, dtype=np.float64)

    rms_curve = np.empty(omega_grid.size, dtype=np.float64)
    for idx, om in enumerate(omega_grid):
        _, _, rms_curve[idx] = _solve_fourier_omega(
            t_centered,
            t_scale,
            y_fit,
            float(om),
            k_harm,
        )

    best_idx = int(np.argmin(rms_curve))
    best_om = float(omega_grid[best_idx])

    if polish and omega_grid.size > 1 and d_omega > 0.0:
        lo_b = max(om_lo, best_om - 2.0 * d_omega)
        hi_b = min(om_hi, best_om + 2.0 * d_omega)
        span_b = hi_b - lo_b
        if span_b > 0.0:
            res_opt = optimize.minimize_scalar(
                lambda u: _solve_fourier_omega(
                    t_centered,
                    t_scale,
                    y_fit,
                    lo_b + float(u) * span_b,
                    k_harm,
                )[2],
                bounds=(0.0, 1.0),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if float(res_opt.fun) <= float(rms_curve[best_idx]):
                best_om = lo_b + float(res_opt.x) * span_b

    design, coeffs, best_rms = _solve_fourier_omega(
        t_centered,
        t_scale,
        y_fit,
        best_om,
        k_harm,
    )
    residuals = y_fit - design @ coeffs

    # Unscale linear slope back to seconds/second
    coeffs_phys = coeffs.copy()
    coeffs_phys[1] /= t_scale

    # Linear 1-sigma parameter uncertainties from (X^T X)^+ * sigma^2
    dof = max(1, t_fit.size - n_params)
    sigma2 = float(np.sum(residuals * residuals)) / float(dof)
    xtx_pinv = np.linalg.pinv(design.T @ design)
    diag_var = np.maximum(np.diag(xtx_pinv) * sigma2, 0.0)
    uncertainties = np.sqrt(diag_var)
    uncertainties[1] /= t_scale

    amps = np.empty(k_harm, dtype=np.float64)
    phases = np.empty(k_harm, dtype=np.float64)
    for k in range(k_harm):
        ck = float(coeffs_phys[2 + 2 * k])
        sk = float(coeffs_phys[2 + 2 * k + 1])
        amps[k] = math.hypot(ck, sk)
        phases[k] = math.atan2(ck, sk)

    x_orb = float(amps[0])
    # z(t)/c = x * sin(omega * t + phi_0) = A1 * sin(omega*(t - t_c) + psi_1)
    phi_orb = float((phases[0] - best_om * t_center) % (2.0 * math.pi))
    ecc_approx = (
        min(0.85, 2.0 * float(amps[1]) / x_orb) if k_harm >= 2 and x_orb > 0.0 else 0.0
    )

    return FourierOrbitFit(
        p_orb=2.0 * math.pi / best_om,
        omega_orb=best_om,
        x_orb=x_orb,
        phi_orb=phi_orb,
        ecc_approx=ecc_approx,
        df_over_f=-float(coeffs_phys[1]),
        rms_residual=best_rms,
        amplitudes=amps,
        phases=phases,
        coeffs=coeffs_phys,
        coeff_uncertainties=uncertainties,
        residuals=residuals,
        omega_grid=omega_grid,
        rms_curve=rms_curve,
        t_center=t_center,
    )


@njit(cache=True, fastmath=True)
def _nearest_index(grid: np.ndarray, val: float) -> int:
    n = grid.size
    if n <= 1:
        return 0
    if val <= grid[0]:
        return 0
    if val >= grid[n - 1]:
        return n - 1
    idx = int(np.searchsorted(grid, val))
    if idx > 0 and abs(val - grid[idx - 1]) <= abs(grid[idx] - val):
        return idx - 1
    return idx


@njit(cache=True, fastmath=True)
def resolve_keplerian_trajectory(
    segment_times: np.ndarray,
    f0_psr: float,
    p_orb: float,
    x_orb: float,
    ecc: float,
    aop: float,
    phi: float,
    a_grid: np.ndarray,
    f_grid: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map an exact Keplerian orbit onto per-segment P-FFA grid indices and shifts."""
    z_arr, v_arr, a_arr = kepler.keplerian_z_derivatives_exact(
        segment_times,
        p_orb,
        ecc,
        phi,
        x_orb * C_VAL,
        aop,
        0.5 * np.pi,
    )
    nseg = segment_times.size
    a_idx = np.empty(nseg, dtype=np.int64)
    f_idx = np.empty(nseg, dtype=np.int64)
    phase_shifts = np.empty(nseg, dtype=np.int64)

    t0 = segment_times[0]
    z0 = z_arr[0]
    for i in range(nseg):
        f_inst = f0_psr * (1.0 - v_arr[i] / C_VAL)
        a_idx[i] = _nearest_index(a_grid, a_arr[i])
        f_idx[i] = _nearest_index(f_grid, f_inst)
        proper_dt = (segment_times[i] - t0) - (z_arr[i] - z0) / C_VAL
        cyc = (proper_dt * f0_psr) % 1.0
        if cyc < 0.0:
            cyc += 1.0
        shift = round(cyc * nbins) % nbins
        phase_shifts[i] = shift
    return a_idx, f_idx, phase_shifts


@njit(cache=True, fastmath=True, parallel=True)
def _eval_keplerian_grid_batch(
    cubes: np.ndarray,
    segment_times: np.ndarray,
    a_grid: np.ndarray,
    f_grid: np.ndarray,
    param_grid: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Shift-and-add P-FFA fold cubes along trial Keplerian orbits in parallel."""
    n_trials = param_grid.shape[0]
    nseg = cubes.shape[0]
    nbins = cubes.shape[4]
    scores = np.empty(n_trials, dtype=np.float32)

    for itrial in prange(n_trials):
        p_pul = param_grid[itrial, 0]
        p_orb = param_grid[itrial, 1]
        x_orb = param_grid[itrial, 2]
        ecc = param_grid[itrial, 3]
        aop = param_grid[itrial, 4]
        phi = param_grid[itrial, 5]
        f0 = 1.0 / p_pul

        a_idx, f_idx, shifts = resolve_keplerian_trajectory(
            segment_times,
            f0,
            p_orb,
            x_orb,
            ecc,
            aop,
            phi,
            a_grid,
            f_grid,
            nbins,
        )
        acc_e = np.zeros(nbins, dtype=np.float32)
        acc_v = np.zeros(nbins, dtype=np.float32)
        for iseg in range(nseg):
            ai = a_idx[iseg]
            fi = f_idx[iseg]
            sh = shifts[iseg]
            for b in range(nbins):
                src = (b - sh) % nbins
                acc_e[b] += cubes[iseg, ai, fi, 0, src]
                acc_v[b] += cubes[iseg, ai, fi, 1, src]
        scores[itrial] = np.float32(
            np.max(scoring.boxcar_snr_glrt_1d(acc_e, acc_v, widths)),
        )
    return scores


class ParameterRecovery:
    """Coherent P-FFA fold-cube shift-and-add Keplerian orbital refiner.

    Concatenates one or more P-FFA fold cubes of shape
    ``(n_seg_m, n_acc, n_freq, 2, n_bins)`` with arbitrary ``segment_times``
    (supporting multi-epoch coherent linking across gaps) and scores trial
    6D Keplerian grids from ``KeplerianParamLimits.generate_grid``.
    """

    def __init__(
        self,
        cubes: npt.NDArray[np.float32],
        segment_times: npt.NDArray[np.float64],
        accel_grid: npt.NDArray[np.float64],
        freq_grid: npt.NDArray[np.float64],
        widths: npt.NDArray[np.int64] | None = None,
    ) -> None:
        arr = np.ascontiguousarray(cubes, dtype=np.float32)
        if arr.ndim != 5 or arr.shape[3] != 2:
            msg = (
                "cubes must have shape (n_segments, n_acc, n_freq, 2, n_bins), "
                f"got {arr.shape}"
            )
            raise ValueError(msg)
        times = np.ascontiguousarray(segment_times, dtype=np.float64)
        if times.ndim != 1 or times.size != arr.shape[0]:
            msg = "segment_times must be 1D with length matching cubes.shape[0]"
            raise ValueError(msg)
        self._cubes = arr
        self._segment_times = times
        self._accel_grid = np.ascontiguousarray(accel_grid, dtype=np.float64)
        self._freq_grid = np.ascontiguousarray(freq_grid, dtype=np.float64)
        self._widths = (
            scoring.generate_box_width_trials(arr.shape[4])
            if widths is None
            else np.ascontiguousarray(widths, dtype=np.int64)
        )

    @classmethod
    def from_pffa_cubes(
        cls,
        cubes: Sequence[npt.NDArray[np.float32]],
        segment_times: Sequence[npt.ArrayLike],
        accel_grid: npt.ArrayLike,
        freq_grid: npt.ArrayLike,
        *,
        widths: npt.NDArray[np.int64] | None = None,
    ) -> ParameterRecovery:
        """Build a ``ParameterRecovery`` instance from one or more P-FFA fold cubes."""
        if len(cubes) == 0 or len(cubes) != len(segment_times):
            msg = "cubes and segment_times must be non-empty sequences of equal length"
            raise ValueError(msg)
        cat_cubes = np.concatenate(
            [np.asarray(c, dtype=np.float32) for c in cubes],
            axis=0,
        )
        cat_times = np.concatenate(
            [np.asarray(t, dtype=np.float64) for t in segment_times],
            axis=0,
        )
        return cls(
            cat_cubes,
            cat_times,
            np.asarray(accel_grid, dtype=np.float64),
            np.asarray(freq_grid, dtype=np.float64),
            widths=widths,
        )

    @property
    def nsegments(self) -> int:
        return int(self._cubes.shape[0])

    @property
    def nbins(self) -> int:
        return int(self._cubes.shape[4])

    def inspect_orbit(self, params: npt.ArrayLike) -> OrbitInspection:
        """Extract aligned per-segment folds and integrated S/N for one orbit.

        Parameters
        ----------
        params : array_like
            6-vector ``[p_pul, p_orb, x_orb, ecc, aop, phi]``.
        """
        p = np.asarray(params, dtype=np.float64)
        if p.shape != (6,):
            msg = (
                "params must be a 6-vector [p_pul, p_orb, x_orb, ecc, aop, phi], "
                f"got {p.shape}"
            )
            raise ValueError(msg)
        a_idx, f_idx, shifts = resolve_keplerian_trajectory(
            self._segment_times,
            1.0 / float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
            float(p[5]),
            self._accel_grid,
            self._freq_grid,
            self.nbins,
        )
        seg_folds = np.empty((self.nsegments, 2, self.nbins), dtype=np.float32)
        for iseg in range(self.nsegments):
            raw = self._cubes[iseg, a_idx[iseg], f_idx[iseg]]
            seg_folds[iseg] = np.roll(raw, int(shifts[iseg]), axis=-1)
        integrated = np.sum(seg_folds, axis=0)
        snr = float(
            np.max(
                scoring.boxcar_snr_glrt_1d(
                    np.ascontiguousarray(integrated[0], dtype=np.float32),
                    np.ascontiguousarray(integrated[1], dtype=np.float32),
                    self._widths,
                ),
            ),
        )
        return OrbitInspection(
            params=p,
            snr=snr,
            integrated_fold=integrated.astype(np.float32),
            segment_folds=seg_folds,
            a_indices=a_idx,
            f_indices=f_idx,
            phase_shifts=shifts,
        )

    def solve_keplerian_grid(
        self,
        param_grid: npt.NDArray[np.float64],
        *,
        top_k: int = 100,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        """Score a 6D Keplerian parameter grid and return the ``top_k`` solutions.

        Parameters
        ----------
        param_grid : NDArray[np.float64]
            Array of shape ``(..., 6)`` with columns
            ``[p_pul, p_orb, x_orb, ecc, aop, phi]`` (e.g. from
            ``KeplerianParamLimits.generate_grid``).
        top_k : int, optional
            Number of highest-S/N candidates to return, by default 100.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float64]]
            ``(top_scores, top_params)`` sorted descending by S/N.
        """
        flat = np.ascontiguousarray(
            np.asarray(param_grid, dtype=np.float64).reshape(-1, 6),
        )
        if flat.shape[0] == 0:
            return np.empty(0, dtype=np.float32), np.empty((0, 6), dtype=np.float64)
        scores = _eval_keplerian_grid_batch(
            self._cubes,
            self._segment_times,
            self._accel_grid,
            self._freq_grid,
            flat,
            self._widths,
        )
        k = min(max(1, top_k), flat.shape[0])
        order = np.argsort(scores)[::-1][:k]
        return scores[order].astype(np.float32), flat[order]


@njit(cache=True, fastmath=True)
def winsorized_squared_loss(
    residuals: np.ndarray,
    sigma_clip: float = 3.0,
) -> float:
    """Outlier-resistant Winsorized sum-of-squared-residuals loss."""
    n = residuals.size
    if n == 0:
        return 0.0
    abs_r = np.abs(residuals)
    med_abs = float(np.median(abs_r))
    mad = 1.4826 * float(np.median(np.abs(residuals - float(np.median(residuals)))))
    scale = mad if mad > 0.0 else 1.4826 * med_abs
    thresh = med_abs + sigma_clip * scale
    cap = thresh * thresh
    total = 0.0
    for i in range(n):
        r = float(abs_r[i])
        if r > thresh:
            total += cap
        else:
            total += r * r
    return total


def fit_keplerian_periods_winsorized(
    times: npt.ArrayLike,
    freqs_obs: npt.ArrayLike,
    initial_guess: Sequence[float],
    *,
    fit_mask: npt.NDArray[np.bool_] | None = None,
    sigma_clip: float = 3.0,
    n_restarts: int = 5,
    seed: int | None = None,
) -> KeplerianFitResult:
    r"""Fit a Doppler frequency curve with outlier-resistant Winsorized loss.

    Models :math:`f(t) = f_0 (1 - v_{\mathrm{Kepler}}(t) / c)` using exact
    analytical Keplerian velocities and multi-start Nelder-Mead optimization,
    remaining robust to corrupted eclipse or RFI segments.

    Parameters
    ----------
    times : array_like
        Segment midpoints in seconds.
    freqs_obs : array_like
        Observed segment spin frequencies in Hz.
    initial_guess : sequence of float
        6-element seed ``[f0, p_orb, x_orb, ecc, aop, phi]``.
    fit_mask : NDArray[np.bool_] | None, optional
        Optional boolean mask selecting valid segments.
    sigma_clip : float, optional
        Winsorization outlier threshold in residual stds, by default 3.0.
    n_restarts : int, optional
        Number of jittered Nelder-Mead restarts in addition to the unperturbed
        seed, by default 5.
    seed : int | None, optional
        Random seed for restart perturbations.
    """
    t_arr = np.asarray(times, dtype=np.float64)
    f_arr = np.asarray(freqs_obs, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.shape != f_arr.shape:
        msg = "times and freqs_obs must be 1D arrays of matching shape"
        raise ValueError(msg)
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=np.bool_)
        t_arr = t_arr[mask]
        f_arr = f_arr[mask]
    x0 = np.asarray(initial_guess, dtype=np.float64)
    if x0.shape != (6,):
        msg = "initial_guess must have 6 elements: [f0, p_orb, x_orb, ecc, aop, phi]"
        raise ValueError(msg)

    f_scale = max(float(np.mean(np.abs(f_arr))), 1e-12)
    f_span = max(
        float(np.percentile(f_arr, 90) - np.percentile(f_arr, 10)), 1e-6 * f_scale
    )
    param_scales = np.array(
        [
            f_span,
            max(abs(float(x0[1])), 1.0),
            max(abs(float(x0[2])), 1e-3),
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )

    def _unpack(u: np.ndarray) -> np.ndarray:
        return x0 + u * param_scales

    def _model_freqs(p: np.ndarray) -> np.ndarray:
        f0, p_orb, x_orb, ecc, aop, phi = p
        _, v_arr, _ = kepler.keplerian_z_derivatives_exact(
            t_arr,
            p_orb,
            ecc,
            phi,
            x_orb * C_VAL,
            aop,
            0.5 * math.pi,
        )
        return f0 * (1.0 - v_arr / C_VAL)

    def _objective(u: np.ndarray) -> float:
        p = _unpack(u)
        f0, p_orb, x_orb, ecc, _aop, _phi = p
        if f0 <= 0.0 or p_orb <= 0.0 or x_orb < 0.0 or not (0.0 <= ecc < 0.99):
            return 1e30
        res = (_model_freqs(p) - f_arr) / f_span
        return winsorized_squared_loss(res, sigma_clip)

    rng = np.random.default_rng(seed)
    starts = [np.zeros(6, dtype=np.float64)]
    for _ in range(max(0, n_restarts)):
        u0 = np.array(
            [
                rng.normal(0.0, 0.05),
                rng.normal(0.0, 0.02),
                rng.normal(0.0, 0.05),
                rng.normal(0.0, 0.03),
                rng.normal(0.0, 0.2),
                rng.normal(0.0, 0.2),
            ],
            dtype=np.float64,
        )
        starts.append(u0)

    best_res: optimize.OptimizeResult | None = None
    for s in starts:
        opt = optimize.minimize(
            _objective,
            s,
            method="Nelder-Mead",
            options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-14},
        )
        if best_res is None or float(opt.fun) < float(best_res.fun):
            best_res = opt

    assert best_res is not None  # noqa: S101
    p_best = _unpack(np.asarray(best_res.x, dtype=np.float64))
    p_best[4] %= 2.0 * math.pi
    p_best[5] %= 2.0 * math.pi
    residuals = _model_freqs(p_best) - f_arr
    rms = math.sqrt(float(np.mean(residuals * residuals)))

    return KeplerianFitResult(
        f0=float(p_best[0]),
        p_orb=float(p_best[1]),
        x_orb=float(p_best[2]),
        ecc=float(p_best[3]),
        aop=float(p_best[4]),
        phi=float(p_best[5]),
        loss=float(best_res.fun),
        rms_residual=rms,
        residuals=residuals,
        success=bool(best_res.success),
    )


def fit_keplerian_multistart(
    times: npt.ArrayLike,
    delays: npt.ArrayLike,
    *,
    fourier_seed: FourierOrbitFit | None = None,
    initial_guess: Sequence[float] | None = None,
    fit_mask: npt.NDArray[np.bool_] | None = None,
    n_restarts: int = 5,
    seed: int | None = None,
) -> KeplerianFitResult:
    r"""Multi-start exact Keplerian phase-delay polisher.

    Fits :math:`\tau(t) = c_0 + c_1 (t - \bar{t}) + z_{\mathrm{Kepler}}(t)/c`
    by profiling out the linear nuisance parameters ``(c_0, c_1)`` analytically
    at each step and optimizing the 5 nonlinear orbital parameters
    ``[p_orb, x_orb, ecc, aop, phi]`` via multi-start Nelder-Mead.
    """
    t_arr = np.asarray(times, dtype=np.float64)
    y_arr = np.asarray(delays, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.shape != y_arr.shape:
        msg = "times and delays must be 1D arrays of matching shape"
        raise ValueError(msg)
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=np.bool_)
        t_arr = t_arr[mask]
        y_arr = y_arr[mask]

    if initial_guess is not None:
        x0 = np.asarray(initial_guess, dtype=np.float64)
        if x0.shape != (5,):
            msg = "initial_guess must have 5 elements: [p_orb, x_orb, ecc, aop, phi]"
            raise ValueError(msg)
    elif fourier_seed is not None:
        x0 = np.array(
            [
                fourier_seed.p_orb,
                max(fourier_seed.x_orb, 1e-6),
                fourier_seed.ecc_approx,
                0.0,
                fourier_seed.phi_orb,
            ],
            dtype=np.float64,
        )
    else:
        msg = "Either fourier_seed or initial_guess must be provided"
        raise ValueError(msg)

    t_c = t_arr - float(np.mean(t_arr))
    t_scale = max(float(np.max(np.abs(t_c))), 1.0)
    lin_design = np.column_stack([np.ones_like(t_c), t_c / t_scale])

    def _residuals(p: np.ndarray) -> tuple[np.ndarray, float]:
        p_orb, x_orb, ecc, aop, phi = p
        z_sec = kepler.keplerian_z(
            t_arr,
            p_orb,
            ecc,
            phi,
            x_orb,
            aop,
            0.5 * math.pi,
        )
        diff = y_arr - z_sec
        lin_c, _, _, _ = np.linalg.lstsq(lin_design, diff, rcond=None)
        res = diff - lin_design @ lin_c
        return res, float(lin_c[1] / t_scale)

    def _objective(p: np.ndarray) -> float:
        p_orb, x_orb, ecc, _aop, _phi = p
        if p_orb <= 0.0 or x_orb < 0.0 or not (0.0 <= ecc < 0.99):
            return 1e30
        res, _ = _residuals(p)
        return float(np.mean(res * res))

    rng = np.random.default_rng(seed)
    starts = [x0.copy()]
    for _ in range(max(0, n_restarts)):
        cand = x0.copy()
        cand[0] *= 1.0 + rng.normal(0.0, 0.02)
        cand[1] = max(1e-6, cand[1] * (1.0 + rng.normal(0.0, 0.05)))
        cand[2] = float(np.clip(cand[2] + rng.normal(0.0, 0.05), 0.0, 0.95))
        cand[3] = float(rng.uniform(0.0, 2.0 * math.pi))
        cand[4] = float((cand[4] + rng.normal(0.0, 0.25)) % (2.0 * math.pi))
        starts.append(cand)

    best_res: optimize.OptimizeResult | None = None
    for s in starts:
        opt = optimize.minimize(
            _objective,
            s,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-14},
        )
        if best_res is None or float(opt.fun) < float(best_res.fun):
            best_res = opt

    assert best_res is not None  # noqa: S101
    p_best = np.asarray(best_res.x, dtype=np.float64)
    p_best[3] %= 2.0 * math.pi
    p_best[4] %= 2.0 * math.pi
    residuals, slope = _residuals(p_best)
    rms = math.sqrt(float(np.mean(residuals * residuals)))

    return KeplerianFitResult(
        f0=-slope,
        p_orb=float(p_best[0]),
        x_orb=float(p_best[1]),
        ecc=float(p_best[2]),
        aop=float(p_best[3]),
        phi=float(p_best[4]),
        loss=float(best_res.fun),
        rms_residual=rms,
        residuals=residuals,
        success=bool(best_res.success),
    )
