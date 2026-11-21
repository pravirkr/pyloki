"""Analytical and semi-analytical sensitivity comparator: FFA/EP vs FFT harmonics.

Compares coherent FFA/EP matched-filter sensitivity against incoherent FFT
power harmonic summing (exact non-central chi-square distribution with
``df = 2 * n_harmonics``), folded-profile :math:`Z^2_m`, and the :math:`H`-test.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import attrs
import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    from numpy import typing as npt


@attrs.frozen(auto_attribs=True, kw_only=True)
class SensitivityComparisonResult:
    """Analytical minimum detectable S/N and flux ratio for FFA/EP vs FFT."""

    duty_cycle: float
    n_harmonics: int
    fap: float
    target_pdet: float
    snr_min_ffa: float
    snr_min_fft: float
    flux_ratio_ffa_over_fft: float
    fft_gain_factor: float
    harmonic_power_fraction: float


def gaussian_profile(
    duty_cycle: float,
    n_grid: int = 4096,
) -> npt.NDArray[np.float64]:
    """Build a zero-mean, unit-energy periodic Gaussian pulse on ``[0, 1)``.

    ``duty_cycle`` is the FWHM as a fraction of the pulse period.
    """
    if not (0.0 < duty_cycle < 1.0):
        msg = f"duty_cycle must be in (0, 1), got {duty_cycle}"
        raise ValueError(msg)
    phi = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float64)
    sigma = duty_cycle / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    # Wrap circular distance to [-0.5, 0.5]
    dphi = ((phi - 0.5 + 0.5) % 1.0) - 0.5
    raw = np.exp(-0.5 * (dphi / sigma) ** 2)
    centered = raw - float(np.mean(raw))
    energy = math.sqrt(float(np.mean(centered * centered)))
    return centered / energy


def von_mises_profile(
    duty_cycle: float,
    n_grid: int = 4096,
) -> npt.NDArray[np.float64]:
    """Build a zero-mean, unit-energy periodic von Mises pulse on ``[0, 1)``."""
    if not (0.0 < duty_cycle < 1.0):
        msg = f"duty_cycle must be in (0, 1), got {duty_cycle}"
        raise ValueError(msg)
    phi = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float64)
    # Match FWHM: exp(kappa * (cos(pi * duty) - 1)) = 0.5
    denom = 1.0 - math.cos(math.pi * duty_cycle)
    kappa = math.log(2.0) / max(denom, 1e-12)
    raw = np.exp(kappa * (np.cos(2.0 * math.pi * (phi - 0.5)) - 1.0))
    centered = raw - float(np.mean(raw))
    energy = math.sqrt(float(np.mean(centered * centered)))
    return centered / energy


def profile_fourier_powers(
    profile: npt.ArrayLike,
    n_harmonics: int,
) -> npt.NDArray[np.float64]:
    r"""Return fractional power :math:`2 |c_k|^2` in harmonics ``k = 1..n_harmonics``.

    For a zero-mean unit-energy profile, :math:`\sum_{k=1}^{\infty} 2|c_k|^2 = 1`.
    """
    if n_harmonics < 1:
        msg = f"n_harmonics must be >= 1, got {n_harmonics}"
        raise ValueError(msg)
    arr = np.asarray(profile, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    norm = math.sqrt(float(np.mean(arr * arr)))
    if norm <= 0.0:
        return np.zeros(n_harmonics, dtype=np.float64)
    arr = arr / norm
    fft_c = np.fft.rfft(arr) / float(arr.size)
    max_avail = min(n_harmonics, (arr.size - 1) // 2)
    powers = np.zeros(n_harmonics, dtype=np.float64)
    if max_avail > 0:
        powers[:max_avail] = 2.0 * (np.abs(fft_c[1 : max_avail + 1]) ** 2)
    return powers


def ffa_threshold(fap: float) -> float:
    """Return the Gaussian matched-filter threshold for false-alarm probability."""
    if not (0.0 < fap < 1.0):
        msg = f"fap must be in (0, 1), got {fap}"
        raise ValueError(msg)
    return float(stats.norm.isf(fap))


def ffa_detection_prob(
    snr_opt: float,
    fap: float,
    *,
    eta_bin: float = 1.0,
    rho_ep: float = 1.0,
) -> float:
    """Compute coherent FFA / EP detection probability at ideal S/N ``snr_opt``."""
    thresh = ffa_threshold(fap)
    eff_snr = snr_opt * eta_bin * rho_ep
    return float(stats.norm.sf(thresh - eff_snr))


def fft_harmonic_threshold(n_harmonics: int, fap: float) -> float:
    r"""Central :math:`\chi^2_{2 N_h}` threshold for false-alarm probability ``fap``."""
    if n_harmonics < 1:
        msg = f"n_harmonics must be >= 1, got {n_harmonics}"
        raise ValueError(msg)
    if not (0.0 < fap < 1.0):
        msg = f"fap must be in (0, 1), got {fap}"
        raise ValueError(msg)
    return float(stats.chi2.isf(fap, df=2 * n_harmonics))


def fft_noncentrality(
    snr_opt: float,
    harmonic_powers: npt.ArrayLike,
    *,
    scalloping: float | npt.ArrayLike = 0.773,
    accel_smear: npt.ArrayLike | None = None,
) -> float:
    r"""Non-centrality :math:`\lambda_{N_h}` of incoherent FFT harmonic summing."""
    hp = np.asarray(harmonic_powers, dtype=np.float64)
    sc = np.asarray(scalloping, dtype=np.float64)
    weights = hp * sc
    if accel_smear is not None:
        weights = weights * np.asarray(accel_smear, dtype=np.float64)
    return float((snr_opt * snr_opt) * np.sum(weights))


def fft_detection_prob(
    snr_opt: float,
    harmonic_powers: npt.ArrayLike,
    fap: float,
    *,
    scalloping: float | npt.ArrayLike = 0.773,
    accel_smear: npt.ArrayLike | None = None,
) -> float:
    r"""Compute FFT harmonic-summing detection probability under :math:`H_1`."""
    hp = np.asarray(harmonic_powers, dtype=np.float64)
    n_harm = int(hp.size)
    thresh = fft_harmonic_threshold(n_harm, fap)
    nc = fft_noncentrality(
        snr_opt,
        hp,
        scalloping=scalloping,
        accel_smear=accel_smear,
    )
    if nc <= 0.0:
        return float(stats.chi2.sf(thresh, df=2 * n_harm))
    return float(stats.ncx2.sf(thresh, df=2 * n_harm, nc=nc))


def z2m_statistic(
    profile: npt.ArrayLike,
    m: int,
    *,
    noise_var: float = 1.0,
) -> float:
    r"""Folded-profile :math:`Z^2_m` harmonic-power statistic (zero scalloping loss).

    For unit-variance Gaussian noise per bin (``noise_var = 1.0``),
    :math:`Z^2_m \sim \chi^2_{2m}` under :math:`H_0`.
    """
    if m < 1:
        msg = f"m must be >= 1, got {m}"
        raise ValueError(msg)
    arr = np.asarray(profile, dtype=np.float64)
    n = arr.size
    if n < 2:
        return 0.0
    fft_c = np.fft.rfft(arr)
    max_k = min(m, (n - 1) // 2)
    if max_k < 1:
        return 0.0
    # Normalization: E[|X_k|^2] = (n / 2) * noise_var for 1 <= k <= (n - 1) // 2
    power = (2.0 / (float(n) * noise_var)) * np.sum(np.abs(fft_c[1 : max_k + 1]) ** 2)
    return float(power)


def h_test_statistic(
    profile: npt.ArrayLike,
    *,
    m_max: int = 20,
    noise_var: float = 1.0,
) -> tuple[float, int]:
    r"""Compute the De Jager :math:`H`-test statistic and optimal harmonic count.

    Evaluates :math:`H = \max_{1 \le m \le m_{\max}} (Z^2_m - 4m + 4)`.
    Returns ``(h_stat, best_m)``.
    """
    arr = np.asarray(profile, dtype=np.float64)
    n = arr.size
    if n < 2 or m_max < 1:
        return 0.0, 1
    fft_c = np.fft.rfft(arr)
    eff_m = min(m_max, (n - 1) // 2)
    if eff_m < 1:
        return 0.0, 1
    harm_p = (2.0 / (float(n) * noise_var)) * (np.abs(fft_c[1 : eff_m + 1]) ** 2)
    z2 = np.cumsum(harm_p)
    ms = np.arange(1, eff_m + 1, dtype=np.float64)
    h_vals = z2 - 4.0 * ms + 4.0
    best_idx = int(np.argmax(h_vals))
    return float(h_vals[best_idx]), best_idx + 1


def fft_scalloping_loss(
    delta_k: float | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""Exact FFT bin-offset scalloping loss :math:`\mathrm{sinc}^2(\pi \Delta_k)`.

    For fractional bin offset :math:`\Delta_k \in [-0.5, 0.5]`, ``np.sinc(x)``
    evaluates :math:`\sin(\pi x)/(\pi x)`. Averaged uniformly over
    :math:`[-0.5, 0.5]`, this equals :math:`\approx 0.7737`.
    """
    arr = np.asarray(delta_k, dtype=np.float64)
    return np.square(np.sinc(arr))


def eta_bin_loss(
    duty_cycle: float,
    nbins: int,
    *,
    profile_shape: Literal["gaussian", "von_mises"] = "gaussian",
) -> float:
    """Compute finite-binning S/N retention ``eta_bin(duty_cycle, nbins) <= 1``.

    Compares the L2 energy of the phase-averaged ``nbins`` profile against the
    continuous ``2048``-bin reference profile of the same shape and duty cycle.
    """
    if nbins < 2:
        msg = f"nbins must be >= 2, got {nbins}"
        raise ValueError(msg)
    n_hi = max(2048, nbins * 16)
    n_hi -= n_hi % nbins
    hi_prof = (
        gaussian_profile(duty_cycle, n_grid=n_hi)
        if profile_shape == "gaussian"
        else von_mises_profile(duty_cycle, n_grid=n_hi)
    )
    binned = hi_prof.reshape(nbins, n_hi // nbins).mean(axis=1)
    binned = binned - float(np.mean(binned))
    e_hi = float(np.mean(hi_prof * hi_prof))
    e_bin = float(np.mean(binned * binned))
    if e_hi <= 0.0:
        return 0.0
    return float(min(1.0, math.sqrt(max(e_bin, 0.0) / e_hi)))


def chi2_to_sigma_eq(
    stat: float,
    df: int,
    *,
    n_trials: int = 1,
) -> float:
    r"""Convert a :math:`\chi^2_{\mathrm{df}}` statistic to Gaussian sigma_eq."""
    if df < 1:
        msg = f"df must be >= 1, got {df}"
        raise ValueError(msg)
    if n_trials < 1:
        msg = f"n_trials must be >= 1, got {n_trials}"
        raise ValueError(msg)
    minus_logsf = float(-stats.chi2.logsf(float(stat), df=df)) - math.log(
        float(n_trials),
    )
    if minus_logsf <= 0.0:
        return 0.0
    return float(stats.norm.isf(math.exp(-minus_logsf)))


def compare_ffa_vs_fft_flux_ratio(
    duty_cycle: float,
    n_harmonics: int,
    *,
    fap: float = 1e-7,
    fap_fft: float | None = None,
    target_pdet: float = 0.5,
    eta_bin: float = 1.0,
    rho_ep: float = 1.0,
    scalloping: float = 0.773,
    profile_shape: Literal["gaussian", "von_mises"] = "gaussian",
) -> SensitivityComparisonResult:
    """Solve exact minimum detectable S/N and flux ratio ``S_min,FFA / S_min,FFT``.

    ``fap`` sets the FFA false-alarm probability; ``fap_fft`` defaults to
    ``fap`` when omitted, allowing distinct trial-adjusted FAPs for FFA vs FFT.
    """
    if not (0.0 < target_pdet < 1.0):
        msg = f"target_pdet must be in (0, 1), got {target_pdet}"
        raise ValueError(msg)
    eff_fap_fft = fap if fap_fft is None else float(fap_fft)
    prof = (
        gaussian_profile(duty_cycle)
        if profile_shape == "gaussian"
        else von_mises_profile(duty_cycle)
    )
    hp = profile_fourier_powers(prof, n_harmonics)
    power_frac = float(np.sum(hp))

    # FFA minimum ideal S/N
    z_fap = ffa_threshold(fap)
    z_det = float(stats.norm.isf(target_pdet))
    snr_min_ffa = max(0.0, (z_fap - z_det) / max(eta_bin * rho_ep, 1e-12))

    # FFT minimum ideal S/N via root finding on ncx2.sf(thresh, 2*N_h, nc) = target_pdet
    if target_pdet <= eff_fap_fft:
        snr_min_fft = 0.0
    else:
        lo, hi = 0.0, max(10.0, snr_min_ffa * 2.0)
        while (
            fft_detection_prob(hi, hp, eff_fap_fft, scalloping=scalloping) < target_pdet
        ):
            hi *= 2.0
            if hi > 1e6:
                break
        snr_min_fft = float(
            optimize.brentq(
                lambda s: (
                    fft_detection_prob(float(s), hp, eff_fap_fft, scalloping=scalloping)
                    - target_pdet
                ),
                lo,
                hi,
                xtol=1e-9,
            ),
        )
    ratio = snr_min_ffa / max(snr_min_fft, 1e-12)
    return SensitivityComparisonResult(
        duty_cycle=duty_cycle,
        n_harmonics=n_harmonics,
        fap=fap,
        target_pdet=target_pdet,
        snr_min_ffa=snr_min_ffa,
        snr_min_fft=snr_min_fft,
        flux_ratio_ffa_over_fft=ratio,
        fft_gain_factor=1.0 / max(ratio, 1e-12),
        harmonic_power_fraction=power_frac,
    )


def spectral_index_sensitivity_increase(
    band_ratio: float,
    alpha: float,
) -> float:
    r"""Return :math:`\mathrm{SNR}^2` gain of spectral-index weighting vs flat summing.

    For a power-law flux spectrum :math:`S(\nu) \propto \nu^\alpha` across a
    fractional bandwidth ratio :math:`A = \nu_{\max} / \nu_{\min} \ge 1`,
    evaluates:

    .. math::
        G_{\mathrm{SNR}^2}(A, \alpha) =
        \frac{(\alpha + 1)^2}{2\alpha + 1}
        \frac{(A - 1)(A^{2\alpha + 1} - 1)}{(A^{\alpha + 1} - 1)^2}

    using smooth Taylor limits at the removable singularities
    :math:`\alpha = -1`, :math:`\alpha = -0.5`, and :math:`A = 1`.
    """
    if band_ratio < 1.0:
        msg = f"band_ratio must be >= 1.0, got {band_ratio}"
        raise ValueError(msg)
    if abs(band_ratio - 1.0) < 1e-12:
        return 1.0
    log_a = math.log(band_ratio)

    def _expm1_over_x(x: float) -> float:
        xl = x * log_a
        if abs(xl) < 1e-7:
            return log_a * (1.0 + 0.5 * xl + (xl * xl) / 6.0)
        return math.expm1(xl) / x

    num = (band_ratio - 1.0) * _expm1_over_x(2.0 * alpha + 1.0)
    den = _expm1_over_x(alpha + 1.0)
    return float(num / (den * den))
