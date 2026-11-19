r"""Heavy-tail (Pareto / lognormal) Neyman-Pearson LLR sensitivity bounds.

Computes exact convolved :math:`H_1` densities
:math:`p_{H_1}(x) = \int_0^\infty \mathcal{N}(x - s; 0, \sigma_n^2) p_S(s)\,ds`,
splined log-likelihood ratios :math:`\ell(x) = \ln(p_{H_1}(x) / p_{H_0}(x))`,
and expected Kullback-Leibler divergence :math:`D_{\mathrm{KL}}(p_{H_1} \| p_{H_0})`
versus the linear Gaussian baseline :math:`\frac{1}{2}(\mu_S / \sigma_n)^2`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import integrate, interpolate, special, stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt


def pareto_a0(mean_s: float, beta: float) -> float:
    r"""Return Pareto lower cutoff :math:`a_0 = \frac{\beta - 2}{\beta - 1} \mu_S`."""
    if mean_s <= 0.0:
        msg = f"mean_s must be positive, got {mean_s}"
        raise ValueError(msg)
    if beta <= 2.0:
        msg = f"Pareto exponent beta must be > 2 for finite mean, got {beta}"
        raise ValueError(msg)
    return ((beta - 2.0) / (beta - 1.0)) * mean_s


def pareto_signal_pdf(
    s: npt.ArrayLike,
    mean_s: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    r"""Evaluate Pareto PDF :math:`(\beta - 1) a_0^{\beta - 1} s^{-\beta}`."""
    a0 = pareto_a0(mean_s, beta)
    norm_c = (beta - 1.0) * (a0 ** (beta - 1.0))
    arr = np.asarray(s, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    mask = arr >= a0
    out[mask] = norm_c * (arr[mask] ** (-beta))
    return out


def lognormal_signal_pdf(
    s: npt.ArrayLike,
    mean_s: float,
    sigma_ln: float,
) -> npt.NDArray[np.float64]:
    r"""Lognormal signal PDF with exact mean ``mean_s`` and log-std ``sigma_ln``."""
    if mean_s <= 0.0 or sigma_ln <= 0.0:
        msg = f"mean_s and sigma_ln must be positive, got {mean_s}, {sigma_ln}"
        raise ValueError(msg)
    mu_ln = math.log(mean_s) - 0.5 * sigma_ln * sigma_ln
    arr = np.asarray(s, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    mask = arr > 0.0
    if np.any(mask):
        z = (np.log(arr[mask]) - mu_ln) / sigma_ln
        out[mask] = np.exp(-0.5 * z * z) / (
            arr[mask] * sigma_ln * math.sqrt(2.0 * math.pi)
        )
    return out


def convolve_gaussian_signal(
    x_grid: npt.ArrayLike,
    signal_pdf: Callable[[np.ndarray], np.ndarray],
    *,
    sigma_n: float = 1.0,
    s_min: float = 0.0,
    s_max: float = 200.0,
    n_quad: int = 2048,
) -> npt.NDArray[np.float64]:
    r"""Convolve a non-negative signal PDF with :math:`\mathcal{N}(0, \sigma_n^2)`.

    Uses a dense log-spaced quadrature grid over ``[s_min, s_max]`` so heavy
    tails are accurately integrated without truncation artifacts.
    """
    if sigma_n <= 0.0:
        msg = f"sigma_n must be positive, got {sigma_n}"
        raise ValueError(msg)
    x = np.asarray(x_grid, dtype=np.float64)
    lo = max(s_min, 1e-12)
    s_nodes = np.geomspace(lo, max(s_max, lo * 10.0), n_quad, dtype=np.float64)
    ps = np.asarray(signal_pdf(s_nodes), dtype=np.float64)
    # Trapezoidal weights in s
    ds = np.empty_like(s_nodes)
    ds[0] = 0.5 * (s_nodes[1] - s_nodes[0])
    ds[-1] = 0.5 * (s_nodes[-1] - s_nodes[-2])
    ds[1:-1] = 0.5 * (s_nodes[2:] - s_nodes[:-2])
    weights = ps * ds
    inv_norm = 1.0 / (sigma_n * math.sqrt(2.0 * math.pi))

    # Chunked matrix evaluation keeps peak memory modest
    out = np.empty_like(x, dtype=np.float64)
    chunk = 256
    for i0 in range(0, x.size, chunk):
        i1 = min(x.size, i0 + chunk)
        diff = (x[i0:i1, None] - s_nodes[None, :]) / sigma_n
        gauss = inv_norm * np.exp(-0.5 * diff * diff)
        out[i0:i1] = gauss @ weights
    return out


def _log_convolve_gaussian_signal(
    x_grid: npt.ArrayLike,
    signal_pdf: Callable[[np.ndarray], np.ndarray],
    *,
    sigma_n: float = 1.0,
    s_min: float = 0.0,
    s_max: float = 200.0,
    n_quad: int = 2048,
) -> npt.NDArray[np.float64]:
    r"""Evaluate :math:`\ln p_{H_1}(x)` via log-sum-exp quadrature without underflow."""
    if sigma_n <= 0.0:
        msg = f"sigma_n must be positive, got {sigma_n}"
        raise ValueError(msg)
    x = np.asarray(x_grid, dtype=np.float64)
    lo = max(s_min, 1e-12)
    s_nodes = np.geomspace(lo, max(s_max, lo * 10.0), n_quad, dtype=np.float64)
    ps = np.asarray(signal_pdf(s_nodes), dtype=np.float64)
    ds = np.empty_like(s_nodes)
    ds[0] = 0.5 * (s_nodes[1] - s_nodes[0])
    ds[-1] = 0.5 * (s_nodes[-1] - s_nodes[-2])
    ds[1:-1] = 0.5 * (s_nodes[2:] - s_nodes[:-2])
    weights = ps * ds
    pos = weights > 0.0
    s_nodes = s_nodes[pos]
    log_weights = np.log(weights[pos])
    log_norm = -math.log(sigma_n * math.sqrt(2.0 * math.pi))

    out = np.empty_like(x, dtype=np.float64)
    chunk = 256
    for i0 in range(0, x.size, chunk):
        i1 = min(x.size, i0 + chunk)
        diff = (x[i0:i1, None] - s_nodes[None, :]) / sigma_n
        log_terms = log_norm - 0.5 * diff * diff + log_weights[None, :]
        out[i0:i1] = special.logsumexp(log_terms, axis=1)
    return out


class HeavyTailLikelihood:
    """Tabulated Neyman-Pearson LLR model for Pareto or lognormal giant pulses."""

    def __init__(
        self,
        *,
        distribution: Literal["pareto", "lognormal"],
        mean_s: float,
        shape_param: float,
        sigma_n: float = 1.0,
        x_range: tuple[float, float] = (-8.0, 40.0),
        n_grid: int = 1024,
    ) -> None:
        self.distribution = distribution
        self.mean_s = float(mean_s)
        self.shape_param = float(shape_param)
        self.sigma_n = float(sigma_n)
        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])

        if distribution == "pareto":
            a0 = pareto_a0(self.mean_s, self.shape_param)
            s_min = a0
            # Choose upper cutoff capturing > 1 - 1e-7 of Pareto mass
            s_max = max(
                self.x_max + 10.0 * self.sigma_n,
                a0 * (1e7 ** (1.0 / (self.shape_param - 1.0))),
            )
            s_max = min(s_max, 1e6)

            def _pareto_pdf(s: np.ndarray) -> np.ndarray:
                return pareto_signal_pdf(s, self.mean_s, self.shape_param)

            pdf_fn: Callable[[np.ndarray], np.ndarray] = _pareto_pdf
        elif distribution == "lognormal":
            s_min = 1e-9
            s_max = max(
                self.x_max + 10.0 * self.sigma_n,
                self.mean_s * math.exp(6.0 * self.shape_param),
            )
            s_max = min(s_max, 1e6)

            def _lognormal_pdf(s: np.ndarray) -> np.ndarray:
                return lognormal_signal_pdf(s, self.mean_s, self.shape_param)

            pdf_fn = _lognormal_pdf
        else:
            msg = f"Unsupported distribution: {distribution!r}"
            raise ValueError(msg)

        self.x_grid = np.linspace(self.x_min, self.x_max, n_grid, dtype=np.float64)
        self.log_p_h1_grid = _log_convolve_gaussian_signal(
            self.x_grid,
            pdf_fn,
            sigma_n=self.sigma_n,
            s_min=s_min,
            s_max=s_max,
        )
        self.log_p_h0_grid = np.asarray(
            stats.norm.logpdf(self.x_grid, loc=0.0, scale=self.sigma_n),
            dtype=np.float64,
        )
        self.p_h1_grid = np.exp(self.log_p_h1_grid)
        self.p_h0_grid = np.exp(self.log_p_h0_grid)
        self.llr_grid = self.log_p_h1_grid - self.log_p_h0_grid
        self._llr_spline = interpolate.CubicSpline(
            self.x_grid,
            self.llr_grid,
            bc_type="not-a-knot",
            extrapolate=True,
        )
        self._h1_spline = interpolate.CubicSpline(
            self.x_grid,
            self.p_h1_grid,
            bc_type="natural",
            extrapolate=False,
        )

    def pdf_h1(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        vals = np.asarray(self._h1_spline(arr), dtype=np.float64)
        return np.where(np.isfinite(vals), np.maximum(vals, 0.0), 0.0)

    def llr(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64)
        return np.asarray(self._llr_spline(arr), dtype=np.float64)

    def score_samples(self, x: npt.ArrayLike) -> float:
        """Sum per-sample Neyman-Pearson log-likelihood ratios."""
        return float(np.sum(self.llr(x)))

    def kl_divergence(self) -> float:
        r"""Compute :math:`D_{\mathrm{KL}}(p_{H_1} \| p_{H_0})` over the grid."""
        integrand = self.p_h1_grid * self.llr_grid
        return float(integrate.trapezoid(integrand, self.x_grid))

    def linear_kl_baseline(self) -> float:
        r"""Return linear Gaussian baseline :math:`\frac{1}{2}(\mu_S/\sigma_n)^2`."""
        snr = self.mean_s / self.sigma_n
        return 0.5 * snr * snr
