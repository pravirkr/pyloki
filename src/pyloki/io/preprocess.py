"""Reference timeseries preprocessing, metadata, and DM=0 RFI masking.

Provides robust dual-array ``(ts_e, ts_v)`` inverse-variance preprocessing,
legacy z-score preprocessing, observation metadata, and ``DM=0`` broadband
time-domain and Fourier birdie RFI mask generation and transfer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import attrs
import numpy as np
from scipy import ndimage, special

from pyloki.io import preprocess_kernels as pk

if TYPE_CHECKING:
    from numpy import typing as npt

# Robust preprocessing block/scale constants
_K_MIN_STAT_BLOCK = 32
_K_MIN_ROBUST_SAMPLES = 16
_K_MIN_FLAG_BLOCK = 8
_K_MIN_FLAG_BLOCKS = 8
_K_REL_VAR_FLOOR = 1e-6
_LN2 = math.log(2.0)


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class ObservationMetadata:
    """Metadata belonging to a single-DM timeseries.

    ``None`` values indicate unknown fields; ``dm = 0.0`` is a valid known value.
    """

    dm: float | None = None
    start_mjd: float | None = None
    barycentric: bool | None = None
    source: str = ""
    input_path: str = ""
    original_nsamps: int = 0

    def __attrs_post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate observation metadata invariants."""
        if self.dm is not None and (not math.isfinite(self.dm) or self.dm < 0.0):
            msg = f"observation DM must be finite and nonnegative, got {self.dm}"
            raise ValueError(msg)
        if self.start_mjd is not None and (
            not math.isfinite(self.start_mjd) or self.start_mjd <= 0.0
        ):
            msg = (
                f"observation start_mjd must be finite and positive, "
                f"got {self.start_mjd}"
            )
            raise ValueError(msg)
        if self.original_nsamps < 0:
            msg = (
                f"observation original_nsamps must be >= 0, got {self.original_nsamps}"
            )
            raise ValueError(msg)

    def shifted(self, start_sample: int, dt: float) -> ObservationMetadata:
        """Return a copy with ``start_mjd`` shifted by ``start_sample * dt``."""
        new_mjd = (
            None
            if self.start_mjd is None
            else self.start_mjd + (start_sample * dt) / 86400.0
        )
        return ObservationMetadata(
            dm=self.dm,
            start_mjd=new_mjd,
            barycentric=self.barycentric,
            source=self.source,
            input_path=self.input_path,
            original_nsamps=self.original_nsamps,
        )


@attrs.frozen(auto_attribs=True)
class Birdie:
    """Known periodic RFI line: zap Fourier bins with ``|f - freq| <= width / 2``."""

    freq: float
    width: float = 0.0


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class PreprocessOptions:
    """Configuration options for ``preprocess_timeseries``."""

    method: Literal["robust", "zscore"] = "robust"
    filter_window: float = 1.0
    gain_model: Literal["additive", "multiplicative"] = "additive"
    variance_window: float = 0.0
    window_blocks: int = 101
    n_iter: int = 2
    block_scales: tuple[float, ...] = (0.016, 0.065, 0.26, 1.05)
    block_sigma: float = 6.0
    min_good_fraction: float = 0.3
    clip_sigma: float = 6.0
    zap_periodic: bool = False
    zap_sigma: float = 8.0
    zap_whiten_bins: int = 1001
    birdies: tuple[Birdie, ...] = ()
    zap_mode: Literal["noise", "phase_kept", "zero"] = "noise"
    zap_seed: int | None = None
    loc_method: Literal["mean", "median"] = "mean"
    scale_method: Literal["iqr", "mad", "std"] = "iqr"

    def __attrs_post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate preprocessing option ranges."""
        if self.method not in ("robust", "zscore"):
            msg = f"preprocess: unsupported method {self.method!r}"
            raise ValueError(msg)
        if self.gain_model not in ("additive", "multiplicative"):
            msg = f"preprocess: unsupported gain_model {self.gain_model!r}"
            raise ValueError(msg)
        if self.zap_mode not in ("noise", "phase_kept", "zero"):
            msg = f"preprocess: unsupported zap_mode {self.zap_mode!r}"
            raise ValueError(msg)
        if not math.isfinite(self.filter_window) or self.filter_window < 0.0:
            msg = "preprocess: filter_window must be finite and >= 0"
            raise ValueError(msg)
        if not math.isfinite(self.variance_window) or self.variance_window < 0.0:
            msg = "preprocess: variance_window must be finite and >= 0"
            raise ValueError(msg)
        if self.window_blocks < 1:
            msg = "preprocess: window_blocks must be >= 1"
            raise ValueError(msg)
        if self.n_iter < 0:
            msg = "preprocess: n_iter must be >= 0"
            raise ValueError(msg)
        for scale in self.block_scales:
            if not math.isfinite(scale) or scale <= 0.0:
                msg = "preprocess: block_scales must be finite and > 0"
                raise ValueError(msg)
        if not math.isfinite(self.block_sigma) or self.block_sigma <= 0.0:
            msg = "preprocess: block_sigma must be > 0"
            raise ValueError(msg)
        if (
            not math.isfinite(self.min_good_fraction)
            or self.min_good_fraction < 0.0
            or self.min_good_fraction > 1.0
        ):
            msg = "preprocess: min_good_fraction must be in [0, 1]"
            raise ValueError(msg)
        if not math.isfinite(self.clip_sigma) or self.clip_sigma < 0.0:
            msg = "preprocess: clip_sigma must be >= 0 (0 disables)"
            raise ValueError(msg)
        if not math.isfinite(self.zap_sigma) or self.zap_sigma <= 0.0:
            msg = "preprocess: zap_sigma must be > 0"
            raise ValueError(msg)
        if self.zap_whiten_bins < 3:
            msg = "preprocess: zap_whiten_bins must be >= 3"
            raise ValueError(msg)
        for birdie in self.birdies:
            if (
                not math.isfinite(birdie.freq)
                or birdie.freq <= 0.0
                or not math.isfinite(birdie.width)
                or birdie.width < 0.0
            ):
                msg = "preprocess: birdies need freq > 0 and width >= 0"
                raise ValueError(msg)


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class PreprocessReport:
    """Diagnostic summary returned by ``preprocess_timeseries``."""

    method: Literal["robust", "zscore"]
    nsamps: int
    baseline_window: int = 0
    variance_window: int = 0
    block_size: int = 0
    n_masked: int = 0
    n_clipped: int = 0
    n_zapped: int = 0
    longest_masked_run: int = 0
    global_scale: float = 0.0
    norm: float = 1.0
    block_mu: npt.NDArray[np.float32] = attrs.Factory(
        lambda: np.empty(0, dtype=np.float32),
    )
    block_sigma: npt.NDArray[np.float32] = attrs.Factory(
        lambda: np.empty(0, dtype=np.float32),
    )
    block_good_fraction: npt.NDArray[np.float32] = attrs.Factory(
        lambda: np.empty(0, dtype=np.float32),
    )

    @property
    def zero_weight_fraction(self) -> float:
        """Fraction of samples with zero weight."""
        if self.nsamps == 0:
            return 0.0
        return (self.n_masked + self.n_clipped) / self.nsamps


@attrs.frozen(auto_attribs=True)
class DM0RfiMask:
    """Broadband ``DM=0`` time-domain and Fourier birdie RFI mask."""

    mask_t: npt.NDArray[np.int64]
    mask_f: npt.NDArray[np.int64]
    nsamps: int
    dt: float


def _window_in_samples(window_sec: float, tsamp: float, n: int) -> int:
    bins = window_sec / tsamp
    if bins >= n:
        return n
    return max(1, round(bins)) if bins >= 1.0 else 1


def _blocks_per_window(window: int, block: int) -> int:
    k = max(1, round(window / block))
    return k + 1 if k % 2 == 0 else k


def _compute_block_stats(
    x: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    block: int,
    k_mu: int,
    k_var: int,
    min_good: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    n = x.size
    all_valid = bool(np.all(mask))
    med, usable, good = pk.compute_block_medians(x, mask, block, min_good, all_valid)
    mu, ok_mu = pk.masked_running_median(med, usable, k_mu)
    if not ok_mu:
        msg = "preprocess: no usable block statistics"
        raise ValueError(msg)
    mu = pk.reflected_running_mean(mu, _blocks_per_window(k_mu, 2))
    mu_interp = pk.interpolate_blocks(n, block, mu)

    var, var_usable = pk.compute_block_variances(
        x,
        mask,
        mu_interp,
        block,
        min_good,
        all_valid,
    )
    if not np.any(var_usable):
        msg = (
            "preprocess: no block has a usable variance "
            "(constant or fully masked series)"
        )
        raise ValueError(msg)

    median_var = float(np.median(var[var_usable]))
    var_smooth, ok_var = pk.masked_running_mean(var, var_usable, k_var)
    if not ok_var:
        msg = "preprocess: no usable block statistics"
        raise ValueError(msg)
    np.maximum(var_smooth, _K_REL_VAR_FLOOR * median_var, out=var_smooth)
    return mu, var_smooth, good, median_var


def _flag_bad_blocks(
    z: npt.NDArray[np.float64],
    scales: list[int],
    nsigma: float,
    min_good: float,
    clip_sigma: float,
) -> npt.NDArray[np.bool_]:
    mask = np.ones(z.size, dtype=np.bool_)
    for scale in scales:
        pk.flag_bad_blocks_scale(z, mask, scale, nsigma, min_good, clip_sigma)
    return mask


def _exp_power_threshold(sigma: float) -> float:
    if sigma < 30.0:
        return float(-math.log(0.5 * special.erfc(sigma / math.sqrt(2.0))))
    return 0.5 * sigma * sigma + math.log(sigma * math.sqrt(2.0 * math.pi))


def _birdie_bins(
    birdies: tuple[Birdie, ...],
    t_obs: float,
    nf: int,
) -> npt.NDArray[np.bool_]:
    flagged = np.zeros(nf, dtype=np.bool_)
    for bd in birdies:
        lo = (bd.freq - 0.5 * bd.width) * t_obs
        hi = (bd.freq + 0.5 * bd.width) * t_obs
        k_lo = max(math.ceil(lo), 1)
        k_hi = max(math.floor(hi), 0)
        if k_lo > k_hi:
            k_lo = k_hi = max(round(bd.freq * t_obs), 1)
        k_hi = min(k_hi, nf - 1)
        if k_lo <= k_hi:
            flagged[k_lo : k_hi + 1] = True
    return flagged


def _zap_spectrum(
    x: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    mu_interp: npt.NDArray[np.float64],
    var_interp: npt.NDArray[np.float64],
    tsamp: float,
    options: PreprocessOptions,
) -> int:
    n = x.size
    sigma_interp = np.sqrt(var_interp)
    z = (x - mu_interp) / sigma_interp
    valid_spec = (
        mask & (np.abs(z) <= options.clip_sigma) if options.clip_sigma > 0.0 else mask
    )
    z_work = np.where(valid_spec, z, 0.0)

    spec = np.fft.rfft(z_work)
    nf = spec.size
    power = (spec.real * spec.real) + (spec.imag * spec.imag)
    flagged = _birdie_bins(options.birdies, n * tsamp, nf)

    if nf > 2:
        win = min(options.zap_whiten_bins, nf - 1)
        if win % 2 == 0:
            win = max(1, win - 1)
        med_p = ndimage.median_filter(power[1:], size=win, mode="reflect")
        local_mean = np.maximum(med_p / _LN2, 1e-30)
    else:
        local_mean = np.full(max(0, nf - 1), max(float(np.median(power)), 1e-30))

    if options.zap_periodic and nf > 1:
        thresh = _exp_power_threshold(options.zap_sigma)
        flagged[1:] |= (power[1:] / local_mean) > thresh
    flagged[0] = False

    n_zapped = int(np.count_nonzero(flagged))
    if n_zapped == 0:
        return 0

    zap_idx = np.flatnonzero(flagged)
    loc_p = local_mean[zap_idx - 1]
    if options.zap_mode == "zero":
        spec[zap_idx] = 0.0
    elif options.zap_mode == "phase_kept":
        cur_p = power[zap_idx]
        scale = np.where(cur_p > loc_p, np.sqrt(loc_p / np.maximum(cur_p, 1e-30)), 1.0)
        spec[zap_idx] *= scale
    else:
        rng = np.random.default_rng(options.zap_seed)
        amp = np.sqrt(0.5 * loc_p)
        spec[zap_idx] = amp * (
            rng.standard_normal(n_zapped) + 1j * rng.standard_normal(n_zapped)
        )
        if n % 2 == 0 and flagged[-1]:
            spec[-1] = math.sqrt(local_mean[-1]) * rng.standard_normal()

    z_clean = np.fft.irfft(spec, n=n)
    x[valid_spec] = (
        mu_interp[valid_spec] + z_clean[valid_spec] * sigma_interp[valid_spec]
    )
    return n_zapped


def _preprocess_zscore(
    raw: npt.NDArray[np.float64],
    tsamp: float,
    options: PreprocessOptions,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], PreprocessReport]:
    n = raw.size
    x = raw.copy()
    baseline_window = 0
    if options.filter_window > 0.0:
        baseline_window = _window_in_samples(options.filter_window, tsamp, n)
        if baseline_window > 1:
            win = baseline_window if baseline_window % 2 == 1 else baseline_window + 1
            x -= ndimage.median_filter(x, size=min(win, n), mode="reflect")
    loc = float(np.mean(x)) if options.loc_method == "mean" else float(np.median(x))
    x -= loc
    if options.scale_method == "iqr":
        q75, q25 = np.percentile(x, [75.0, 25.0])
        scale = float((q75 - q25) / 1.3489795003921634)
    elif options.scale_method == "mad":
        scale = pk.K_MAD_SCALE * float(np.median(np.abs(x - np.median(x))))
    else:
        scale = float(np.std(x))
    if not math.isfinite(scale) or scale <= 0.0:
        msg = "preprocess: series is constant (zero scale)"
        raise ValueError(msg)
    ts_e = (x / scale).astype(np.float32)
    ts_v = np.ones(n, dtype=np.float32)
    report = PreprocessReport(
        method="zscore",
        nsamps=n,
        baseline_window=baseline_window,
        global_scale=scale,
        norm=1.0,
    )
    return ts_e, ts_v, report


def preprocess_timeseries(
    raw: npt.ArrayLike,
    tsamp: float,
    options: PreprocessOptions | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], PreprocessReport]:
    """Build folding sufficient statistics ``(ts_e, ts_v)`` from a raw timeseries.

    On return, ``ts_e`` and ``ts_v`` are finite ``float32`` arrays with
    ``ts_v >= 0``, ``ts_e == 0`` wherever ``ts_v == 0``, and at least one
    sample with ``ts_v > 0``.
    """
    opts = PreprocessOptions() if options is None else options
    opts.validate()
    if not math.isfinite(tsamp) or tsamp <= 0.0:
        msg = f"preprocess: tsamp must be finite and positive, got {tsamp}"
        raise ValueError(msg)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        msg = "preprocess: input must be a non-empty 1D array"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "preprocess: input has non-finite values"
        raise ValueError(msg)

    if opts.method == "zscore":
        return _preprocess_zscore(arr, tsamp, opts)

    n = arr.size
    if n < _K_MIN_ROBUST_SAMPLES:
        msg = (
            f"preprocess: the robust method needs at least "
            f"{_K_MIN_ROBUST_SAMPLES} samples, got {n}"
        )
        raise ValueError(msg)

    x = arr.copy()
    w_mu = (
        _window_in_samples(opts.filter_window, tsamp, n)
        if opts.filter_window > 0.0
        else n
    )
    w_var = (
        _window_in_samples(opts.variance_window, tsamp, n)
        if opts.variance_window > 0.0
        else w_mu
    )
    block = max(min(_K_MIN_STAT_BLOCK, n), min(w_mu // opts.window_blocks, n))
    k_mu = _blocks_per_window(w_mu, block)
    k_var = _blocks_per_window(w_var, block)

    scales = sorted(
        {
            round(sc / tsamp)
            for sc in opts.block_scales
            if round(sc / tsamp) >= _K_MIN_FLAG_BLOCK
            and n // max(round(sc / tsamp), 1) >= _K_MIN_FLAG_BLOCKS
        },
    )

    mask = np.ones(n, dtype=np.bool_)
    mu, var, good, median_var = _compute_block_stats(
        x,
        mask,
        block,
        k_mu,
        k_var,
        opts.min_good_fraction,
    )
    if scales:
        for _ in range(opts.n_iter):
            mu_s = pk.interpolate_blocks(n, block, mu)
            var_s = pk.interpolate_blocks(n, block, var)
            z = (x - mu_s) / np.sqrt(var_s)
            mask = _flag_bad_blocks(
                z,
                scales,
                opts.block_sigma,
                opts.min_good_fraction,
                opts.clip_sigma,
            )
            mu, var, good, median_var = _compute_block_stats(
                x,
                mask,
                block,
                k_mu,
                k_var,
                opts.min_good_fraction,
            )

    n_zapped = 0
    if opts.zap_periodic or opts.birdies:
        mu_s = pk.interpolate_blocks(n, block, mu)
        var_s = pk.interpolate_blocks(n, block, var)
        n_zapped = _zap_spectrum(x, mask, mu_s, var_s, tsamp, opts)
        if n_zapped > 0:
            mu, var, good, median_var = _compute_block_stats(
                x,
                mask,
                block,
                k_mu,
                k_var,
                opts.min_good_fraction,
            )

    mult = opts.gain_model == "multiplicative"
    if mult and float(np.min(mu)) <= 0.0:
        msg = (
            "preprocess: the multiplicative gain model needs a positive baseline "
            "(is the data already mean-subtracted?)"
        )
        raise ValueError(msg)

    good_blocks = good > 0.0
    if not np.any(good_blocks):
        msg = "preprocess: every sample is masked"
        raise ValueError(msg)
    g_block = mu[good_blocks] if mult else np.ones(int(np.count_nonzero(good_blocks)))
    info = (g_block * g_block) / var[good_blocks]
    c = 1.0 / math.sqrt(float(np.median(info)))

    mu_s = pk.interpolate_blocks(n, block, mu)
    var_s = pk.interpolate_blocks(n, block, var)
    res = x - mu_s
    g_s = mu_s if mult else np.ones(n, dtype=np.float64)

    clipped = (
        mask & ((res * res) > (opts.clip_sigma * opts.clip_sigma) * var_s)
        if opts.clip_sigma > 0.0
        else np.zeros(n, dtype=np.bool_)
    )
    keep = mask & ~clipped
    n_masked = int(np.count_nonzero(~mask))
    n_clipped = int(np.count_nonzero(clipped))
    if not np.any(keep):
        msg = "preprocess: every sample is masked"
        raise ValueError(msg)

    ts_e = np.zeros(n, dtype=np.float32)
    ts_v = np.zeros(n, dtype=np.float32)
    ts_e[keep] = (c * res[keep] * g_s[keep] / var_s[keep]).astype(np.float32)
    ts_v[keep] = ((c * c) * (g_s[keep] * g_s[keep]) / var_s[keep]).astype(np.float32)

    report = PreprocessReport(
        method="robust",
        nsamps=n,
        baseline_window=w_mu,
        variance_window=w_var,
        block_size=block,
        n_masked=n_masked,
        n_clipped=n_clipped,
        n_zapped=n_zapped,
        longest_masked_run=int(pk.longest_zero_run(ts_v)),
        global_scale=math.sqrt(median_var),
        norm=c,
        block_mu=mu.astype(np.float32),
        block_sigma=np.sqrt(var).astype(np.float32),
        block_good_fraction=good.astype(np.float32),
    )
    return ts_e, ts_v, report


def generate_dm0_rfi_mask(
    dm0_e: npt.ArrayLike,
    dm0_v: npt.ArrayLike,
    dt: float,
    *,
    t_sigma: float = 6.0,
    f_cutoff_hz: float = 10.0,
    f_sigma: float = 10.0,
) -> DM0RfiMask:
    """Build a broadband ``DM=0`` time-domain and MAD-robust Fourier birdie mask."""
    if not math.isfinite(dt) or dt <= 0.0:
        msg = f"dt must be finite and positive, got {dt}"
        raise ValueError(msg)
    if t_sigma <= 0.0 or f_sigma <= 0.0 or f_cutoff_hz < 0.0:
        msg = "t_sigma and f_sigma must be > 0, and f_cutoff_hz must be >= 0"
        raise ValueError(msg)
    e = np.asarray(dm0_e, dtype=np.float64)
    v = np.asarray(dm0_v, dtype=np.float64)
    if e.ndim != 1 or e.shape != v.shape or e.size == 0:
        msg = "dm0_e and dm0_v must be non-empty 1D arrays of matching shape"
        raise ValueError(msg)

    bad_t = (v <= 0.0) | ((e * e) > (t_sigma * t_sigma) * v)
    mask_t = np.flatnonzero(bad_t).astype(np.int64)

    work = e.copy()
    work[mask_t] = 0.0
    spec = np.abs(np.fft.rfft(work))
    k0 = max(1, math.ceil(f_cutoff_hz * work.size * dt))
    if k0 < spec.size:
        tail = spec[k0:]
        med = float(np.median(tail))
        mad_std = pk.K_MAD_SCALE * float(np.median(np.abs(tail - med)))
        if mad_std > 0.0:
            mask_f = (np.flatnonzero(tail > med + f_sigma * mad_std) + k0).astype(
                np.int64,
            )
        else:
            mask_f = np.empty(0, dtype=np.int64)
    else:
        mask_f = np.empty(0, dtype=np.int64)

    return DM0RfiMask(mask_t=mask_t, mask_f=mask_f, nsamps=e.size, dt=dt)


def apply_dm0_rfi_mask(
    ts_e: npt.ArrayLike,
    ts_v: npt.ArrayLike,
    rfi_mask: DM0RfiMask,
    *,
    zap_mode: Literal["noise", "zero"] = "noise",
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Transfer a ``DM=0`` RFI mask onto a target-DM ``(ts_e, ts_v)`` timeseries.

    Strictly enforces the folding validity contract: masked samples receive
    ``ts_e[mask_t] = 0.0`` and ``ts_v[mask_t] = 0.0``.
    """
    if zap_mode not in ("noise", "zero"):
        msg = f"Unsupported zap_mode: {zap_mode!r}"
        raise ValueError(msg)
    e = np.array(ts_e, dtype=np.float64, copy=True)
    v = np.array(ts_v, dtype=np.float64, copy=True)
    if e.ndim != 1 or e.shape != v.shape:
        msg = "ts_e and ts_v must be 1D arrays of matching shape"
        raise ValueError(msg)
    if e.size != rfi_mask.nsamps:
        msg = (
            f"Series length ({e.size}) does not match DM=0 mask length "
            f"({rfi_mask.nsamps})"
        )
        raise ValueError(msg)

    if rfi_mask.mask_t.size > 0:
        e[rfi_mask.mask_t] = 0.0
    if rfi_mask.mask_f.size > 0:
        spec = np.fft.rfft(e)
        if zap_mode == "zero":
            spec[rfi_mask.mask_f] = 0.0
        else:
            high_start = max(1, spec.size // 2)
            high_power = np.abs(spec[high_start:]) ** 2
            sigma_f = math.sqrt(max(float(np.median(high_power)) / _LN2, 0.0) * 0.5)
            rng = np.random.default_rng(seed)
            n_f = rfi_mask.mask_f.size
            spec[rfi_mask.mask_f] = sigma_f * (
                rng.standard_normal(n_f) + 1j * rng.standard_normal(n_f)
            )
            if e.size % 2 == 0 and np.any(rfi_mask.mask_f == spec.size - 1):
                spec[-1] = math.sqrt(2.0) * sigma_f * rng.standard_normal()
        e = np.fft.irfft(spec, n=e.size)

    if rfi_mask.mask_t.size > 0:
        e[rfi_mask.mask_t] = 0.0
        v[rfi_mask.mask_t] = 0.0

    return e.astype(np.float32), v.astype(np.float32)
