from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import attrs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sigpyproc.core import filters as sig_filters
from sigpyproc.readers import FilReader
from sigpyproc.timeseries import TimeSeries as sigTimeSeries
from sigpyproc.viz.styles import PlotTable

from pyloki.core import brutefold
from pyloki.detection.scoring import boxcar_snr_1d
from pyloki.io.preprocess import (
    DM0RfiMask,
    ObservationMetadata,
    PreprocessOptions,
    apply_dm0_rfi_mask,
    preprocess_timeseries,
)
from pyloki.io.preprocess_kernels import fold_subbands_kernel
from pyloki.simulation.modulate import type_to_mods
from pyloki.utils import np_utils

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy import typing as npt

# Pulsar dispersion constant in MHz^2 pc^-1 cm^3 s
K_DM_CONST = 4.148808e3


class TimeSeries:
    def __init__(
        self,
        ts_e: np.ndarray,
        ts_v: np.ndarray,
        dt: float,
        *,
        metadata: ObservationMetadata | None = None,
    ) -> None:
        self._ts_e = np.asarray(ts_e, dtype=np.float32)
        self._ts_v = np.asarray(ts_v, dtype=np.float32)
        self._dt = dt
        self._metadata = metadata

    @property
    def ts_e(self) -> np.ndarray:
        return self._ts_e

    @property
    def ts_v(self) -> np.ndarray:
        return self._ts_v

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def metadata(self) -> ObservationMetadata | None:
        return self._metadata

    @property
    def nsamps(self) -> int:
        return len(self.ts_e)

    @property
    def tobs(self) -> float:
        return self.nsamps * self.dt

    def downsample(self, factor: int) -> TimeSeries:
        if factor < 1:
            msg = f"factor must be >= 1, got {factor}"
            raise ValueError(msg)
        n_keep = (self.nsamps // factor) * factor
        if n_keep == 0:
            msg = f"factor ({factor}) exceeds nsamps ({self.nsamps})"
            raise ValueError(msg)
        ts_e = np_utils.downsample_1d(self.ts_e[:n_keep], factor)
        ts_v = np_utils.downsample_1d(self.ts_v[:n_keep], factor) / factor
        return TimeSeries(ts_e, ts_v, self.dt * factor, metadata=self._metadata)

    def apply_dm0_rfi_mask(
        self,
        rfi_mask: DM0RfiMask,
        *,
        zap_mode: Literal["noise", "zero"] = "noise",
        seed: int | None = None,
    ) -> TimeSeries:
        """Return a new ``TimeSeries`` cleaned with a ``DM=0`` RFI mask."""
        ts_e, ts_v = apply_dm0_rfi_mask(
            self._ts_e,
            self._ts_v,
            rfi_mask,
            zap_mode=zap_mode,
            seed=seed,
        )
        return TimeSeries(ts_e, ts_v, self._dt, metadata=self._metadata)

    @classmethod
    def from_raw(
        cls,
        raw: npt.ArrayLike,
        dt: float,
        *,
        options: PreprocessOptions | None = None,
        metadata: ObservationMetadata | None = None,
    ) -> TimeSeries:
        """Build a ``TimeSeries`` from raw samples via ``preprocess_timeseries``."""
        ts_e, ts_v, _ = preprocess_timeseries(raw, dt, options=options)
        return cls(ts_e, ts_v, dt, metadata=metadata)

    def fold_ephem(
        self,
        freq: float,
        nbins: int,
        nsubints: int = 1,
        *,
        normalize: bool = True,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
        mod_tref: float | None = None,
    ) -> np.ndarray:
        cycles = int(self.tobs * freq)
        if cycles < 1:
            msg = (
                f"Period ({1 / freq}) is greater than the "
                f"total data length ({self.tobs})"
            )
            raise ValueError(msg)
        if nsubints < 1 or nsubints > cycles:
            msg = f"subints must be >= 1 and <= {cycles}"
            raise ValueError(msg)
        if mod_kwargs is None:
            mod_kwargs = {}
        if mod_tref is None:
            mod_tref = self.tobs / 2
        mod_func = type_to_mods[mod_type](**mod_kwargs)
        proper_time = mod_func.generate(np.arange(self.nsamps) * self.dt, mod_tref)
        fold = brutefold(
            self.ts_e,
            self.ts_v,
            proper_time,
            freq,
            nsubints,
            nbins,
        ).squeeze()
        if normalize:
            return np.divide(
                fold[..., 0, :],
                np.sqrt(fold[..., 1, :]),
                out=np.zeros_like(fold[..., 0, :], dtype=np.float32),
                where=~np.isclose(fold[..., 1, :], 0, atol=1e-5),
            )
        return fold

    def plot_fold(
        self,
        freq: float,
        nbins: int,
        nsubints: int = 64,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
        mod_tref: float | None = None,
        figsize: tuple[float, float] = (10, 6.5),
        dpi: int = 100,
        cmap: str = "magma_r",
    ) -> plt.Figure:
        ephem_fold = self.fold_ephem(
            freq,
            nbins,
            nsubints=1,
            mod_type=mod_type,
            mod_kwargs=mod_kwargs,
            mod_tref=mod_tref,
        )
        ephem_fold_subints = self.fold_ephem(
            freq,
            nbins,
            nsubints=nsubints,
            mod_type=mod_type,
            mod_kwargs=mod_kwargs,
            mod_tref=mod_tref,
        )
        sig_boxcar = sig_filters.MatchedFilter(
            ephem_fold,
            loc_method="norm",
            scale_method="norm",
            nbins_max=nbins // 2,
            spacing_factor=1,
        )
        match_boxcar = boxcar_snr_1d(ephem_fold, np.arange(1, nbins // 2), 1.0)
        if mod_kwargs is None:
            mod_kwargs = {"acc": 0, "jerk": 0, "snap": 0}

        figure = plt.figure(figsize=figsize, dpi=dpi)
        grid = figure.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=(1.5, 1),
            width_ratios=(1, 1.5),
        )
        grid.update(left=0.1, right=0.98, bottom=0.08, top=0.95, hspace=0.2)
        axtable = plt.subplot(grid[0, 0])
        axsubints = plt.subplot(grid[0, 1])
        axprofile = plt.subplot(grid[1, :])
        ducy = sig_boxcar.best_temp.width / len(ephem_fold)
        table = PlotTable(
            col_offsets={
                "name": 0.10,
                "value": 0.75,
                "unit": 0.85,
            },
            top_margin=0.1,
            line_height=0.12,
            font_size=12,
        )
        table.add_entry("Tsamp", f"{self.dt * 1e3:.3f}", unit="ms")
        table.add_entry("Period", 1 / freq, unit="s")
        table.add_entry("Accel", mod_kwargs.get("acc", 0))
        table.add_entry("Jerk", mod_kwargs.get("jerk", 0))
        table.add_entry("Snap", mod_kwargs.get("snap", 0))
        table.add_entry("Width (box)", sig_boxcar.best_temp.width)
        table.add_entry("Ducy (box)", f"{ducy:.3f}")
        table.plot(axtable)

        axsubints.imshow(
            ephem_fold_subints,
            aspect="auto",
            interpolation="none",
            extent=(0, nbins, 0, self.tobs),
            cmap=plt.get_cmap(cmap),
            origin="lower",
        )
        axsubints.set_ylabel("Time (seconds)")
        axsubints.set_xlabel("Phase bin")
        axprofile.plot(
            range(nbins),
            ephem_fold,
            color="#404040",
            label="Folded Profile",
        )
        axprofile.plot(
            range(nbins),
            sig_boxcar.best_model,
            color="#d62728",
            alpha=0.65,
            ls="--",
            label="Boxcar template",
        )
        axprofile.set_xlim(0, nbins)
        axprofile.set_xlabel("Phase bin")
        axprofile.set_ylabel("Normalised amplitude")

        textstr = (
            f"S/N (Boxcar) = {sig_boxcar.snr:.2f}\n"
            f"S/N (Boxcar, 1D) = {match_boxcar.max():.2f}\n"
        )

        at = AnchoredText(textstr, loc="upper left", prop={"size": 10}, frameon=True)
        at.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
        axprofile.add_artist(at)
        axprofile.legend(loc="upper right")
        return figure

    @classmethod
    def from_tim(
        cls,
        timfile: str,
        tim_type: str = "dat",
        filter_window: float = 1.0,
        norm_scale_method: str = "iqr",
        *,
        start_sample: int = 0,
        nsamps: int | None = None,
        preprocess_method: Literal["zscore", "robust"] = "zscore",
        preprocess_options: PreprocessOptions | None = None,
    ) -> TimeSeries:
        """Load a time series from a .tim or .dat file.

        Parameters
        ----------
        timfile : str
            Path to the .tim or .dat file.
        tim_type : str, optional
            Type of the time series file, by default "dat"
        filter_window : float, optional
            Window size for the median baseline/rednoise filter, by default 1.0 seconds
        norm_scale_method : str, optional
            Method for normalising the time series when using ``zscore``, by default
            "iqr". Other options are "std" and "mad".
            See sigpyproc.timeseries.TimeSeries.normalise for more details.
        start_sample : int, optional
            Starting sample index for subinterval slicing, by default 0.
        nsamps : int | None, optional
            Number of samples to read from ``start_sample`` (``None`` reads to EOF).
        preprocess_method : {"zscore", "robust"}, optional
            Preprocessing method: legacy ``"zscore"`` (default, backward-compatible)
            or ``"robust"`` dual-array inverse-variance ``(ts_e, ts_v)``.
        preprocess_options : PreprocessOptions | None, optional
            Explicit preprocessing options. When provided, overrides
            ``preprocess_method``, ``filter_window``, and ``norm_scale_method``.

        Returns
        -------
        TimeSeries
            A TimeSeries object with populated ``metadata``.
        """
        if start_sample < 0:
            msg = f"start_sample must be >= 0, got {start_sample}"
            raise ValueError(msg)
        if nsamps is not None and nsamps <= 0:
            msg = f"nsamps must be > 0, got {nsamps}"
            raise ValueError(msg)
        if tim_type == "dat":
            tim_load = sigTimeSeries.from_dat(timfile)
        elif tim_type == "tim":
            tim_load = sigTimeSeries.from_tim(timfile)
        else:
            msg = f"Invalid tim type: {tim_type}"
            raise ValueError(msg)

        header = tim_load.header
        dt = float(header.tsamp)
        total_nsamps = len(tim_load.data)
        if start_sample >= total_nsamps:
            msg = f"start_sample ({start_sample}) exceeds total nsamps ({total_nsamps})"
            raise ValueError(msg)
        stop_sample = (
            total_nsamps if nsamps is None else min(start_sample + nsamps, total_nsamps)
        )

        raw_dm = getattr(header, "dm", None)
        raw_mjd = getattr(header, "tstart", None)
        raw_bary = getattr(header, "barycentric", None)
        metadata = ObservationMetadata(
            dm=float(raw_dm) if raw_dm is not None and float(raw_dm) >= 0.0 else None,
            start_mjd=(
                float(raw_mjd) if raw_mjd is not None and float(raw_mjd) > 0.0 else None
            ),
            barycentric=bool(raw_bary) if raw_bary is not None else None,
            source=str(getattr(header, "source", "") or ""),
            input_path=str(timfile),
            original_nsamps=int(getattr(header, "nsamples", total_nsamps)),
        ).shifted(start_sample, dt)

        if preprocess_options is not None or preprocess_method == "robust":
            opts = (
                preprocess_options
                if preprocess_options is not None
                else PreprocessOptions(method="robust", filter_window=filter_window)
            )
            sliced = np.asarray(tim_load.data[start_sample:stop_sample])
            ts_e, ts_v, _ = preprocess_timeseries(sliced, dt, options=opts)
            return cls(ts_e, ts_v, dt, metadata=metadata)

        if preprocess_method != "zscore":
            msg = f"Invalid preprocess_method: {preprocess_method!r}"
            raise ValueError(msg)

        if start_sample > 0 or stop_sample < total_nsamps:
            tim_load = sigTimeSeries(
                tim_load.data[start_sample:stop_sample],
                header,
            )
        tim_load = tim_load.deredden(method="median", window=filter_window, fast=True)
        tim_load = tim_load.normalise(loc_method="mean", scale_method=norm_scale_method)  # ty:ignore[invalid-argument-type]
        signal = tim_load.data
        return cls(signal, np.ones_like(signal), dt, metadata=metadata)

    def __str__(self) -> str:
        name = type(self).__name__
        return (
            f"{name} {{nsamps = {self.nsamps:d}, tsamp = {self.dt:.4e}, "
            f"tobs = {self.tobs:.3f}}}"
        )

    def __repr__(self) -> str:
        return str(self)


@attrs.frozen(auto_attribs=True, kw_only=True)
class SubbandFoldResult:
    """2D ``(freq_subband x phase)`` dispersive waterfall and pulsar spectrum."""

    subband_folds: npt.NDArray[np.float64]
    subband_snr_map: npt.NDArray[np.float64]
    subband_freqs_mhz: npt.NDArray[np.float64]
    integrated_profile: npt.NDArray[np.float64]
    spectrum: npt.NDArray[np.float64]
    dm: float


def fold_subbands(
    source: str | Path | npt.NDArray[np.floating],
    dm: float,
    phase_func: Callable[[np.ndarray], np.ndarray],
    *,
    n_subbands: int = 32,
    n_bins: int = 64,
    tsamp: float | None = None,
    chan_freqs_mhz: npt.ArrayLike | None = None,
    ref_freq_mhz: float | None = None,
    zap_channels: Sequence[int] | None = None,
    start_sample: int = 0,
    nsamps: int | None = None,
) -> SubbandFoldResult:
    """Fold filterbank channels into a 2D ``(freq_subband x phase)`` waterfall.

    Computes exact per-channel dedispersion time offsets without cyclic wrap
    (skipping out-of-range shifted samples), folds each sub-band along
    ``phase_func(t)``, and extracts both the per-subband normalized S/N map
    and the integrated on-pulse pulsar spectrum across frequency.

    Parameters
    ----------
    source : str | Path | ndarray
        Path to a SIGPROC filterbank file or an in-memory 2D array of shape
        ``(n_chan, n_samp)``.
    dm : float
        Dispersion measure in pc cm^-3.
    phase_func : callable
        Vectorized function mapping timestamps ``t`` (seconds) to pulse phase
        in cycles.
    n_subbands : int, optional
        Number of frequency sub-bands, by default 32.
    n_bins : int, optional
        Number of pulse phase bins, by default 64.
    tsamp : float | None, optional
        Sampling interval in seconds (required when ``source`` is an array).
    chan_freqs_mhz : array_like | None, optional
        Channel center frequencies in MHz (required when ``source`` is an array).
    ref_freq_mhz : float | None, optional
        Reference frequency for dedispersion delays (defaults to highest channel).
    zap_channels : sequence of int | None, optional
        Channel indices to exclude before folding.
    start_sample : int, optional
        First time sample to read when ``source`` is a file, by default 0.
    nsamps : int | None, optional
        Number of time samples to read when ``source`` is a file.
    """
    if isinstance(source, (str, Path)):
        reader = FilReader(str(source))
        hdr = reader.header
        total = int(hdr.nsamples)
        count = (
            total - start_sample
            if nsamps is None
            else min(nsamps, total - start_sample)
        )
        block = reader.read_block(start_sample, count)
        data = np.ascontiguousarray(block.data, dtype=np.float32)
        dt = float(hdr.tsamp)
        freqs = np.asarray(hdr.chan_freqs, dtype=np.float64)
    else:
        data = np.ascontiguousarray(source, dtype=np.float32)
        if data.ndim != 2:
            msg = f"Expected 2D (n_chan, n_samp) array, got shape {data.shape}"
            raise ValueError(msg)
        if tsamp is None or tsamp <= 0.0:
            msg = "tsamp must be positive when folding an in-memory array"
            raise ValueError(msg)
        if chan_freqs_mhz is None:
            msg = "chan_freqs_mhz must be provided when folding an in-memory array"
            raise ValueError(msg)
        dt = float(tsamp)
        freqs = np.asarray(chan_freqs_mhz, dtype=np.float64)
        if freqs.shape != (data.shape[0],):
            msg = f"chan_freqs_mhz length ({freqs.size}) != n_chan ({data.shape[0]})"
            raise ValueError(msg)

    n_chan, n_samp = data.shape
    if n_subbands < 1 or n_bins < 2:
        msg = "n_subbands must be >= 1 and n_bins >= 2"
        raise ValueError(msg)
    eff_sub = min(n_subbands, n_chan)

    chan_valid = np.ones(n_chan, dtype=np.bool_)
    if zap_channels is not None:
        for ch in zap_channels:
            if 0 <= int(ch) < n_chan:
                chan_valid[int(ch)] = False

    chan_subband = np.floor(
        np.arange(n_chan, dtype=np.float64) * eff_sub / float(n_chan),
    ).astype(np.int64)
    chan_subband = np.clip(chan_subband, 0, eff_sub - 1)

    nu_ref = float(np.max(freqs)) if ref_freq_mhz is None else float(ref_freq_mhz)
    delays_s = K_DM_CONST * float(dm) * (freqs**-2 - nu_ref**-2)
    delay_samps = np.rint(delays_s / dt).astype(np.int64)

    times = np.arange(n_samp, dtype=np.float64) * dt
    phases = np.mod(np.asarray(phase_func(times), dtype=np.float64), 1.0)
    sample_bins = np.floor(phases * n_bins).astype(np.int64) % n_bins

    folds = fold_subbands_kernel(
        data,
        chan_valid,
        chan_subband,
        delay_samps,
        sample_bins,
        eff_sub,
        n_bins,
    )

    snr_map = np.zeros((eff_sub, n_bins), dtype=np.float64)
    sub_freqs = np.zeros(eff_sub, dtype=np.float64)
    for s in range(eff_sub):
        sub_mask = (chan_subband == s) & chan_valid
        if np.any(sub_mask):
            sub_freqs[s] = float(np.mean(freqs[sub_mask]))
        else:
            sub_freqs[s] = float(np.mean(freqs[chan_subband == s]))
        v = folds[s, 1]
        e = folds[s, 0]
        valid = v > 0.0
        if np.any(valid):
            v_sum = float(np.sum(v[valid]))
            e_sum = float(np.sum(e[valid]))
            ratio = e_sum / v_sum if v_sum > 0.0 else 0.0
            snr_map[s, valid] = (e[valid] - ratio * v[valid]) / np.sqrt(v[valid])

    tot_e = np.sum(folds[:, 0, :], axis=0)
    tot_v = np.sum(folds[:, 1, :], axis=0)
    int_prof = np.zeros(n_bins, dtype=np.float64)
    valid_tot = tot_v > 0.0
    if np.any(valid_tot):
        ratio_tot = float(np.sum(tot_e[valid_tot])) / float(np.sum(tot_v[valid_tot]))
        int_prof[valid_tot] = (
            tot_e[valid_tot] - ratio_tot * tot_v[valid_tot]
        ) / np.sqrt(tot_v[valid_tot])

    # Extract on-pulse spectrum across sub-bands using matched weighting from int_prof
    pos_weights = np.maximum(int_prof, 0.0)
    w_norm = float(np.linalg.norm(pos_weights))
    if w_norm > 0.0:
        weights = pos_weights / w_norm
        spectrum = snr_map @ weights
    else:
        spectrum = np.zeros(eff_sub, dtype=np.float64)

    return SubbandFoldResult(
        subband_folds=folds,
        subband_snr_map=snr_map,
        subband_freqs_mhz=sub_freqs,
        integrated_profile=int_prof,
        spectrum=spectrum,
        dm=float(dm),
    )
