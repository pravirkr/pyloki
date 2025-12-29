from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sigpyproc.core import filters as sig_filters
from sigpyproc.timeseries import TimeSeries as sigTimeSeries
from sigpyproc.viz.styles import PlotTable

from pyloki.core import brutefold
from pyloki.detection.scoring import boxcar_snr_1d
from pyloki.simulation.modulate import type_to_mods
from pyloki.utils import np_utils


class TimeSeries:
    def __init__(self, ts_e: np.ndarray, ts_v: np.ndarray, dt: float) -> None:
        self._ts_e = np.asarray(ts_e, dtype=np.float32)
        self._ts_v = np.asarray(ts_v, dtype=np.float32)
        self._dt = dt

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
    def nsamps(self) -> int:
        return len(self.ts_e)

    @property
    def tobs(self) -> float:
        return self.nsamps * self.dt

    def downsample(self, factor: int) -> TimeSeries:
        ts_e = np_utils.downsample_1d(self.ts_e, factor)
        ts_v = np_utils.downsample_1d(self.ts_v, factor)
        return TimeSeries(ts_e, ts_v, self.dt)

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
                f"Period ({1 / freq}) is less than the total data length ({self.tobs})"
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
        proper_time = mod_func.generate(np.arange(0, self.tobs, self.dt), mod_tref)
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
    def from_tim(cls, timfile: str, tim_type: str = "dat") -> TimeSeries:
        if tim_type == "dat":
            tim_load = sigTimeSeries.from_dat(timfile)
        elif tim_type == "tim":
            tim_load = sigTimeSeries.from_tim(timfile)
        else:
            msg = f"Invalid tim type: {tim_type}"
            raise ValueError(msg)
        tim_load = tim_load.deredden(method="median", window=4.0, fast=True)
        tim_load = tim_load.normalise()
        signal = tim_load.data
        return cls(signal, np.ones_like(signal), tim_load.header.tsamp)

    def __str__(self) -> str:
        name = type(self).__name__
        return (
            f"{name} {{nsamps = {self.nsamps:d}, tsamp = {self.dt:.4e}, "
            f"tobs = {self.tobs:.3f}}}"
        )

    def __repr__(self) -> str:
        return str(self)
