from __future__ import annotations
import numpy as np

from pruning import kernels, simulate, baseplot
from pruning.utils import Spyden
from pruning.scores import boxcar_snr_1d

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def generate_signal(params, shape="gaussian", phi0=0.5, stdnoise=None):
    pulse = simulate.PulseShape(
        params.proper_time,
        params.dt,
        params.period,
        shape=shape,
        width=params.ducy,
        pos=phi0,
    )
    signal = pulse.generate()
    if stdnoise is None:
        stdnoise = np.sqrt(params.nsamps * params.ducy) / params.snr / params.tol
    signal += np.random.normal(0, stdnoise, params.nsamps)
    signal_v = np.ones(params.nsamps) * stdnoise**2
    ts_data = TSData(signal, signal_v, params.dt)
    return PulsarSignal(ts_data, params)


class SignalParams(object):
    """A pulsar signal generator.

    Parameters
    ----------
    period : float
        Period of the pulsar.
    dt : float
        Time resolution of the generated time series
    nsamps : int, optional
        Number of samples in the generated time series, by default 2**21
    snr : float, optional
        Signal to noise ratio of the generated time series, by default 200
    ducy : float, optional
        Duty cycle of the pulse (FWTM) in phase units, by default 0.1
    over_sampling : int, optional
        Over sampling factor, by default 2
    """

    def __init__(
        self,
        period: float,
        dt: float,
        nsamps: int = 2097152,
        snr: float = 200,
        ducy: float = 0.1,
        over_sampling: int = 7,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
    ) -> None:
        self.period = period
        self.dt = dt
        self.nsamps = nsamps
        self.snr = snr
        self.ducy = ducy
        self.over_sampling = over_sampling

        self._set_mod_func(mod_type, mod_kwargs)
        self._check_params()
        print(f"ducy = {self.ducy:.3f}, over_sampling = {self.over_sampling:d}")
        print(f"Tol = {self.tol}, fold bins = {self.fold_bins:d}")

    @property
    def freq(self) -> float:
        """Frequency of the pulsar."""
        return 1 / self.period

    @property
    def tobs(self) -> float:
        """Total observation time."""
        return self.nsamps * self.dt

    @property
    def fold_bins(self) -> int:
        """Number of phase bins in the folded profile."""
        return int(self.over_sampling / self.ducy)

    @property
    def tol(self) -> int:
        """Number of time bins across the pulse duty cycle."""
        return self.period * self.ducy / self.dt

    @property
    def mod_func(self) -> simulate.DerivativeModulating:
        return self._mod_func

    @property
    def proper_time(self) -> np.ndarray:
        return self.mod_func.generate(np.arange(0, self.tobs, self.dt))

    def _set_mod_func(self, mod_type="derivative", mod_kwargs: dict | None = None):
        type_to_mods = {
            "derivative": simulate.DerivativeModulating,
            "circular": simulate.CircularModulating,
            "keplerian": simulate.KeplerianModulating,
        }
        if mod_kwargs is None:
            mod_kwargs = {}
        self._mod_func = type_to_mods[mod_type](**mod_kwargs)

    def _check_params(self):
        if self.ducy > 1:
            raise ValueError("Duty cycle should be less than 1")
        if self.tol < 1:
            raise ValueError(
                "Pulse bin width is shorter than sampling time. FWTM should be at least 1 time bin."
            )
        if self.fold_bins > int(self.period / self.dt):
            raise ValueError(
                "Fold bins exceed the number of bins in the pulse period. Decrease over_sampling or increase ducy."
            )


class TSData(object):
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

    def downsample(self, factor: int) -> None:
        self._ts_e = kernels.downsample_1d(self.ts_e, factor)
        self._ts_v = kernels.downsample_1d(self.ts_v, factor)
        self._dt *= factor

    def get_chunk_len(self, p_max, init_levels=1):
        levels = int(np.log2(self.tobs / p_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def resample(self, accel) -> TSData:
        ts_e, ts_v = kernels.resample(self.ts_e, self.ts_v, self.dt, accel)
        return TSData(ts_e, ts_v, self.dt)

    def fold_subints(self, freq, nbins, nsubints=None):
        cycles_total = int(self.tobs * freq)
        tbin = 1 / (freq * nbins)
        if cycles_total < 1:
            raise ValueError(f"Period ({1/freq}) exceeds total data length ({self.tobs})")

        if tbin < self.dt:
            raise ValueError("Bin width is shorter than sampling time")

        if nsubints is None:
            nsubints = cycles_total
        else:
            if nsubints < 1:
                raise ValueError("subints must be >= 1 or None")
            if nsubints > cycles_total:
                raise ValueError(
                    f"subints ({nsubints}) exceeds the number of signal cycles that fit in the data ({cycles_total})"
                )
        proper_time = np.arange(0, self.tobs, self.dt)
        ind_arr = kernels.get_phase_idx(proper_time, freq, nbins, 0)
        ind_arrs = np.array([ind_arr])
        fold = kernels.fold_ts(self.ts_e, self.ts_v, ind_arrs, nbins, nsubints).squeeze()
        return fold[:, 0, :] / np.sqrt(fold[:, 1, :])

    def ephemeris_fold(
        self, mod_func: simulate.DerivativeModulating, freq: float, nbins: float
    ) -> np.ndarray:
        proper_time = mod_func.generate(np.arange(0, self.tobs, self.dt))
        ind_arr = kernels.get_phase_idx(proper_time, freq, nbins, 0)
        ephemeris_fold = self.fold(ind_arr, nbins)
        return ephemeris_fold[0] / np.sqrt(ephemeris_fold[1])

    def fold(self, proper_ind_arr, nbins):
        ind_arrs = np.array([proper_ind_arr])
        fold = kernels.fold_ts(self.ts_e, self.ts_v, ind_arrs, nbins, 1)
        return fold.squeeze()

    def __str__(self):
        name = type(self).__name__
        return f"{name} {{nsamps = {self.nsamps:d}, tsamp = {self.dt:.4e}, tobs = {self.tobs:.3f}}}"

    def __repr__(self):
        return str(self)


class PulsarSignal(object):
    def __init__(self, ts_data: TSData, params: SignalParams) -> None:
        self._ts_data = ts_data
        self._params = params

        self._ephemeris_fold = ts_data.ephemeris_fold(
            params.mod_func, params.freq, params.fold_bins
        )
        self._subint_fold = ts_data.fold_subints(
            params.freq, params.fold_bins, nsubints=64
        )

    @property
    def ts_data(self) -> TSData:
        return self._ts_data

    @property
    def params(self) -> SignalParams:
        return self._params

    @property
    def ephemeris(self) -> np.ndarray:
        return self._ephemeris

    @property
    def ephemeris_fold(self) -> np.ndarray:
        return self._ephemeris_fold

    @property
    def subint_fold(self):
        return self._subint_fold

    def plot(self, figsize=(10, 6.5), dpi=100, cmap="magma_r"):
        spyden_boxcar = Spyden(
            self.ephemeris_fold,
            tempwidth_max=self.params.fold_bins // 2,
            template_kind="boxcar",
        )
        match_boxcar = boxcar_snr_1d(
            self.ephemeris_fold, np.arange(1, self.params.fold_bins // 2)
        )

        figure = plt.figure(figsize=figsize, dpi=dpi)
        grid = figure.add_gridspec(
            nrows=2, ncols=2, height_ratios=(1.5, 1), width_ratios=(1, 1.5)
        )
        grid.update(left=0.1, right=0.98, bottom=0.08, top=0.95, hspace=0.2)
        axtable = plt.subplot(grid[0, 0])
        axsubints = plt.subplot(grid[0, 1])
        axprofile = plt.subplot(grid[1, :])

        ducy = spyden_boxcar.best_width / len(self.ephemeris_fold)
        table = baseplot.Table(
            col_off=[0.01, 0.75], top_margin=0.1, line_height=0.12, fontsize=12
        )
        table.add_row(["Tsamp", f"{self.params.dt*1e3:.3f}"])
        table.add_row(["Period", self.params.period])
        table.add_row(["Accel", self.params.mod_func.acc])
        table.add_row(["Jerk", self.params.mod_func.jerk])
        table.add_row(["Snap", self.params.mod_func.snap])
        table.add_row(["S/N", self.params.snr])
        table.add_row(["Width", spyden_boxcar.best_width])
        table.add_row(["Ducy", f"{ducy:.3f}"])
        table.plot(axtable)

        nsubints, nbins = self.subint_fold.shape
        axsubints.imshow(
            self.subint_fold,
            aspect="auto",
            interpolation="none",
            extent=[0, nbins, 0, self.ts_data.tobs],
            cmap=plt.get_cmap(cmap),
            origin="lower",
        )
        axsubints.set_ylabel("Time (seconds)")
        axsubints.set_xlabel("Phase bin")
        axprofile.plot(
            range(nbins), self.ephemeris_fold, color="#404040", label="Folded Profile"
        )
        axprofile.plot(
            range(nbins),
            spyden_boxcar.best_model,
            color="#d62728",
            alpha=0.65,
            ls="--",
            label="Boxcar template",
        )
        axprofile.set_xlim(0, nbins)
        axprofile.set_xlabel("Phase bin")
        axprofile.set_ylabel("Normalised amplitude")

        textstr = (
            f"S/N (Boxcar) = {spyden_boxcar.snr:.2f}\n"
            f"S/N (Boxcar, 1D) = {match_boxcar.max():.2f}\n"
        )

        at = AnchoredText(textstr, loc="upper left", prop={"size": 10}, frameon=True)
        at.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
        axprofile.add_artist(at)
        axprofile.legend(loc="upper right")
        return figure
