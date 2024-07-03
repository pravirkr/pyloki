from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

from pruning import baseplot, kernels, simulate

if TYPE_CHECKING:
    from matplotlib import pyplot as plt


@attrs.define(auto_attribs=True, kw_only=True)
class SignalConfig:
    """A pulsar signal generator configuration.

    Parameters
    ----------
    period : float
        Period of the pulsar.
    dt : float
        Time resolution of the generated time series
    nsamps : int, optional
        Number of samples in the generated time series, by default 2**21
    snr : float, optional
        Signal to noise ratio of the folded pulse profile, by default 100
    ducy : float, optional
        Duty cycle of the pulse (FWTM) in fractional phase, by default 0.1
        For slow pulsars, ducy ~ 0.03, for millisecond pulsars, ducy ~ 0.1 - 0.3
    over_sampling : int, optional
        Over sampling factor for the folded phase bins, by default 1
    mod_type : str, optional
        Type of modulation, by default "derivative"
    mod_kwargs : dict, optional
        Modulation function parameters, by default None
    """

    period: float
    dt: float
    nsamps: int = 2**21
    snr: float = 100
    ducy: float = 0.1
    over_sampling: float = 1.0
    mod_type: str = "derivative"
    mod_kwargs: dict | None = None
    _mod_func: simulate.DerivativeModulating = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._set_mod_func(self.mod_type, self.mod_kwargs)
        self._check()

    @property
    def freq(self) -> float:
        """Pulsar frequency."""
        return 1 / self.period

    @property
    def tobs(self) -> float:
        """Total observation time."""
        return self.nsamps * self.dt

    @property
    def tol_bins(self) -> float:
        """Pulsar pulse width in bins."""
        return self.ducy * self.period / self.dt

    @property
    def fold_bins(self) -> int:
        """Number of phase bins in the folded profile."""
        return int(self.period / self.dt / self.over_sampling)

    @property
    def fold_bins_ideal(self) -> int:
        """Number of ideal phase bins in the folded profile."""
        return int(self.period / self.dt)

    @property
    def mod_func(self) -> simulate.DerivativeModulating:
        """Modulation function."""
        return self._mod_func

    @property
    def proper_time(self) -> np.ndarray:
        """Proper time array."""
        return self.mod_func.generate(np.arange(0, self.tobs, self.dt), self.tobs / 2)

    def get_updated(self, update_dict: dict) -> SignalConfig:
        new = attrs.asdict(self, filter=attrs.filters.exclude("_mod_func"))
        if update_dict is not None:
            new.update(update_dict)
        new_checked = {
            key: value for key, value in new.items() if key in attrs.asdict(self)
        }
        return SignalConfig(**new_checked)

    def _set_mod_func(
        self,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
    ) -> None:
        if mod_kwargs is None:
            mod_kwargs = {}
        self._mod_func = simulate.type_to_mods[mod_type](**mod_kwargs)

    def _check(self) -> None:
        if self.ducy <= 0 or self.ducy >= 1:
            msg = f"Duty cycle ({self.ducy}) should be in (0, 1)"
            raise ValueError(msg)


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
        ts_e = kernels.downsample_1d(self.ts_e, factor)
        ts_v = kernels.downsample_1d(self.ts_v, factor)
        return TimeSeries(ts_e, ts_v, self.dt)

    def resample(self, accel: float) -> TimeSeries:
        ts_e, ts_v = kernels.resample(self.ts_e, self.ts_v, self.dt, accel)
        return TimeSeries(ts_e, ts_v, self.dt)

    def fold_ephem(
        self,
        freq: float,
        nbins: int,
        nsubints: int = 1,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
    ) -> np.ndarray:
        cycles = int(self.tobs * freq)
        if cycles < 1:
            msg = f"Period ({1/freq}) is less than the total data length ({self.tobs})"
            raise ValueError(msg)
        if nsubints < 1 or nsubints > cycles:
            msg = f"subints must be >= 1 and <= {cycles}"
            raise ValueError(msg)
        if mod_kwargs is None:
            mod_kwargs = {}
        mod_func = simulate.type_to_mods[mod_type](**mod_kwargs)
        proper_time = mod_func.generate(np.arange(0, self.tobs, self.dt), self.tobs / 2)
        ind_arr = kernels.get_phase_idx(proper_time, freq, nbins, 0)
        return self.fold(ind_arr, nbins, nsubints)

    def plot_fold(
        self,
        freq: float,
        fold_bins: int,
        nsubints: int = 64,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
    ) -> plt.Figure:
        ephem_fold = self.fold_ephem(
            freq,
            fold_bins,
            nsubints=1,
            mod_type=mod_type,
            mod_kwargs=mod_kwargs,
        )
        ephem_fold_subints = self.fold_ephem(freq, fold_bins, nsubints=nsubints)
        return baseplot.fold_ephemeris_plot(
            ephem_fold,
            ephem_fold_subints,
            freq,
            self.dt,
            self.tobs,
            mod_kwargs=mod_kwargs,
        )

    def fold(
        self,
        ind_arr: np.ndarray,
        nbins: int,
        nsubints: int = 1,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        ind_arrs = np.array([ind_arr])
        fold = kernels.fold_ts(
            self.ts_e,
            self.ts_v,
            ind_arrs,
            nbins,
            nsubints,
        ).squeeze()
        if normalize:
            return np.divide(
                fold[..., 0, :],
                np.sqrt(fold[..., 1, :]),
                out=np.full_like(fold[..., 0, :], np.nan),
                where=~np.isclose(fold[..., 1, :], 0, atol=1e-5),
            )

        return fold

    def __str__(self) -> str:
        name = type(self).__name__
        return (
            f"{name} {{nsamps = {self.nsamps:d}, tsamp = {self.dt:.4e}, "
            f"tobs = {self.tobs:.3f}}}"
        )

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def generate_signal(
        cls,
        cfg: SignalConfig,
        shape: str = "gaussian",
        phi0: float = 0.5,
    ) -> TimeSeries:
        pulse = simulate.PulseShape(
            cfg.proper_time,
            cfg.dt,
            cfg.period,
            shape=shape,
            width=cfg.ducy,
            pos=phi0,
        )
        signal = pulse.generate()
        stdnoise = np.sqrt(cfg.nsamps * cfg.ducy) / cfg.snr / cfg.tol_bins
        rng = np.random.default_rng()
        signal += rng.normal(0, stdnoise, cfg.nsamps)
        signal_v = np.ones(cfg.nsamps) * stdnoise**2
        return TimeSeries(signal, signal_v, cfg.dt)
